"""Verdicts from strong cloud judges, as training labels.

A person can label a few hundred claims; training wants tens of thousands.
So two strong judges ("teachers", by default Claude Opus and Claude Sonnet)
are asked about claims from train and tune articles with the pipeline's own
judge prompt, one call per article, and a claim becomes a training label
only when both teachers give the same verdict. Where they disagree, one of
them is wrong, and a student would learn the mistake half the time.

A "supported" or "partial" label must also come with a quote the reasoning
takes from the article that really is there (``passages_from_reasoning()``,
the same check the reader uses), since the tuned model is taught to answer
with that quote first. Before training, a person audits a sample of the
agreed labels yes or no in the labelling page, to measure how often the
teachers are wrong together.

Calls are written to the output file as they finish, so a stopped run picks
up where it left off: a (teacher, article) call already in the file is not
made again.
"""

import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

from src.citecheck.items import append_jsonl, read_jsonl
from src.logging_config import get_logger
from src.utils import grounding
from src.utils.grounding import (
    GROUNDING_MAX_ARTICLE_CHARS,
    GROUNDING_MAX_CLAIMS_PER_CALL,
    judge_claims,
    passages_from_reasoning,
)

logger = get_logger("citecheck.teach")

TEACHERS = ("anthropic/claude-opus-5-5", "anthropic/claude-sonnet-5-5")
PER_ARTICLE = 12
# Verdicts kept only with a passage found in the article. A "partial" verdict
# often names what the article lacks (a dropped "allegedly", an added cause),
# which has no passage to quote, so it is kept without one: on the 26 Nov strict
# run a quote rule on partial kept 267 of 766 agreed partials against 1,072 of
# 1,773 supported, and threw away most of the errors the strict rules teach.
QUOTED = ("supported",)


def teacher_sample(
    rows: List[Dict[str, Any]],
    have_text: set,
    n: int,
    seed: int = 11,
    splits: Tuple[str, ...] = ("train", "tune"),
) -> List[Dict[str, Any]]:
    """Up to ``n`` claims from train and tune articles with text, at most
    ``PER_ARTICLE`` per article, so many articles are covered."""
    rng = random.Random(seed)
    pool = [r for r in rows if r["split"] in splits and r["article_id"] in have_text]
    rng.shuffle(pool)
    per_article: Dict[str, int] = defaultdict(int)
    out = []
    for r in pool:
        if len(out) >= n:
            break
        if per_article[r["article_id"]] >= PER_ARTICLE:
            continue
        per_article[r["article_id"]] += 1
        out.append(r)
    return out


def article_chunks(
    claims: List[Dict[str, Any]], size: int = GROUNDING_MAX_CLAIMS_PER_CALL
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """(article id, up to ``size`` claims) per judge call, in a fixed order."""
    by_article: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for c in claims:
        by_article[c["article_id"]].append(c)
    out = []
    for aid in sorted(by_article):
        group = by_article[aid]
        out += [(aid, group[i : i + size]) for i in range(0, len(group), size)]
    return out


def estimate_tokens(
    chunks: List[Tuple[str, List[Dict[str, Any]]]], texts: Dict[str, str]
) -> int:
    """Rough prompt tokens for one teacher over ``chunks`` (4 characters per
    token, plus about 600 for the instructions)."""
    chars = sum(
        len((texts.get(aid) or "")[:GROUNDING_MAX_ARTICLE_CHARS])
        + sum(len(c["claim"]) + len(c["entity_name"]) + 8 for c in group)
        + 2400
        for aid, group in chunks
    )
    return chars // 4


def _done(out_path: str) -> set:
    return {(r["teacher"], r["item_id"]) for r in read_jsonl(out_path)}


def _ask(
    teacher: str, aid: str, group: List[Dict[str, Any]], texts: Dict[str, str]
) -> List[Dict[str, Any]]:
    seen = (texts.get(aid) or "")[:GROUNDING_MAX_ARTICLE_CHARS]
    verdicts = judge_claims(
        article_id=aid,
        article_text=seen,
        claims=[(c["entity_name"], c["claim"]) for c in group],
        model_type="cloud",
        ask_passage=False,
        model=teacher,
    )
    rows = []
    for c, v in zip(group, verdicts):
        if v is None:
            continue
        level, reasoning, _ = v
        level = getattr(level, "value", level)
        rows.append(
            {
                "item_id": c["item_id"],
                "teacher": teacher,
                "article_id": aid,
                "entity_name": c["entity_name"],
                "claim": c["claim"],
                "level": level,
                "reasoning": reasoning,
                "passages": passages_from_reasoning(seen, reasoning, c["claim"]),
                "rules": grounding.GROUNDING_RULES,
            }
        )
    return rows


def same_rules(rows: List[Dict[str, Any]], path: str, fix: str) -> None:
    """Stop when teacher ``rows`` were asked under other judge rules than
    ``HINBOX_GROUNDING_RULES`` now names (a row without ``rules`` predates
    them: standard), so verdicts never meet a prompt they weren't asked with."""
    other = {r.get("rules", "standard") for r in rows if "teacher" in r}
    other.discard(grounding.GROUNDING_RULES)
    if other:
        raise SystemExit(
            f"{path} holds verdicts under rules={sorted(other)}, not "
            f"{grounding.GROUNDING_RULES}: {fix}"
        )


def run_teachers(
    claims: List[Dict[str, Any]],
    texts: Dict[str, str],
    out_path: str,
    teachers: Tuple[str, ...] = TEACHERS,
    workers: int = 4,
) -> Dict[str, int]:
    """Ask each teacher about ``claims`` and append its verdicts to
    ``out_path``; calls already there are skipped. Returns counts. A file
    holding verdicts asked under other judge rules (``HINBOX_GROUNDING_RULES``)
    is refused, so one file never mixes two standards."""
    same_rules(read_jsonl(out_path), out_path, "use another --out")
    done = _done(out_path)
    jobs = [
        (t, aid, [c for c in group if (t, c["item_id"]) not in done])
        for t in teachers
        for aid, group in article_chunks(claims)
    ]
    jobs = [j for j in jobs if j[2]]
    counts = {"calls": len(jobs), "failed": 0, "verdicts": 0}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(_ask, t, aid, g, texts): (t, aid) for t, aid, g in jobs}
        for fut in as_completed(futures):
            teacher, aid = futures[fut]
            try:
                rows = fut.result()
            except Exception as exc:  # a failed call is asked again next run
                counts["failed"] += 1
                logger.warning("%s on %s failed: %s", teacher, aid, exc)
                continue
            for row in rows:
                append_jsonl(out_path, row)
            counts["verdicts"] += len(rows)
    return counts


def agreed_labels(
    teacher_rows: List[Dict[str, Any]], teachers: Tuple[str, ...] = TEACHERS
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Claims every teacher judged the same way, each with the reasoning of
    the first teacher that quoted the article (else the first teacher's); a
    supported label also needs such a quote. Returns the labels and counts of what was left
    out (``disagree``, ``missing``, ``no_quote``)."""
    by_item: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for r in teacher_rows:
        by_item[r["item_id"]][r["teacher"]] = r
    out, dropped = [], defaultdict(int)
    for got in by_item.values():
        if any(t not in got for t in teachers):
            dropped["missing"] += 1
            continue
        levels = {got[t]["level"] for t in teachers}
        if len(levels) != 1:
            dropped["disagree"] += 1
            continue
        level = levels.pop()
        # the first teacher whose reasoning holds a quote found in the article
        quoted = [got[t] for t in teachers if got[t].get("passages")]
        first: Dict[str, Any] = quoted[0] if quoted else got[teachers[0]]
        if level in QUOTED and not quoted:
            dropped["no_quote"] += 1
            continue
        out.append({**first, "source": "teachers", "teachers": list(teachers)})
    return out, dict(dropped)
