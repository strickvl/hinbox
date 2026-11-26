"""Training and test data for a tuned merge check.

The merge check asks whether a name from a new article ("Rubio", with what
the article says about him) is the same entity as a page already in the
tables ("Marco Rubio", with its profile). A wrong "yes" folds two people
into one page; a wrong "no" makes a duplicate page.

Three sources of labelled pairs:

- **Hand labels** (``merge_gold.jsonl``, ``merge_gold_v2.jsonl``, read by
  ``load_merge_items()``). Pairs in their ``test`` split are the score and
  never go into training.
- **Split pages** (``split_positives()``), known "yes" without any judge:
  take a page citing several articles, cut out the sentences that cite only
  one of them as the "new article" evidence, keep the rest as the existing
  page, and (for people) shorten the new name the way articles do ("Ann B.
  Lee" -> "Ann Lee" or "Lee"). It is the same entity by construction.
- **Teacher labels** (``teach_pairs()``): mined look-alike pairs judged by two
  strong cloud models, kept where both agree (and agree with the hand label
  or split-page truth when there is one), with the first teacher's
  reasoning as the answer to learn.

Examples use the pipeline's own sectioned messages
(``_build_sectioned_messages()``, the existing page cut by ``profile_view()``
as ``_check_match()`` cuts it) and its reply layout, so a tuned model drops in
as ``HINBOX_MATCH_MODEL``.
"""

import random
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from src.citecheck.items import append_jsonl, read_jsonl
from src.constants import MATCH_PROFILE_CHARS
from src.engine.match_checker import (
    _build_sectioned_messages,
    check_failed,
    cloud_model_check_match,
    parse_sectioned_reply,
)
from src.engine.profile_view import profile_view
from src.evals.datasets import assign_split, load_merge_items
from src.evals.metrics import binary_metrics
from src.logging_config import get_logger
from src.utils.cache_utils import sha256_text
from src.utils.grounding import _extract_cited_claims

logger = get_logger("citecheck.merge")

MIN_NEW_CHARS, MIN_EXISTING_CHARS = 80, 200
_TITLES = {"mr", "mrs", "ms", "dr", "col", "gen", "lt", "capt", "maj", "sgt", "judge"}


def gold_pairs(eval_dir: str) -> List[Dict[str, Any]]:
    """Hand-labelled pairs from both gold sets, with their split."""
    rows = []
    for merge_set in ("v1", "v2"):
        try:
            items = load_merge_items(eval_dir, merge_set)
        except FileNotFoundError:
            continue
        for it in items:
            rows.append(
                {
                    "item_id": sha256_text(it.item_id)[:16],
                    "inputs": it.inputs,
                    "truth": bool(it.label),
                    "split": it.split or assign_split(it.item_id),
                    "source": f"gold_{merge_set}",
                }
            )
    return rows


def shorter_name(name: str, rng: random.Random) -> str:
    """A shorter way an article might name a person: middle names dropped,
    or the surname alone; the name itself when it has no spare words."""
    words = [w for w in name.split() if w.lower().strip(".") not in _TITLES]
    if len(words) >= 3:
        return f"{words[0]} {words[-1]}" if rng.random() < 0.6 else words[-1]
    if len(words) == 2 and len(words[-1]) > 3 and rng.random() < 0.4:
        return words[-1]
    return name


def _cited(text: str) -> List[Tuple[str, List[str]]]:
    """(claim with its marker, cited ids) per marker in a profile."""
    out: List[Tuple[str, List[str]]] = []
    by_citation: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    order: List[Tuple[str, str]] = []
    for c in _extract_cited_claims(text):
        key = (c["claim"], c["citation"])
        if key not in by_citation:
            order.append(key)
        by_citation[key].append(c["article_id"])
    for claim, citation in order:
        out.append((f"{claim}{citation}", by_citation[(claim, citation)]))
    return out


def split_positives(
    entities: Dict[str, Dict],
    n: int,
    seed: int = 5,
    avoid_names: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """Up to ``n`` "same entity" pairs made by cutting pages in two (see the
    module docstring). ``avoid_names`` (names in hand-labelled test pairs)
    are skipped, so the test pairs' entities are not trained on."""
    rng = random.Random(seed)
    avoid = {a.lower() for a in (avoid_names or set())}
    pool = []
    for table, entries in entities.items():
        for data in entries.values():
            profile = data.get("profile") or {}
            if profile.get("linked_only") or profile.get("stub"):
                continue
            name = str(data.get("name") or data.get("title") or "")
            if name and name.lower() not in avoid:
                pool.append((table, name, profile.get("text") or ""))
    rng.shuffle(pool)
    out: List[Dict[str, Any]] = []
    for table, name, text in pool:
        if len(out) >= n:
            break
        claims = _cited(text)
        ids = sorted({i for _, cited in claims for i in cited})
        if len(ids) < 2:
            continue
        article = rng.choice(ids)
        new = [c for c, cited in claims if cited == [article]]
        rest = [c for c, cited in claims if article not in cited]
        new_text, existing_text = " ".join(new), "\n\n".join(rest)
        if len(new_text) < MIN_NEW_CHARS or len(existing_text) < MIN_EXISTING_CHARS:
            continue
        new_name = shorter_name(name, rng) if table == "people" else name
        out.append(
            {
                "item_id": sha256_text(f"split|{table}|{name}|{article}")[:16],
                "inputs": {
                    "new_name": new_name,
                    "existing_name": name,
                    "new_profile_text": new_text,
                    "existing_profile_text": existing_text,
                    "entity_type": table,
                },
                "truth": True,
                "split": "train",
                "source": "split_page",
            }
        )
    return out


def match_messages(inputs: Dict[str, Any]) -> List[Dict[str, str]]:
    """The pipeline's match-check messages for one pair."""
    existing = profile_view(
        inputs["existing_profile_text"], inputs["new_profile_text"], MATCH_PROFILE_CHARS
    )
    return _build_sectioned_messages(
        inputs["new_name"],
        inputs["existing_name"],
        inputs["new_profile_text"],
        existing,
        inputs["entity_type"],
    )


def match_reply(reasoning: str, is_match: bool, confidence: float) -> str:
    """A reply in the sectioned layout ``parse_sectioned_reply()`` reads."""
    return (
        f"[[ ## reasoning ## ]]\n{reasoning.strip()}\n\n"
        f"[[ ## is_match ## ]]\n{is_match}\n\n"
        f"[[ ## confidence ## ]]\n{confidence:.2f}\n\n"
        "[[ ## completed ## ]]"
    )


def teach_pairs(
    pairs: List[Dict[str, Any]], teachers: Tuple[str, ...], out_path: str
) -> Dict[str, int]:
    """Ask each teacher about each pair (skipping answers already in
    ``out_path``) and append ``{item_id, teacher, is_match, confidence,
    reasoning}`` rows."""
    done = {(r["teacher"], r["item_id"]) for r in read_jsonl(out_path)}
    counts = {"asked": 0, "failed": 0}
    for teacher in teachers:
        for p in pairs:
            if (teacher, p["item_id"]) in done:
                continue
            counts["asked"] += 1
            result = cloud_model_check_match(**p["inputs"], model=teacher)
            if check_failed(result):
                counts["failed"] += 1
                continue
            append_jsonl(
                out_path,
                {
                    "item_id": p["item_id"],
                    "teacher": teacher,
                    "is_match": result.is_match,
                    "confidence": result.confidence,
                    "reasoning": result.reason,
                },
            )
    return counts


def agreed_pairs(
    pairs: List[Dict[str, Any]],
    teacher_rows: List[Dict[str, Any]],
    teachers: Tuple[str, ...],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Pairs every teacher answered alike, and alike with the pair's known
    truth if it has one; each gets the first teacher's reasoning and the
    lowest confidence."""
    answers: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for r in teacher_rows:
        answers[r["item_id"]][r["teacher"]] = r
    out, dropped = [], defaultdict(int)
    for p in pairs:
        got = answers.get(p["item_id"], {})
        if any(t not in got for t in teachers):
            dropped["missing"] += 1
            continue
        votes = {got[t]["is_match"] for t in teachers}
        if len(votes) != 1:
            dropped["disagree"] += 1
            continue
        vote = votes.pop()
        if p.get("truth") is not None and vote != p["truth"]:
            dropped["against_truth"] += 1
            continue
        out.append(
            {
                **p,
                "is_match": vote,
                "confidence": min(float(got[t]["confidence"]) for t in teachers),
                "reasoning": got[teachers[0]]["reasoning"],
            }
        )
    return out, dict(dropped)


def merge_examples(rows: List[Dict[str, Any]], seed: int = 9) -> List[Dict[str, Any]]:
    """``{"messages": [...]}`` training rows; test-split pairs are refused."""
    out = []
    for r in rows:
        if r.get("split") == "test":
            raise ValueError(f"pair {r['item_id']} is in the test split")
        messages = match_messages(r["inputs"])
        messages.append(
            {
                "role": "assistant",
                "content": match_reply(r["reasoning"], r["is_match"], r["confidence"]),
            }
        )
        out.append({"messages": messages})
    random.Random(seed).shuffle(out)
    return out


def merge_prompts(pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """``{"item_ids": [id], "messages"}`` per pair, for sampling on Tinker."""
    return [
        {"item_ids": [p["item_id"]], "messages": match_messages(p["inputs"])}
        for p in pairs
    ]


_THINK = re.compile(r"<think>.*?</think>", re.S)


def score_replies(
    pairs: List[Dict[str, Any]], replies: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Scoreboard metrics for raw replies against the pairs' truth; an
    unreadable reply counts as an error."""
    predicted: Dict[str, Optional[bool]] = {}
    for row in replies:
        for iid in row.get("item_ids") or []:
            try:
                predicted[iid] = parse_sectioned_reply(
                    _THINK.sub("", row.get("reply") or "")
                ).is_match
            except ValueError:
                predicted[iid] = None
    return binary_metrics([(p["truth"], predicted.get(p["item_id"])) for p in pairs])
