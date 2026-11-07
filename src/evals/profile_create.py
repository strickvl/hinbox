"""Compare ways of writing a first profile on the same (entity, article) pairs.

``just eval profile-create --model <m>`` takes entities from an entity
database (each with the article its profile was first written from), or the
first replay article of each replay entity (``--cases replay``), and runs
``create_profile()`` once per approach (``reflect`` and ``check`` by default)
on each pair. Every approach sees the same pairs, so differences are paired.

Scores, per profile, without a model:

- ``calls``: model calls it took (draft, reflections, regenerations, repairs)
- ``chars``, ``citations``, ``headings``, ``tags``
- ``wrong_ids``: citations that aren't the article's ID
- ``uncited_paragraphs``: paragraphs grounding can't check
- ``names_not_in_article``: capitalised words and numbers in the profile that
  the article doesn't contain (a rough invented-detail signal)

With ``--grounding``, ``verify_profile_grounding()`` checks each cited claim
against the article with the cloud model and reports the supported share. With
a claim-labels file (``{eval_dir}/claim_labels.jsonl``, rows with
``"use": "create"``; ``--claim-labels draft`` also uses unreviewed rows), the labelled facts found, core facts missed and
must-not hits are added (``src/evals/claim_labels.py``).

Per-profile records keep the profile text, which is article-derived: they stay
in ``{eval_dir}/profile_create/`` and must not go to GitHub.
"""

import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pyarrow.parquet as pq

from src.engine.profile_create import type_tag
from src.logging_config import console

ENTITY_TABLES = ("people", "organizations", "locations", "events")
_CITE = re.compile(r"\^\[([^\]]*)\]")
_HEADING = re.compile(r"^\s*#{2,4}\s+\S", re.MULTILINE)
_NAME_OR_NUMBER = re.compile(r"\b(?:[A-Z][a-zA-Z'’-]{2,}|\d[\d,.]*\d|\d)\b")
_SENTENCE_START = re.compile(r"(?:^|[.!?]\s+|\n)([A-Z][a-zA-Z'’-]+)")


def _first_article(row: Dict[str, Any]) -> Optional[str]:
    versions = (row.get("profile_versions") or {}).get("versions") or []
    if versions and versions[0].get("trigger_article_id"):
        return versions[0]["trigger_article_id"]
    articles = row.get("articles") or []
    first = articles[0] if articles else None
    return (first.get("article_id") if isinstance(first, dict) else first) or None


def _entity_name(row: Dict[str, Any]) -> str:
    return str(row.get("name") or row.get("title") or "")


def entity_cases(
    entities_dir: str, texts: Dict[str, str], limit: int, seed: int = 0
) -> List[Dict[str, Any]]:
    """Up to ``limit`` pairs, spread evenly over the four entity tables."""
    per_table: Dict[str, List[Dict[str, Any]]] = {}
    for table in ENTITY_TABLES:
        path = os.path.join(entities_dir, f"{table}.parquet")
        if not os.path.exists(path):
            continue
        rows = pq.read_table(path).to_pylist()
        cases = []
        for row in rows:
            aid, name = _first_article(row), _entity_name(row)
            if aid and name and texts.get(aid):
                cases.append(
                    {
                        "entity_name": name,
                        "entity_type": type_tag(table),
                        "article_id": aid,
                    }
                )
        random.Random(seed).shuffle(cases)
        per_table[table] = cases
    out: List[Dict[str, Any]] = []
    while len(out) < limit and any(per_table.values()):
        for table in ENTITY_TABLES:
            if per_table.get(table) and len(out) < limit:
                out.append(per_table[table].pop())
    return out


def replay_cases(eval_dir: str) -> List[Dict[str, Any]]:
    """The first article of each replay entity (any model's manifest)."""
    root = os.path.join(eval_dir, "profile_replay")
    seen: Dict[str, Dict[str, Any]] = {}
    for model_dir in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        for slug in sorted(os.listdir(os.path.join(root, model_dir))):
            path = os.path.join(root, model_dir, slug, "manifest.json")
            if slug in seen or not os.path.exists(path):
                continue
            with open(path, encoding="utf-8") as f:
                manifest = json.load(f)
            if manifest.get("articles"):
                seen[slug] = {
                    "entity_name": manifest["entity"]["name"],
                    "entity_type": manifest["entity"]["entity_type"],
                    "article_id": manifest["articles"][0]["id"],
                }
    return list(seen.values())


def count_calls(approach: str, history: List[Dict[str, Any]]) -> int:
    """Model calls behind one profile, read from its history."""
    if approach == "check":
        return max((h.get("model_calls") or 0 for h in history), default=1) or 1
    regenerations = sum(
        1
        for h in history
        if h.get("valid") is False
        and not str(h.get("reasoning", "")).startswith("Reflection failed")
    )
    return 1 + len(history) + regenerations


def _words_in(text: str) -> set:
    return {w.lower().strip(".,") for w in _NAME_OR_NUMBER.findall(text)}


def score_profile(text: str, article_id: str, article: str) -> Dict[str, Any]:
    """No-model checks on one first profile."""
    cites = _CITE.findall(text)
    paragraphs = [  # blocks with prose; a heading line on top doesn't exclude one
        p
        for p in re.split(r"\n\s*\n", text)
        if any(ln.strip() and not ln.lstrip().startswith("#") for ln in p.splitlines())
    ]
    body = _CITE.sub("", re.sub(r"^\s*#.*$", "", text, flags=re.MULTILINE))
    starts = {m.lower() for m in _SENTENCE_START.findall(body)}
    in_article = {w.strip(".,") for w in re.findall(r"[\w'’.,-]+", article.lower())}
    novel = sorted(
        w for w in _words_in(body) if w not in in_article and w not in starts
    )
    return {
        "chars": len(text),
        "citations": len(cites),
        "wrong_ids": sum(1 for c in cites if c.strip() != article_id),
        "headings": len(_HEADING.findall(text)),
        "uncited_paragraphs": sum(1 for p in paragraphs if not _CITE.search(p)),
        "names_not_in_article": novel[:20],
    }


def _grounding(text: str, article_id: str, article: str) -> Dict[str, Any]:
    from src.utils.grounding import verify_profile_grounding

    report = verify_profile_grounding(
        profile_text=text, article_texts={article_id: article}, model_type="gemini"
    )
    levels = [v.support_level.value for v in report.verifications]
    return {
        "grounding_score": report.grounding_score,
        "claims": len(levels),
        "not_supported": levels.count("not_supported"),
        "partial": levels.count("partial"),
        "unclear": levels.count("unclear"),
        "unsupported_claims": [
            v.claim[:300]
            for v in report.verifications
            if v.support_level.value == "not_supported"
        ],
    }


def _claim_scores(
    labels: Dict[Tuple[str, str], Dict[str, Any]], case: Dict[str, Any], text: str
) -> Optional[Dict[str, Any]]:
    from src.evals.claim_labels import score_claims

    label = labels.get((case["entity_name"], case["article_id"]))
    return score_claims(label, "", text, case["article_id"]) if label else None


def load_create_labels(
    eval_dir: str, status: str = "reviewed"
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Claim labels written for first profiles (``"use": "create"``)."""
    from src.evals.claim_labels import load_labels

    return {
        k: v
        for k, v in load_labels(eval_dir, status).items()
        if v.get("use") == "create"
    }


def run_creates(
    cases: List[Dict[str, Any]],
    texts: Dict[str, str],
    approaches: List[str],
    model: str,
    domain: str,
    workers: int = 4,
    grounding: bool = False,
    labels: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """One record per (case, approach), in case order."""
    from src.engine.profiles import create_profile
    from src.utils.tracing import annotate, item_span

    def one(job: Tuple[int, Dict[str, Any], str]) -> Dict[str, Any]:
        i, case, approach = job
        article = texts[case["article_id"]]
        record: Dict[str, Any] = {"case": i, **case, "approach": approach}
        with item_span(
            f"eval create {case['entity_name']}",
            eval_item_id=f"{i}",
            eval_kind=approach,
        ):
            start = time.perf_counter()
            try:
                profile, _, history = create_profile(
                    case["entity_type"],
                    case["entity_name"],
                    article,
                    case["article_id"],
                    domain=domain,
                    model=model,
                    approach=approach,
                )
                text = str(profile.get("text") or "")
                record.update(
                    seconds=round(time.perf_counter() - start, 2),
                    calls=count_calls(approach, history),
                    text=text,
                    tags=profile.get("tags") or [],
                    qc=profile.get("qc"),
                    history=history,
                    scores=score_profile(text, case["article_id"], article),
                )
                if grounding:
                    record["grounding"] = _grounding(text, case["article_id"], article)
                claims = _claim_scores(labels or {}, case, text)
                if claims is not None:
                    record["claims"] = claims
            except Exception as e:  # noqa: BLE001 — one failure must not stop the run
                record["error"] = f"{type(e).__name__}: {e}"
                record["seconds"] = round(time.perf_counter() - start, 2)
            annotate(eval_error=record.get("error"), eval_scores=record.get("scores"))
        return record

    jobs = [(i, c, a) for i, c in enumerate(cases) for a in approaches]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(one, jobs))


def _median(values: List[float]) -> Optional[float]:
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    mid = len(values) // 2
    return (
        values[mid]
        if len(values) % 2
        else round((values[mid - 1] + values[mid]) / 2, 3)
    )


def summarize_creates(records: List[Dict[str, Any]], approach: str) -> Dict[str, Any]:
    rows = [r for r in records if r["approach"] == approach]
    ok = [r for r in rows if "error" not in r]
    s = [r["scores"] for r in ok]
    g = [r["grounding"] for r in ok if r.get("grounding")]
    c = [r["claims"] for r in ok if r.get("claims")]
    out: Dict[str, Any] = {
        "approach": approach,
        "n": len(rows),
        "errors": len(rows) - len(ok),
        "calls": sum(r["calls"] for r in ok),
        "seconds": round(sum(r["seconds"] for r in ok), 1),
        "median_seconds": _median([r["seconds"] for r in ok]),
        "median_chars": _median([x["chars"] for x in s]),
        "no_headings": sum(1 for x in s if not x["headings"]),
        "wrong_ids": sum(x["wrong_ids"] for x in s),
        "uncited_paragraphs": sum(x["uncited_paragraphs"] for x in s),
        "names_not_in_article": sum(len(x["names_not_in_article"]) for x in s),
    }
    if g:
        out["claims_checked"] = sum(x["claims"] for x in g)
        out["not_supported"] = sum(x["not_supported"] for x in g)
        out["partial"] = sum(x["partial"] for x in g)
        out["unclear"] = sum(x.get("unclear", 0) for x in g)
        scores = [x["grounding_score"] for x in g if x["grounding_score"] is not None]
        out["mean_grounding"] = round(sum(scores) / len(scores), 3) if scores else None
    if c:
        out["labelled"] = len(c)
        for key in ("claims_expected", "claims_found", "core_missing", "must_not_hits"):
            vals = [x.get(key) for x in c]
            out[key] = sum(len(v) if isinstance(v, list) else (v or 0) for v in vals)
    return out


def save_create_run(
    eval_dir: str, model: str, records: List[Dict[str, Any]], summaries: List[Dict]
) -> str:
    out_dir = os.path.join(eval_dir, "profile_create")
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip("-")
    path = os.path.join(out_dir, f"{stamp}-{slug}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"summary": summaries, "model": model}) + "\n")
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def load_article_texts(data_path: str) -> Dict[str, str]:
    table = pq.read_table(data_path, columns=["id", "content"]).to_pylist()
    return {a["id"]: a["content"] or "" for a in table}


def print_summaries(summaries: List[Dict[str, Any]]) -> None:
    from rich.table import Table

    keys = list(dict.fromkeys(k for s in summaries for k in s))
    table = Table(title="First-profile comparison")
    table.add_column("")
    for s in summaries:
        table.add_column(str(s["approach"]))
    for key in keys[1:]:
        table.add_row(key, *(str(s.get(key, "")) for s in summaries))
    console.print(table)


def run_profile_create(args, parser, domain_cfg, eval_dir: str) -> None:
    """``just eval profile-create``: both approaches on the same pairs."""
    from src.utils.llm import configure_llm_concurrency

    texts = load_article_texts(domain_cfg.get_data_path())
    if args.cases == "replay":
        cases = [c for c in replay_cases(eval_dir) if texts.get(c["article_id"])]
        cases = cases[: args.limit] if args.limit else cases
    else:
        entities_dir = args.entities_dir or domain_cfg.get_output_dir()
        cases = entity_cases(entities_dir, texts, args.limit or 40, seed=args.seed)
    if not cases:
        parser.error("No (entity, article) pairs found")
    approaches = [a.strip() for a in args.create_approaches.split(",") if a.strip()]
    configure_llm_concurrency(
        cloud_in_flight=args.workers, local_in_flight=args.workers
    )
    labels = load_create_labels(eval_dir, args.claim_labels)
    console.print(
        f"Writing {len(cases)} first profiles x {approaches} with {args.model}"
        f"{' + grounding' if args.grounding else ''}"
        f"{f', {len(labels)} labelled' if labels else ''} …"
    )
    records = run_creates(
        cases,
        texts,
        approaches,
        args.model,
        args.domain,
        workers=args.workers,
        grounding=args.grounding,
        labels=labels,
    )
    summaries = [summarize_creates(records, a) for a in approaches]
    path = save_create_run(eval_dir, args.model, records, summaries)
    print_summaries(summaries)
    console.print(f"Profiles and scores (article-derived, keep off GitHub): {path}")
