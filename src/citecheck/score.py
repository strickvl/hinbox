"""Score a citation checker against the hand labels.

The person labelling answers one question per claim: does the article state
everything in this sentence (tone aside)? "yes" means a good checker says
"supported"; "no" means it should say anything else. Two numbers come out,
each with a 95% range (Wilson interval), since with a few hundred labels a
gap of a few points can be luck:

- ``wrong_passed``: of the claims labelled "no", the share the checker
  called "supported". This is the error that matters most: a wrong sentence
  shown to a historian as checked.
- ``good_doubted``: of the claims labelled "yes", the share it did not call
  "supported". These cost reading time, not trust.

"unsure" labels are left out. A failed call counts in ``errors`` and as a
doubt, never as a pass. Both numbers are also given per stratum ("random",
"doubted"; see ``sample_test_set()``): only the random half estimates the
real-world rate.

Verdicts come from one of three places: a model asked now through the
pipeline's own ``judge_claims()``; the verdicts a finished run stored; or
raw replies from a tuned model sampled on Tinker (``scripts/tinker_citecheck.py
sample``) for the prompts ``prompt_rows()`` writes, read by ``read_reply()``.
"""

import json
import math
import re
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from src.citecheck.export import LEVELS
from src.citecheck.teach import article_chunks
from src.utils.grounding import (
    GROUNDING_MAX_ARTICLE_CHARS,
    grounding_messages,
    judge_claims,
)


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% range for a share of ``k`` out of ``n``."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def _rate(k: int, n: int) -> Dict[str, Any]:
    lo, hi = wilson(k, n)
    return {"count": k, "of": n, "share": k / n if n else None, "low": lo, "high": hi}


def score(
    items: List[Dict[str, Any]],
    labels: Dict[str, Dict[str, Any]],
    predicted: Dict[str, Optional[str]],
) -> Dict[str, Any]:
    """The report for ``predicted`` (item id -> verdict, None = failed)."""

    def tally(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        no = [r for r in rows if labels[r["item_id"]]["label"] == "no"]
        yes = [r for r in rows if labels[r["item_id"]]["label"] == "yes"]
        passed = lambda r: predicted.get(r["item_id"]) == "supported"  # noqa: E731
        return {
            "wrong_passed": _rate(sum(passed(r) for r in no), len(no)),
            "good_doubted": _rate(sum(not passed(r) for r in yes), len(yes)),
            "errors": sum(1 for r in rows if predicted.get(r["item_id"]) is None),
        }

    scored = [
        r for r in items if labels.get(r["item_id"], {}).get("label") in ("yes", "no")
    ]
    strata: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in scored:
        strata[r.get("stratum") or "?"].append(r)
    return {
        "labelled": len(scored),
        "unsure": sum(
            1 for r in items if labels.get(r["item_id"], {}).get("label") == "unsure"
        ),
        "unlabelled": sum(1 for r in items if r["item_id"] not in labels),
        **tally(scored),
        "by_stratum": {k: tally(v) for k, v in sorted(strata.items())},
    }


def describe(report: Dict[str, Any], name: str) -> str:
    def pct(rate: Dict[str, Any]) -> str:
        if not rate["of"]:
            return "n/a"
        return (
            f"{rate['count']}/{rate['of']} = {100 * rate['share']:.1f}% "
            f"(95% range {100 * rate['low']:.1f}-{100 * rate['high']:.1f}%)"
        )

    lines = [
        f"{name}: {report['labelled']} labelled claims "
        f"({report['unsure']} unsure, {report['unlabelled']} unlabelled left out)",
        f"  wrong claims passed: {pct(report['wrong_passed'])}",
        f"  good claims doubted: {pct(report['good_doubted'])}",
        f"  failed calls: {report['errors']}",
    ]
    for stratum, t in report["by_stratum"].items():
        lines.append(
            f"  [{stratum}] wrong passed {pct(t['wrong_passed'])}; "
            f"good doubted {pct(t['good_doubted'])}"
        )
    return "\n".join(lines)


def predict_with_model(
    items: List[Dict[str, Any]],
    texts: Dict[str, str],
    model: str,
    workers: int = 1,
) -> Tuple[Dict[str, Optional[str]], Dict[str, Any]]:
    """Ask ``model`` (through the pipeline's judge call) about every item,
    one call per article chunk. Returns verdicts and timing."""
    from concurrent.futures import ThreadPoolExecutor

    chunks = article_chunks(items)
    predicted: Dict[str, Optional[str]] = {r["item_id"]: None for r in items}

    def ask(chunk: Tuple[str, List[Dict[str, Any]]]) -> List[Tuple[str, Any]]:
        aid, group = chunk
        try:
            verdicts = judge_claims(
                article_id=aid,
                article_text=(texts.get(aid) or "")[:GROUNDING_MAX_ARTICLE_CHARS],
                claims=[(c["entity_name"], c["claim"]) for c in group],
                model_type="ollama" if model.startswith("ollama/") else "cloud",
                ask_passage=False,
                model=model,
            )
        except Exception:
            return [(c["item_id"], None) for c in group]
        return [
            (c["item_id"], getattr(v[0], "value", v[0]) if v else None)
            for c, v in zip(group, verdicts)
        ]

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for pairs in pool.map(ask, chunks):
            predicted.update(dict(pairs))
    seconds = time.perf_counter() - start
    return predicted, {
        "seconds": round(seconds, 1),
        "calls": len(chunks),
        "articles": len({aid for aid, _ in chunks}),
    }


def stored_predictions(
    items: List[Dict[str, Any]], rows: List[Dict[str, Any]]
) -> Dict[str, Optional[str]]:
    """The verdicts a finished run stored (``claim_rows()``) for ``items``."""
    stored = {r["item_id"]: r.get("stored_verdict") for r in rows}
    return {r["item_id"]: stored.get(r["item_id"]) for r in items}


def prompt_rows(
    items: List[Dict[str, Any]], texts: Dict[str, str]
) -> List[Dict[str, Any]]:
    """One judge prompt per article chunk of ``items``, for a model sampled
    outside Hinbox (a tuned model on Tinker): ``{"item_ids", "messages"}``."""
    return [
        {
            "item_ids": [c["item_id"] for c in group],
            "messages": grounding_messages(
                aid,
                (texts.get(aid) or "")[:GROUNDING_MAX_ARTICLE_CHARS],
                [(c["entity_name"], c["claim"]) for c in group],
            ),
        }
        for aid, group in article_chunks(items)
    ]


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


_PLAIN_LEVEL = re.compile(
    r"support[ _]level\W{0,4}(supported|partial|not[ _]supported|unclear)", re.I
)


def _plain_verdicts(body: str, count: int) -> List[Optional[str]]:
    """Verdicts from a reply written as text ("1. Reasoning: … Support_level:
    partial"), as an untuned model answers: the levels in claim order, used
    only when there is exactly one per claim."""
    found = [m.group(1).lower().replace(" ", "_") for m in _PLAIN_LEVEL.finditer(body)]
    return list(found) if len(found) == count else [None] * count


def read_reply(text: str, count: int) -> List[Optional[str]]:
    """Verdicts from a raw reply in the pipeline's JSON format, by claim
    number, or written as text (``_plain_verdicts()``); a claim with no
    readable verdict is None."""
    out: List[Optional[str]] = [None] * count
    body = _FENCE.sub("", (text or "").strip())
    start, end = body.find("{"), body.rfind("}")
    try:
        payload = json.loads(body[start : end + 1]) if 0 <= start < end else None
    except json.JSONDecodeError:
        payload = None
    if payload is None:
        return _plain_verdicts(body, count)
    items = payload.get("items") if isinstance(payload, dict) else None
    for pos, item in enumerate(items or []):
        if not isinstance(item, dict):
            continue
        n = item.get("claim_number")
        i = n - 1 if isinstance(n, int) and 1 <= n <= count else pos
        level = item.get("support_level")
        if i < count and level in LEVELS:
            out[i] = level
    return out


def reply_predictions(replies: List[Dict[str, Any]]) -> Dict[str, Optional[str]]:
    """Verdicts from rows ``{"item_ids", "reply"}`` (Tinker sampling output)."""
    predicted: Dict[str, Optional[str]] = {}
    for row in replies:
        ids = row.get("item_ids") or []
        for iid, level in zip(ids, read_reply(row.get("reply") or "", len(ids))):
            predicted[iid] = level
    return predicted
