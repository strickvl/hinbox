"""Several answers per test claim, and fair comparisons between checkers.

Two questions the single-answer score can't settle:

1. **Is a gap between two checkers real?** Both answered the same 400 test
   claims, so what decides it is the claims where they answer differently.
   ``paired()`` counts, among the wrong claims, those only checker A passed
   and those only B passed (and the same for good claims doubted), and gives
   a two-sided sign test p-value on those disagreements. If v4 and untuned
   disagree on 8 wrong claims, 5 against 3, the gap is noise (p = 0.73); if
   they disagree on 40, 30 against 10, it is not (p = 0.002), whatever the
   two totals look like.
2. **Does asking the same model several times help?** ``scripts/
   tinker_citecheck.py sample --votes N`` writes N answers per prompt (rows
   with ``sample`` 0..N-1). ``vote_counts()`` counts, per claim, how many
   readable answers said "supported". From that: each answer set scored on
   its own (how much one run differs from the next), the majority vote, and
   a dial, ``at_least(k)``: pass a claim only when at least k answers say
   supported. Raising k passes fewer wrong claims and doubts more good ones,
   so two models can be compared as whole curves.

Only item ids and counts come out; no claim or article text.
"""

import json
import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.citecheck.score import reply_predictions, score

Predicted = Dict[str, Optional[str]]


def by_sample(replies: Iterable[Dict[str, Any]]) -> Dict[int, Predicted]:
    """Sample number -> verdicts, from reply rows (``sample`` missing = 0)."""
    groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in replies:
        groups[int(row.get("sample") or 0)].append(row)
    return {n: reply_predictions(rows) for n, rows in sorted(groups.items())}


def vote_counts(samples: Dict[int, Predicted]) -> Dict[str, Tuple[int, int]]:
    """Item id -> (answers saying supported, readable answers)."""
    out: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for predicted in samples.values():
        for iid, level in predicted.items():
            tally = out[iid]
            if level is not None:
                tally[0] += level == "supported"
                tally[1] += 1
    return {k: (v[0], v[1]) for k, v in out.items()}


def at_least(counts: Dict[str, Tuple[int, int]], k: int) -> Predicted:
    """Supported when at least ``k`` answers said so; a claim with no
    readable answer is a failed call (None), which the score counts as a
    doubt."""
    return {
        iid: None if n == 0 else ("supported" if yes >= k else "partial")
        for iid, (yes, n) in counts.items()
    }


def majority(counts: Dict[str, Tuple[int, int]]) -> Predicted:
    """Supported when more than half of the readable answers said so; a tie
    is a doubt."""
    return {
        iid: None if n == 0 else ("supported" if 2 * yes > n else "partial")
        for iid, (yes, n) in counts.items()
    }


def _totals(report: Dict[str, Any]) -> Dict[str, int]:
    return {
        "wrong_passed": report["wrong_passed"]["count"],
        "good_doubted": report["good_doubted"]["count"],
        "errors": report["errors"],
    }


def vote_report(
    items: List[Dict[str, Any]],
    labels: Dict[str, Dict[str, Any]],
    samples: Dict[int, Predicted],
) -> Dict[str, Any]:
    """Each sample alone, the spread across them, the majority vote and the
    dial at every k."""
    alone = {n: _totals(score(items, labels, p)) for n, p in samples.items()}
    counts = vote_counts(samples)
    n = len(samples)

    def spread(key: str) -> Dict[str, float]:
        vals = [a[key] for a in alone.values()]
        mean = sum(vals) / len(vals)
        sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1))
        return {
            "min": min(vals),
            "mean": round(mean, 1),
            "max": max(vals),
            "sd": round(sd, 1),
        }

    return {
        "samples": n,
        "alone": alone,
        "spread": {k: spread(k) for k in ("wrong_passed", "good_doubted")},
        "majority": score(items, labels, majority(counts)),
        "dial": {
            k: _totals(score(items, labels, at_least(counts, k)))
            for k in range(1, n + 1)
        },
    }


def describe_votes(report: Dict[str, Any], name: str) -> str:
    s = report["spread"]
    m = report["majority"]
    lines = [
        f"{name}: {report['samples']} answers per claim",
        "  each answer set alone: wrong passed "
        f"{s['wrong_passed']['min']}-{s['wrong_passed']['max']} (mean {s['wrong_passed']['mean']}, sd {s['wrong_passed']['sd']}); "
        f"good doubted {s['good_doubted']['min']}-{s['good_doubted']['max']} (mean {s['good_doubted']['mean']}, sd {s['good_doubted']['sd']})",
        f"  majority vote: wrong passed {m['wrong_passed']['count']}/{m['wrong_passed']['of']}, "
        f"good doubted {m['good_doubted']['count']}/{m['good_doubted']['of']}, failed {m['errors']}",
        "  dial (pass only if at least k answers say supported):",
    ]
    for k, t in report["dial"].items():
        lines.append(
            f"    k={k}: wrong passed {t['wrong_passed']}, good doubted {t['good_doubted']}"
        )
    return "\n".join(lines)


def sign_test(a: int, b: int) -> float:
    """Two-sided exact sign test p-value for ``a`` against ``b`` disagreements."""
    n = a + b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(a, b) + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired(
    items: List[Dict[str, Any]],
    labels: Dict[str, Dict[str, Any]],
    base: Predicted,
    other: Predicted,
) -> Dict[str, Any]:
    """Where ``other`` and ``base`` answer the same claim differently. On
    wrong claims, ``other_better`` counts those base passed and other didn't;
    on good claims, those base doubted and other passed."""

    def passed(p: Predicted, iid: str) -> bool:
        return p.get(iid) == "supported"

    out: Dict[str, Any] = {}
    for label, key in (("no", "wrong"), ("yes", "good")):
        ids = [
            r["item_id"]
            for r in items
            if labels.get(r["item_id"], {}).get("label") == label
        ]
        only_base = sum(passed(base, i) and not passed(other, i) for i in ids)
        only_other = sum(passed(other, i) and not passed(base, i) for i in ids)
        better, worse = (
            (only_base, only_other) if label == "no" else (only_other, only_base)
        )
        out[key] = {
            "other_better": better,
            "other_worse": worse,
            "p": round(sign_test(better, worse), 4),
        }
    return out


def load_predictions(path: str) -> Tuple[Predicted, str]:
    """Verdicts from a saved score (``score --save``: JSON with
    ``predicted``) or a replies file; a replies file with several samples is
    read as its majority vote."""
    if path.endswith(".json"):
        with open(path) as f:
            return json.load(f)["predicted"], "saved"
    from src.citecheck.items import read_jsonl

    samples = by_sample(read_jsonl(path))
    if len(samples) == 1:
        return next(iter(samples.values())), "replies"
    return majority(vote_counts(samples)), f"majority of {len(samples)}"


def wrong_passed_by_all(
    items: List[Dict[str, Any]],
    labels: Dict[str, Dict[str, Any]],
    sources: Iterable[Predicted],
) -> List[str]:
    """Ids of wrong claims every source passed."""
    sources = list(sources)
    return [
        r["item_id"]
        for r in items
        if labels.get(r["item_id"], {}).get("label") == "no"
        and all(p.get(r["item_id"]) == "supported" for p in sources)
    ]


def all_disagree(
    items: List[Dict[str, Any]],
    labels: Dict[str, Dict[str, Any]],
    sources: Iterable[Predicted],
) -> List[str]:
    """Ids of labelled claims where every source's yes/no differs from the
    label: candidates for a second look at the label, never changed in code.
    A source with no verdict for a claim doesn't count as disagreeing."""
    sources = list(sources)
    out = []
    for r in items:
        label = labels.get(r["item_id"], {}).get("label")
        if label not in ("yes", "no"):
            continue
        said = [p.get(r["item_id"]) for p in sources]
        if all(s is not None and (s == "supported") != (label == "yes") for s in said):
            out.append(r["item_id"])
    return out
