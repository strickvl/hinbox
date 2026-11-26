"""Score a recorded citation check (a Kitaru ``grounding`` step session).

A session's inputs are ``judge_claims()``'s arguments, its outputs the
verdicts it recorded (``verdicts``: one per claim, ``None`` where the reply
had none). Hand labels travel in the inputs under ``labels``:
``{"item_ids": [...], "gold": ["yes" | "no" | "unsure" | None, ...],
"strata": [...]}``, one entry per claim; ``just citecheck kitaru-sessions``
writes them, and ``just citecheck kitaru-pull`` fills ``gold`` in from a
Kitaru investigation's answers. A replay keeps the labels
(``scripts/kitaru_step.py``), and when a replay's have no ``gold`` the
baseline session's metadata is read, which holds the same keys (Kitaru
hands an evaluator the baseline's metadata, not its inputs).

``citation_results()`` returns the rows a Kitaru evaluator gives back
(``scripts/kitaru_evaluators/citation_check_checks.py``). Kitaru shows each
row's mean over the sessions that have it, so the quality rows are
true/false per session and only present where they apply:

- ``wrong_passed``: the session holds a claim labelled "no" and the judge
  called one of them "supported" (the error that matters most);
- ``good_doubted``: it holds a claim labelled "yes" and the judge called
  one of them anything else;
- ``agrees``: every labelled claim judged the way its label says;
- ``answered``: every claim got a verdict (a failed reply is never a pass).

With one claim per session, which is what ``kitaru-sessions`` writes, the
means are exactly the shares ``just citecheck score`` reports. Speed and
price rows: ``seconds`` (the session's wall time), ``input_tokens``,
``output_tokens``, and ``cost_usd`` from the evaluator's ``prices`` (model
-> [dollars per million input tokens, per million output]) or else the cost
Kitaru worked out itself.
"""

from typing import Any, Dict, List, Optional, Sequence


def _labels(
    inputs: Dict[str, Any], baseline_metadata: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    labels = inputs.get("labels") or {}
    if not labels.get("gold") and (baseline_metadata or {}).get("gold"):
        labels = baseline_metadata or {}
    return labels


def _levels(outputs: Optional[Dict[str, Any]], count: int) -> List[Optional[str]]:
    verdicts = (outputs or {}).get("verdicts") or []
    out: List[Optional[str]] = [None] * count
    for i, v in enumerate(verdicts[:count]):
        if isinstance(v, dict) and v.get("support_level"):
            out[i] = str(v["support_level"])
    return out


def cost_usd(
    model: str,
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    prices: Optional[Dict[str, Sequence[float]]],
) -> Optional[float]:
    """Dollars for one session at ``prices[model]`` (per million tokens)."""
    price = (prices or {}).get(model)
    if not price or input_tokens is None or output_tokens is None:
        return None
    return (input_tokens * float(price[0]) + output_tokens * float(price[1])) / 1e6


def citation_results(
    inputs: Dict[str, Any],
    outputs: Optional[Dict[str, Any]],
    *,
    seconds: Optional[float] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    kitaru_cost: Optional[float] = None,
    baseline_metadata: Optional[Dict[str, Any]] = None,
    prices: Optional[Dict[str, Sequence[float]]] = None,
) -> List[Dict[str, Any]]:
    """Evaluation rows for one citation-check session (see the module note)."""
    claims = inputs.get("claims") or []
    levels = _levels(outputs, len(claims))
    labels = _labels(inputs, baseline_metadata)
    gold = list(labels.get("gold") or [])[: len(claims)]
    gold += [None] * (len(claims) - len(gold))
    rows: List[Dict[str, Any]] = [
        {
            "name": "answered",
            "score": all(level is not None for level in levels),
            "explanation": f"{sum(lv is not None for lv in levels)}/{len(levels)} claims got a verdict",
        }
    ]
    if len(claims) == 1 and levels[0]:
        rows.append({"name": "verdict", "value": levels[0]})
    no = [i for i, g in enumerate(gold) if g == "no"]
    yes = [i for i, g in enumerate(gold) if g == "yes"]
    if no:
        passed = [i for i in no if levels[i] == "supported"]
        rows.append(
            {
                "name": "wrong_passed",
                "score": bool(passed),
                "explanation": f"{len(passed)} of {len(no)} claims labelled 'no' called supported",
            }
        )
    if yes:
        doubted = [i for i in yes if levels[i] != "supported"]
        rows.append(
            {
                "name": "good_doubted",
                "score": bool(doubted),
                "explanation": f"{len(doubted)} of {len(yes)} claims labelled 'yes' not called supported",
            }
        )
    if no or yes:
        right = all(levels[i] == "supported" for i in yes) and all(
            levels[i] not in (None, "supported") for i in no
        )
        rows.append({"name": "agrees", "score": right})
    if seconds is not None:
        rows.append({"name": "seconds", "score": float(seconds), "min_score": 0.0})
    for name, value in (
        ("input_tokens", input_tokens),
        ("output_tokens", output_tokens),
    ):
        if value is not None:
            rows.append({"name": name, "score": float(value), "min_score": 0.0})
    cost = cost_usd(str(inputs.get("model") or ""), input_tokens, output_tokens, prices)
    if cost is None and kitaru_cost is not None:
        cost = float(kitaru_cost)
    if cost is not None:
        rows.append({"name": "cost_usd", "score": cost, "min_score": 0.0})
    return rows
