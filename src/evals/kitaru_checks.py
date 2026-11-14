"""The profile-update eval's model-free checks, on one recorded step.

A Kitaru step session (``scripts/kitaru_traces.py --step profile.update``,
or a replay of one by ``scripts/kitaru_step.py``) holds the step's arguments
as its inputs and its result as its outputs. ``step_checks()`` scores the
updated profile against the old one with ``score_update()``'s checks that
need no case label: collapse, kept facts, unchanged text, added sentences,
citing the article, invented citations, "sources disagree" sentences and
numbers the article doesn't have. The kind-specific checks (a planted
dispute kept, a labelled fact stated) need a case, so they stay in
``just eval profile-update``.

``step_results()`` turns those scores into the rows a Kitaru evaluator
returns (``scripts/kitaru_evaluators/profile_update_checks.py`` wraps each in
an ``EvaluationResult``), kept here so they are tested without Kitaru.
"""

from typing import Any, Dict, List, Optional

from src.evals.profile_update import COLLAPSE_RATIO, MIN_RETENTION, score_update


def step_checks(
    inputs: Dict[str, Any], outputs: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """``score_update()`` on a step's recorded arguments and result.

    Reads ``update_profile()``'s argument names (``new_article_id``…) or
    ``edit_profile()``'s (``article_id``…), whichever the step recorded. A
    step that recorded no profile (it failed) is scored as if it kept the
    old one, which is what the pipeline does then.
    """
    old = inputs.get("existing_profile") or {}
    article_id = inputs.get("new_article_id") or inputs.get("article_id") or ""
    article_text = inputs.get("new_article_text") or inputs.get("article_text") or ""
    new = (outputs or {}).get("profile") or old
    case = {
        "profile": old,
        "article_id": article_id,
        "article_text": article_text,
        "kind": "step",  # no kind of its own: only the checks every case has
    }
    scores = score_update(case, str(new.get("text") or ""))
    # Every real update adds the article to the profile's sources, even one
    # that rightly leaves the text alone; a failed one doesn't
    scores["updated"] = article_id in (new.get("sources") or [])
    return scores


def step_results(
    inputs: Dict[str, Any], outputs: Optional[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """The checks as evaluator rows: ``name``, ``score`` and, for the checks
    that fail a case, ``passed``; the rest are reported, as in the eval."""
    s = step_checks(inputs, outputs)
    invented = s["invented_citations"]
    failures = [
        why
        for failed, why in (
            (s["collapsed"], f"collapsed to {s['length_ratio']} of its length"),
            (s["low_retention"], f"kept {s['fact_retention']:.0%} of cited sentences"),
            (bool(invented), f"invented citations: {', '.join(invented)}"),
        )
        if failed
    ]
    return [
        {
            "name": "update_ok",
            "score": s["passed"],
            "passed": s["passed"],
            "explanation": "; ".join(failures)
            or f"no collapse, at least {MIN_RETENTION:.0%} of cited sentences "
            "kept, no invented citation",
        },
        {
            "name": "length_ratio",
            "score": float(s["length_ratio"]),
            "passed": not s["collapsed"],
            "target_score": 1.0,
            "explanation": f"{s['old_chars']} -> {s['new_chars']} characters "
            f"(below {COLLAPSE_RATIO} is a collapse)",
        },
        {
            "name": "fact_retention",
            "score": float(s["fact_retention"]),
            "passed": not s["low_retention"],
            "min_score": 0.0,
            "max_score": 1.0,
            "explanation": f"{s['facts_lost']} cited sentences lost",
        },
        {
            "name": "invented_citations",
            "score": float(len(invented)),
            "passed": not invented,
            "explanation": ", ".join(invented) or "none",
        },
        {
            "name": "unchanged_share",
            "score": float(s["unchanged_share"]),
            "min_score": 0.0,
            "max_score": 1.0,
            "explanation": "share of old sentences kept word for word",
        },
        {
            "name": "added_sentences",
            "score": float(s["added_sentences"]),
            "explanation": "new sentences with no close match in the old profile",
        },
        {"name": "cites_article", "score": bool(s["cites_article"])},
        {
            "name": "updated",
            "score": bool(s["updated"]),
            "explanation": "the article was added to the profile's sources",
        },
        {
            "name": "dispute_claims",
            "score": float(s["dispute_claims"]),
            "explanation": "new sentences saying the sources disagree",
        },
        {
            "name": "unsupported_numbers",
            "score": float(s["unsupported_numbers"]),
            "explanation": "new sentences citing the article with numbers it lacks",
        },
    ]
