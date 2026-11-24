"""The profile-update eval's model-free checks, on one recorded step.

A Kitaru step session (``scripts/kitaru_traces.py --step profile.update``,
or a replay of one by ``scripts/kitaru_step.py``) holds the step's arguments
as its inputs and its result as its outputs. ``step_checks()`` scores the
updated profile against the old one with ``score_update()``'s checks that
need no case label: collapse, kept facts, unchanged text, added sentences,
citing the article, invented citations, "sources disagree" sentences and
numbers the article doesn't have. For an eval case, whose kind and check
data its step session holds in its inputs (``labels``: ``eval_kind``,
``eval_expected``, ``eval_claim_label``; a replay keeps them), ``case_ok`` is
the eval's own verdict, ``case_passed()``: at most 2 added sentences for an
``irrelevant`` case, the labelled facts (or a citation, unlabelled) for a
``new_fact`` case, both years kept for a ``disputed`` one.

``step_results()`` turns those scores into the rows a Kitaru evaluator
returns (``scripts/kitaru_evaluators/profile_update_checks.py`` wraps each in
an ``EvaluationResult``), kept here so they are tested without Kitaru.
``match_results()`` does the same for a ``match_check`` step session
(``scripts/kitaru_evaluators/match_check_checks.py``): the model's answer,
and whether it agrees with a merge eval pair's hand label.
"""

from typing import Any, Dict, List, Optional

from src.evals.profile_update import (
    COLLAPSE_RATIO,
    MIN_RETENTION,
    case_passed,
    score_update,
)


def _case(labels: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The eval case's own check data from a step session's labels: its kind,
    ``expected`` (a disputed case's two years and sentences) and claim label.
    A kind whose data is missing (sessions from before these labels) is
    scored with the checks every case has only (kind ``step``)."""
    labels = labels or {}
    kind = labels.get("eval_kind")
    expected = labels.get("eval_expected") or {}
    label = labels.get("eval_claim_label")
    if (
        kind not in ("irrelevant", "new_fact", "disputed")
        or (kind == "disputed" and "original_value" not in expected)
        or label is True  # an older session said only that a label applied
    ):
        return {"kind": "step"}
    return {"kind": kind, "expected": expected, "claim_label": label or None}


def step_checks(
    inputs: Dict[str, Any],
    outputs: Optional[Dict[str, Any]],
    labels: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """``score_update()`` on a step's recorded arguments and result.

    Reads ``update_profile()``'s argument names (``new_article_id``…) or
    ``edit_profile()``'s (``article_id``…), whichever the step recorded. A
    step that recorded no profile (it failed) is scored as if it kept the
    old one, which is what the pipeline does then. With an eval case's
    ``labels``, ``case_passed`` is the eval's verdict on the case and
    ``case_kind`` its kind (``step`` when it can't be judged here).
    """
    old = inputs.get("existing_profile") or {}
    article_id = inputs.get("new_article_id") or inputs.get("article_id") or ""
    article_text = inputs.get("new_article_text") or inputs.get("article_text") or ""
    new = (outputs or {}).get("profile") or old
    case = {
        "profile": old,
        "article_id": article_id,
        "article_text": article_text,
        **_case(labels),
    }
    scores = score_update(case, str(new.get("text") or ""))
    scores["case_kind"] = case["kind"]
    scores["case_passed"] = scores["passed"]
    # update_ok: only the checks every case has (collapse, retention, citations)
    scores["passed"] = case_passed({"kind": "step"}, scores)
    # Every real update adds the article to the profile's sources, even one
    # that rightly leaves the text alone; a failed one doesn't
    scores["updated"] = article_id in (new.get("sources") or [])
    return scores


def case_result(scores: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """``case_ok``: the eval's own verdict on the session's case (its kind and
    check data in the inputs' ``labels``, which a replay keeps), or None for
    a session that isn't an eval case or lacks its case's data."""
    kind, ok = scores["case_kind"], scores["case_passed"]
    if kind == "step":
        return None
    if not scores["passed"]:
        why = "the profile was damaged (see update_ok)"
    elif kind == "irrelevant":
        why = f"{scores['added_sentences']} sentences added (at most 2)"
    elif kind == "disputed":
        why = "keeps both years" if ok else "doesn't keep both years with their sources"
    elif "claims_ok" in scores:
        missing, hits = (
            scores.get("core_missing") or [],
            scores.get("must_not_hits") or [],
        )
        why = "states every labelled fact" if ok else "; ".join(
            filter(None, [
                f"misses {len(missing)} labelled fact(s)" if missing else "",
                f"adds {len(hits)} wrong reading(s)" if hits else "",
                "says sources disagree" if scores["dispute_claims"]
                and not scores.get("disputes_ok") else "",
                "" if scores["cites_article"] else "doesn't cite the article",
            ])
        )  # fmt: skip
    else:
        why = "cites the article" if ok else "the article is not cited"
    return {
        "name": "case_ok",
        "score": ok,
        "passed": ok,
        "explanation": f"{kind} case: {why}",
    }


def step_results(
    inputs: Dict[str, Any],
    outputs: Optional[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """The checks as evaluator rows: ``name``, ``score`` and, for the checks
    that fail a case, ``passed``; the rest are reported, as in the eval.
    With an eval case's kind in ``metadata``, also ``case_ok``."""
    s = step_checks(inputs, outputs, metadata)
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
    rows = [
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
    case = case_result(s)
    return rows + ([case] if case else [])


def match_results(
    outputs: Optional[Dict[str, Any]], labels: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """A ``match_check`` step's verdict as Kitaru evaluation rows.

    ``is_match`` and ``confidence`` are what the model answered. For a merge
    eval pair, whose hand label (``eval_label``: the same entity or not) its
    step session holds in its inputs, ``match_ok`` says whether the answer
    agrees, and ``false_merge``/``missed_merge`` which way it was wrong: a
    false merge joins two entities' profiles, so it costs more than a missed
    one. A call that failed (``check_failed()``) has no verdict: every row
    fails.
    """
    from src.engine.match_checker import FAILED_CHECK

    result = (outputs or {}).get("result") or {}
    reason = str(result.get("reason") or "")
    if "is_match" not in result or reason.startswith(FAILED_CHECK):
        failed = reason or "the step recorded no verdict"
        return [{"name": "match_ok", "score": False, "explanation": failed}]
    said = bool(result["is_match"])
    rows: List[Dict[str, Any]] = [
        {"name": "is_match", "score": said, "explanation": reason[:500]},
        {
            "name": "confidence",
            "score": float(result.get("confidence") or 0.0),
            "min_score": 0.0,
            "max_score": 1.0,
        },
    ]
    label = (labels or {}).get("eval_label")
    if not isinstance(label, bool):
        return rows
    words = {True: "the same", False: "different"}
    return rows + [
        {
            "name": "match_ok",
            "score": said == label,
            "explanation": f"answered {words[said]}, labelled {words[label]}",
        },
        {"name": "false_merge", "score": said and not label},
        {"name": "missed_merge", "score": label and not said},
    ]


_SEVERE_EXTRACTION_FLAGS = {
    "zero_entities",
    "high_drop_rate",
    "many_duplicates",
    "many_low_quality_names",
    "extraction_exception",
}


def extraction_results(
    inputs: Dict[str, Any],
    outputs: Optional[Dict[str, Any]],
    labels: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """An ``extract.<type>`` step's health checks as Kitaru evaluation rows.

    Tracks completion (no error / cutoff), final and raw entity counts, QC
    dedups/drops/flags, retries, and for people/organizations/locations the
    share of extracted entities actually mentioned in the article.
    """
    if not isinstance(outputs, dict) or outputs.get("error"):
        failed = (outputs or {}).get("error") or "the step recorded no result"
        return [
            {
                "name": "extract_ok",
                "score": False,
                "passed": False,
                "explanation": str(failed)[:500],
            }
        ]
    entities = outputs.get("entities") or []
    counts = outputs.get("counts") or {}
    flags = list(outputs.get("qc_flags") or [])
    severe = sorted(set(flags) & _SEVERE_EXTRACTION_FLAGS)
    raw_count = int(counts.get("raw_count", len(entities)))
    deduped = int(counts.get("deduped", 0))
    dropped = int(counts.get("dropped_missing_required", 0))
    entity_type = str(inputs.get("entity_type") or "entities")
    ok = "extraction_exception" not in flags
    rows: List[Dict[str, Any]] = [
        {
            "name": "extract_ok",
            "score": ok,
            "passed": ok,
            "explanation": (
                f"{len(entities)} {entity_type} ({raw_count} raw, "
                f"{deduped} deduped, {dropped} dropped)"
                if ok
                else "extraction failed with exception"
            ),
        },
        {"name": "entity_count", "score": float(len(entities))},
        {"name": "raw_count", "score": float(raw_count)},
        {
            "name": "deduped",
            "score": float(deduped),
            "explanation": "duplicate entities collapsed by QC",
        },
        {
            "name": "dropped_missing_required",
            "score": float(dropped),
            "explanation": "entities dropped for missing required fields",
        },
        {"name": "retry_attempted", "score": bool(outputs.get("retry_attempted"))},
        {
            "name": "qc_flags",
            "score": float(len(flags)),
            "passed": not severe,
            "explanation": ", ".join(flags) or "none",
        },
    ]
    article_content = str(inputs.get("article_content") or "")
    if (
        entity_type in ("people", "organizations", "locations")
        and entities
        and article_content
    ):
        from src.utils.quality_controls import filter_entities_by_article_relevance

        _, rel = filter_entities_by_article_relevance(
            entity_type=entity_type,
            entities=entities,
            article_text=article_content,
            domain=str(inputs.get("domain") or "guantanamo"),
            require_mention=True,
        )
        rate = rel.output_count / rel.input_count if rel.input_count else 1.0
        rows.append(
            {
                "name": "mention_rate",
                "score": round(rate, 4),
                "passed": rel.dropped == 0,
                "min_score": 0.0,
                "max_score": 1.0,
                "explanation": f"{rel.output_count}/{rel.input_count} mentioned in article",
            }
        )
    return rows


def grounding_results(
    inputs: Dict[str, Any],
    outputs: Optional[Dict[str, Any]],
    labels: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """A ``grounding`` step's verdicts as Kitaru evaluation rows.

    Checks that every claim received a verdict, reports verdict counts and
    shares (``supported_share``, ``accepted_share`` for supported+partial),
    checks how often quoted passages match the article text, and compares
    against ``labels["expected_levels"]`` when hand labels are attached.
    """
    if not isinstance(outputs, dict) or outputs.get("error"):
        failed = (outputs or {}).get("error") or "the step recorded no verdicts"
        return [
            {
                "name": "grounding_ok",
                "score": False,
                "passed": False,
                "explanation": str(failed)[:500],
            }
        ]
    claims = inputs.get("claims") or []
    verdicts = outputs.get("verdicts") or []
    total = max(len(claims), len(verdicts))
    counts = {k: 0 for k in ("supported", "partial", "not_supported", "unclear")}
    missing = max(0, len(claims) - len(verdicts))
    for v in verdicts:
        level = v.get("support_level") if isinstance(v, dict) else None
        if level in counts:
            counts[level] += 1
        else:
            missing += 1
    ok = missing == 0 and len(verdicts) == len(claims)
    supported_share = counts["supported"] / total if total else 1.0
    accepted_share = (counts["supported"] + counts["partial"]) / total if total else 1.0
    rows: List[Dict[str, Any]] = [
        {
            "name": "grounding_ok",
            "score": ok,
            "passed": ok,
            "explanation": (
                f"judged all {len(claims)} claim(s)"
                if ok
                else f"missing {missing} verdict(s) of {len(claims)} claim(s)"
            ),
        },
        {
            "name": "missing_verdicts",
            "score": float(missing),
            "passed": missing == 0,
        },
        {
            "name": "supported_share",
            "score": round(supported_share, 4),
            "min_score": 0.0,
            "max_score": 1.0,
            "explanation": f"{counts['supported']}/{total} supported",
        },
        {
            "name": "accepted_share",
            "score": round(accepted_share, 4),
            "min_score": 0.0,
            "max_score": 1.0,
            "explanation": (
                f"{counts['supported'] + counts['partial']}/{total} "
                "supported or partial"
            ),
        },
        {"name": "supported", "score": float(counts["supported"])},
        {"name": "partial", "score": float(counts["partial"])},
        {"name": "not_supported", "score": float(counts["not_supported"])},
        {"name": "unclear", "score": float(counts["unclear"])},
    ]
    article_text = str(inputs.get("article_text") or "")
    if article_text and verdicts:
        from src.utils.grounding_passages import passages_for

        with_passage = sum(
            1
            for v in verdicts
            if isinstance(v, dict)
            and passages_for(article_text, v.get("reasoning"), v.get("passage"))
        )
        rows.append(
            {
                "name": "passage_rate",
                "score": round(with_passage / len(verdicts), 4),
                "min_score": 0.0,
                "max_score": 1.0,
                "explanation": (
                    f"{with_passage}/{len(verdicts)} verdicts quote a passage in the article"
                ),
            }
        )
    expected = (labels or {}).get("expected_levels")
    if isinstance(expected, list) and expected and len(expected) == len(verdicts):
        agree = sum(
            1
            for exp, v in zip(expected, verdicts)
            if isinstance(v, dict) and v.get("support_level") == exp
        )
        rows.append(
            {
                "name": "label_agreement",
                "score": round(agree / len(expected), 4),
                "passed": agree == len(expected),
                "min_score": 0.0,
                "max_score": 1.0,
                "explanation": f"matched {agree}/{len(expected)} expected verdict(s)",
            }
        )
    return rows
