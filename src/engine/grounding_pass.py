"""Profile grounding as a post-processing pass over all changed profiles.

Many profiles cite the same article (one article about a hearing gives a
judge, two lawyers, a detainee, a court and a camp, each with a profile
citing it). The pass collects every claim that needs checking, then makes
one model call per article for all of them, instead of one call per profile
per article. The calls run in a thread pool; the LLM concurrency limiters in
``src/utils/llm.py`` still cap how many are in flight. Results are written
back to the entity dicts on the calling thread.
"""

from typing import Any, Dict, List, Tuple

from src.engine.article_processor import ArticleProcessor
from src.logging_config import log
from src.utils.grounding import (
    COUNTED_LEVELS,
    claim_counts,
    describe_claim_counts,
    finish_report,
    grounding_is_current,
    grounding_judge,
    judge_pending,
    judged_by,
    plan_grounding,
)
from src.utils.quality_controls import CITATION_RE


def _display_name(entity_key: Any, entity_data: Dict[str, Any]) -> str:
    """The entity's name (events: title), shown next to each claim so the
    model knows who "he" or "it" is."""
    name = entity_data.get("name") or entity_data.get("title")
    if not name:
        name = entity_key[0] if isinstance(entity_key, tuple) else entity_key
    return str(name)


def run_profile_grounding_postprocess(
    *,
    entities: Dict[str, Dict],
    rows: List[Dict],
    processor: ArticleProcessor,
    model_type: str,
    workers: int = 1,
) -> Dict[str, int]:
    """Run grounding verification on entity profiles as a batch post-processing step.

    For each entity with citations in its profile, verifies that claims are
    supported by source article text. Stores the grounding report on each
    entity as entity["profile_grounding"]. Skips entities whose stored report
    is still current (``grounding_is_current()``: same text, and no failed or
    newly available source) and was judged by the same model
    (``HINBOX_GROUNDING_MODEL`` or the run's default). Otherwise only claims the stored report has no
    verdict for go to the model, grouped by article across profiles. Up to
    ``workers`` calls run at once.

    Returns summary counts: ``verified`` profiles, ``skipped_unchanged``,
    ``skipped_no_citations``, ``llm_checks`` (profile-article pairs that had
    claims sent to the model), ``reused_checks`` (pairs kept whole from
    earlier reports) and ``llm_calls`` (model calls made). It also counts
    claims per verdict over every stored report after the pass, checked
    this run or not (``supported``, ``partial``, ``not_supported``,
    ``unclear``, ``missing_source``, ``claims``), and
    ``profiles_with_problems``: profiles with any claim that isn't plainly
    supported. A mean grounding score near 1 can hide dozens of these.
    """
    # Build article_id -> content map from rows
    article_texts: Dict[str, str] = {}
    for i, row in enumerate(rows):
        info = processor.prepare_article_info(row, i)
        if info["id"] and info["content"]:
            article_texts[info["id"]] = info["content"]

    counts = {
        "verified": 0,
        "skipped_unchanged": 0,
        "skipped_no_citations": 0,
        "llm_checks": 0,
        "reused_checks": 0,
        "llm_calls": 0,
    }
    jobs: List[Tuple[str, str, Dict[str, Any], str]] = []
    model = grounding_judge(model_type)

    for entity_type, entity_dict in entities.items():
        for entity_key, entity_data in entity_dict.items():
            profile_text = (entity_data.get("profile") or {}).get("text", "")
            if not profile_text:
                continue

            # Check if profile has citations worth verifying
            if not CITATION_RE.findall(profile_text):
                counts["skipped_no_citations"] += 1
                continue

            # Skip if the stored report still answers for this text and these
            # articles. A report with a failed call or a missing source that is
            # now available is not current, even when the text is unchanged.
            # A report from another judge model is checked again in full.
            existing = entity_data.get("profile_grounding")
            if judged_by(existing, model) and grounding_is_current(
                existing, profile_text, article_texts
            ):
                counts["skipped_unchanged"] += 1
                continue
            jobs.append((entity_type, entity_key, entity_data, profile_text))

    # Plan every profile first, then ask about each cited article once for
    # all the profiles that cite it (``judge_pending()``)
    plans = [
        plan_grounding(
            profile_text=profile_text,
            article_texts=article_texts,
            previous=entity_data.get("profile_grounding")
            if judged_by(entity_data.get("profile_grounding"), model)
            else None,
            entity_name=_display_name(entity_key, entity_data),
        )
        for _, entity_key, entity_data, profile_text in jobs
    ]
    counts["llm_calls"] = judge_pending(
        plans, article_texts, model_type=model_type, workers=workers
    )
    for (entity_type, entity_key, entity_data, _), plan in zip(jobs, plans):
        report = finish_report(plan)
        report.model = model
        entity_data["profile_grounding"] = report.model_dump(exclude_none=True)
        counts["verified"] += 1
        counts["llm_checks"] += report.checked_articles
        counts["reused_checks"] += report.reused_articles
        if report.grounding_score is not None:
            log(
                f"Grounding for {entity_type}/{entity_key}: "
                f"score={report.grounding_score:.2f} "
                f"({describe_claim_counts(claim_counts(report))})",
                level="info",
            )

    counts.update(stored_claim_totals(entities))
    return counts


def stored_claim_totals(entities: Dict[str, Dict]) -> Dict[str, int]:
    """Claims per verdict summed over every entity's stored grounding report."""
    totals = {k: 0 for k in COUNTED_LEVELS}
    problems = failed = 0
    for entity_dict in entities.values():
        for entity_data in entity_dict.values():
            report = entity_data.get("profile_grounding")
            per = claim_counts(report)
            for k, n in per.items():
                totals[k] += n
            problems += any(n for k, n in per.items() if k != "supported")
            sources = (report or {}).get("sources") or []
            failed += any(src.get("status") == "error" for src in sources)
    return {
        **totals,
        "claims": sum(totals.values()),
        "profiles_with_problems": problems,
        # checks that failed (server gone, bad reply): asked again next run
        "profiles_with_failed_checks": failed,
    }
