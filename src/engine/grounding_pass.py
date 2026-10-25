"""Profile grounding as a post-processing pass over all changed profiles.

Each profile's check is independent (it reads its own text and the cited
articles), so the checks run in a thread pool. The LLM concurrency limiters
in ``src/utils/llm.py`` still cap how many calls are in flight. Results are
written back to the entity dicts on the calling thread.
"""

import contextvars
import hashlib
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple

from src.engine.article_processor import ArticleProcessor
from src.logging_config import log
from src.utils.quality_controls import (
    CITATION_RE,
    GroundingReport,
    verify_profile_grounding,
)


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
    entity as entity["profile_grounding"]. Skips entities whose profile
    hasn't changed since the last verification (hash-based). Up to
    ``workers`` profiles are checked at once.

    Returns summary counts of verified/skipped/failed entities.
    """
    # Build article_id -> content map from rows
    article_texts: Dict[str, str] = {}
    for i, row in enumerate(rows):
        info = processor.prepare_article_info(row, i)
        if info["id"] and info["content"]:
            article_texts[info["id"]] = info["content"]

    counts = {"verified": 0, "skipped_unchanged": 0, "skipped_no_citations": 0}
    jobs: List[Tuple[str, str, Dict[str, Any], str]] = []

    for entity_type, entity_dict in entities.items():
        for entity_key, entity_data in entity_dict.items():
            profile_text = (entity_data.get("profile") or {}).get("text", "")
            if not profile_text:
                continue

            # Check if profile has citations worth verifying
            if not CITATION_RE.findall(profile_text):
                counts["skipped_no_citations"] += 1
                continue

            # Skip if profile hasn't changed since last grounding check
            current_hash = hashlib.sha256(profile_text.encode()).hexdigest()
            existing_grounding = entity_data.get("profile_grounding")
            if (
                isinstance(existing_grounding, dict)
                and existing_grounding.get("profile_text_hash") == current_hash
            ):
                counts["skipped_unchanged"] += 1
                continue
            jobs.append((entity_type, entity_key, entity_data, profile_text))

    def verify(profile_text: str) -> GroundingReport:
        return verify_profile_grounding(
            profile_text=profile_text,
            article_texts=article_texts,
            model_type=model_type,
        )

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        # copy_context() so each check's timing nests under the caller's step
        futures = [
            pool.submit(contextvars.copy_context().run, verify, job[3]) for job in jobs
        ]
        for (entity_type, entity_key, entity_data, _), future in zip(jobs, futures):
            report = future.result()
            entity_data["profile_grounding"] = report.model_dump(exclude_none=True)
            counts["verified"] += 1
            if report.grounding_score is not None:
                log(
                    f"Grounding for {entity_type}/{entity_key}: "
                    f"score={report.grounding_score:.2f}, "
                    f"verified={report.verified}/{report.total_citations}",
                    level="info",
                )

    return counts
