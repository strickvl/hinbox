"""End-of-run summary lines: article counts, entity counts, reflection stats."""

from typing import Dict, List, Tuple

from src.logging_config import log


def calculate_reflection_statistics(
    processed_rows: List[Dict], processed_count: int
) -> Tuple[int, float]:
    """Calculate reflection statistics from processed articles."""
    total_reflection_attempts = 0
    for row in processed_rows:
        if (
            "processing_metadata" in row
            and row["processing_metadata"] is not None
            and "reflection_summary" in row["processing_metadata"]
            and row["processing_metadata"]["reflection_summary"] is not None
        ):
            total_reflection_attempts += row["processing_metadata"][
                "reflection_summary"
            ].get("total_attempts", 0)

    avg_reflections = (
        total_reflection_attempts / processed_count if processed_count > 0 else 0
    )
    return total_reflection_attempts, avg_reflections


def log_processing_summary(
    article_count: int,
    processed_count: int,
    skipped_relevance_count: int,
    skipped_already_processed: int,
    entities: Dict[str, Dict],
    processed_rows: List[Dict],
    skipped_relevance_error: int = 0,
    skipped_relevance_cached: int = 0,
    skipped_incomplete: int = 0,
) -> None:
    """Log comprehensive processing summary and statistics."""
    log("\n[bold]Processing complete[/bold]", level="info")

    # Article processing summary
    log(f"Articles read: {article_count}", level="info")
    log(f"Articles processed: {processed_count}", level="info")
    log(f"Articles skipped (relevance): {skipped_relevance_count}", level="info")
    if skipped_relevance_cached:
        log(
            f"Articles skipped (not relevant in an earlier run, unchanged, "
            f"no LLM call): {skipped_relevance_cached}",
            level="info",
        )
    if skipped_relevance_error:
        log(
            f"Articles skipped (relevance check failed, will retry next run): "
            f"{skipped_relevance_error}",
            level="warning",
        )
    if skipped_incomplete:
        log(
            f"Articles left unprocessed (a model call failed, will retry next "
            f"run): {skipped_incomplete}",
            level="warning",
        )
    log(
        f"Articles skipped (already processed): {skipped_already_processed}",
        level="info",
    )

    # Entity counts
    log("\nFinal entity counts:", level="info")
    log(f"• People: {len(entities['people'])}", level="info")
    log(f"• Organizations: {len(entities['organizations'])}", level="info")
    log(f"• Locations: {len(entities['locations'])}", level="info")
    log(f"• Events: {len(entities['events'])}", level="info")

    # Reflection statistics
    if processed_count > 0:
        total_reflection_attempts, avg_reflections = calculate_reflection_statistics(
            processed_rows, processed_count
        )
        if total_reflection_attempts > 0:
            log("\nReflection statistics:", level="info")
            log(
                f"• Total reflection attempts: {total_reflection_attempts}",
                level="info",
            )
            log(
                f"• Average reflection attempts per article: {avg_reflections:.2f}",
                level="info",
            )
