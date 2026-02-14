"""Engine package entrypoint exposing the stable API surface.

This module re-exports the core engine primitives from their current locations.
It exists to provide a forward-compatible import path (src.engine.*) while the
internal organization is being refactored. There are no logic changes here;
imports are intentionally explicit to make the public API clear and stable.
"""

from src.engine.article_processor import ArticleProcessor
from src.engine.extractors import EntityExtractor
from src.engine.match_checker import (
    MatchCheckResult,
    cloud_model_check_match,
    local_model_check_match,
)
from src.engine.merge_dispute_agent import (
    MergeDisputeAction,
    MergeDisputeDecision,
    run_merge_dispute_agent,
)
from src.engine.mergers import EntityMerger
from src.engine.profiles import (
    ProfileVersion,
    VersionedProfile,
    create_profile,
    create_profile_with_outcome,
    update_profile,
    update_profile_with_outcome,
)
from src.engine.relevance import (
    gemini_check_relevance,
    ollama_check_relevance,
)

__all__ = [
    # Extractors
    "EntityExtractor",
    # Orchestrator
    "ArticleProcessor",
    # Relevance
    "gemini_check_relevance",
    "ollama_check_relevance",
    # Profiles
    "ProfileVersion",
    "VersionedProfile",
    "create_profile",
    "create_profile_with_outcome",
    "update_profile",
    "update_profile_with_outcome",
    # Match checking
    "MatchCheckResult",
    "cloud_model_check_match",
    "local_model_check_match",
    # Merge dispute agent
    "MergeDisputeAction",
    "MergeDisputeDecision",
    "run_merge_dispute_agent",
    # Merging
    "EntityMerger",
]
