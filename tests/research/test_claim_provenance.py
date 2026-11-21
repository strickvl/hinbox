from datetime import datetime, timezone

from src.config_loader import DomainConfig
from src.research.extraction import ObservationManifest


def manifest(**updates) -> ObservationManifest:
    values = {
        "task_id": "release-zero/assertions",
        "workspace_revision_id": "revision_1",
        "representation_id": "representation_1",
        "source_unit_id": "unit_1",
        "candidate_context": {"passage_candidates": ["candidate_1"]},
        "accepted_state_before": {"accepted_record_ids": []},
        "prior_human_decisions": [{"decision": "reject", "record_id": "old_1"}],
        "available_tool_results": [],
        "source_created_or_published_at": "2002-01-15",
        "model": "local/test-model",
        "prompt_version": "passages-v1+assertions-v1",
        "task_version": "1",
        "schema_version": "1",
        "code_version": "git:test",
        "observed_at": datetime(2026, 7, 15, tzinfo=timezone.utc),
    }
    values.update(updates)
    return ObservationManifest(**values)


def test_idempotency_key_covers_context_and_all_behavior_versions():
    original = manifest()
    assert original.request_key == manifest().request_key

    for field, value in (
        ("workspace_revision_id", "revision_2"),
        ("candidate_context", {"passage_candidates": ["candidate_2"]}),
        ("model", "local/new-model"),
        ("prompt_version", "passages-v2"),
        ("task_version", "2"),
        ("schema_version", "2"),
        ("code_version", "git:new"),
    ):
        assert original.request_key != manifest(**{field: value}).request_key

    assert original.attempt_key(1) != original.attempt_key(2)


def test_record_time_does_not_replace_source_or_occurrence_time():
    observation = manifest(source_created_or_published_at="2002-01-15")

    assert observation.source_created_or_published_at == "2002-01-15"
    assert observation.observed_at == datetime(2026, 7, 15, tzinfo=timezone.utc)


def test_research_prompts_and_versions_use_a_separate_namespace():
    config = DomainConfig("guantanamo")

    assert "Do not write or update a profile" in config.load_research_prompt("passages")
    assert "source assertions" in config.load_research_prompt("assertions")
    assert config.get_research_proposal_config()["cache_namespace"] == (
        "research_proposals"
    )
