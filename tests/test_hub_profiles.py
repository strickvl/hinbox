"""Frozen hub profiles: later articles are linked, not written in
(src/engine/hub_profiles.py)."""

from contextlib import ExitStack
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from src.engine import EntityMerger, MatchCheckResult, profile_stubs
from src.engine.hub_profiles import is_hub, link_article
from src.engine.profiles import VersionedProfile
from tests.test_entity_merger_merge_smoke import (
    StubEmbeddingManager,
    create_profile_stub,
    make_empty_entities,
    update_profile_stub,
)

ART1 = "The United States held 166 men at Guantanamo in 2013, officials said."
ART2 = "Lawyers met on Monday. The United States moved two men to Oman."


def test_link_article_keeps_text_and_records_the_article():
    profile = {"text": "Old.^[a0]", "sources": ["a0"], "stub": True}
    out = link_article(profile, "a1")
    assert out["text"] == "Old.^[a0]" and out["stub"] is True
    assert out["sources"] == ["a0", "a1"] and out["linked_only"] == ["a1"]
    assert link_article(out, "a1")["linked_only"] == ["a1"]  # no repeats
    assert profile["sources"] == ["a0"]  # not changed in place


def test_is_hub_reads_domain_config(monkeypatch):
    assert is_hub("guantanamo", "location", "country")
    assert is_hub("guantanamo", "locations", "region")
    assert not is_hub("guantanamo", "location", "detention_facility")
    assert not is_hub("guantanamo", "person", "country")
    assert not is_hub("guantanamo", "location", None)
    assert is_hub("guantanamo", "location", "LocationType.COUNTRY")
    monkeypatch.setenv("HINBOX_HUB_LINK_ONLY", "off")
    assert not is_hub("guantanamo", "location", "country")


def _no_model(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("a hub update must not call the model")


def test_update_or_fill_hub_makes_no_model_call(monkeypatch):
    monkeypatch.setattr(profile_stubs, "update_profile", _no_model)
    monkeypatch.setattr(profile_stubs, "create_profile", _no_model)
    versions = VersionedProfile()
    stub = {"text": "US stub.^[a0]", "sources": ["a0"], "stub": True}
    updated, out_versions, history = profile_stubs.update_or_fill(
        "location", "United States", stub, versions, ART2, "a1", "cloud",
        "guantanamo", article_texts={"a0": ART1},
        entity={"name": "United States", "type": "country"},
    )  # fmt: skip
    assert updated["text"] == stub["text"]
    assert updated["sources"] == ["a0", "a1"]
    assert out_versions is versions and versions.versions == []
    assert history == []


def test_update_or_fill_non_hub_still_calls_the_updater(monkeypatch):
    calls: List[str] = []

    def fake_update(*args: Any, **kwargs: Any):
        calls.append(args[1])
        return {"text": "new", "sources": ["a0", "a1"]}, args[3], []

    monkeypatch.setattr(profile_stubs, "update_profile", fake_update)
    profile_stubs.update_or_fill(
        "location", "Camp Delta", {"text": "Old.^[a0]", "sources": ["a0"]},
        VersionedProfile(), ART2, "a1", "cloud", "guantanamo",
        entity={"name": "Camp Delta", "type": "detention_facility"},
    )  # fmt: skip
    assert calls == ["Camp Delta"]


def _merge(entities: Dict, aid: str, content: str, kind: str, create, update):
    match = MatchCheckResult(is_match=True, confidence=0.95, reason="same")
    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "src.engine.mergers.get_embedding_manager",
                return_value=StubEmbeddingManager(vec=[0.42, 0.58, 0.0]),
            )
        )
        stack.enter_context(
            patch("src.engine.mergers.cloud_model_check_match", return_value=match)
        )
        stack.enter_context(patch("src.engine.mergers.create_profile", create))
        stack.enter_context(patch("src.engine.profile_stubs.create_profile", create))
        stack.enter_context(patch("src.engine.profile_stubs.update_profile", update))
        return EntityMerger("locations").merge_entities(
            [{"name": "United States", "type": kind}],
            entities,
            article_id=aid,
            article_title="T",
            article_url="u",
            article_published_date="2025-02-01",
            article_content=content,
            extraction_timestamp="2025-02-02T00:00:00Z",
            model_type="gemini",
            similarity_threshold=0.5,
            domain="guantanamo",
            article_texts={"a1": ART1, "a2": ART2},
        )


@pytest.mark.parametrize("kind, hub", [("country", True), ("city", False)])
def test_second_article_for_a_hub_is_linked_without_model_calls(monkeypatch, kind, hub):
    monkeypatch.delenv("HINBOX_PROFILE_STUBS", raising=False)
    monkeypatch.delenv("HINBOX_HUB_LINK_ONLY", raising=False)
    create = MagicMock(side_effect=create_profile_stub)
    update = MagicMock(side_effect=update_profile_stub)
    entities = make_empty_entities()
    _merge(entities, "a1", ART1, kind, create, update)
    first_text = entities["locations"][("United States", kind)]["profile"]["text"]
    stats = _merge(entities, "a2", ART2, kind, create, update)
    us = entities["locations"][("United States", kind)]
    assert stats.merged == 1 and stats.failed_profiles == 0
    assert [a["article_id"] for a in us["articles"]] == ["a1", "a2"]
    assert us["profile"]["sources"] == ["a1", "a2"]
    if hub:
        assert create.call_count == 0 and update.call_count == 0
        assert us["profile"]["text"] == first_text
        assert us["profile"]["linked_only"] == ["a2"]
    else:
        assert create.call_count == 1 and update.call_count == 1
