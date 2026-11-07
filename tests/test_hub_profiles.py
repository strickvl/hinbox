"""Frozen hub profiles: later articles are linked, not written in
(src/engine/hub_profiles.py)."""

from typing import Any, List

from src.engine import profile_stubs
from src.engine.hub_profiles import is_hub, link_article
from src.engine.profiles import VersionedProfile

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
        "guantanamo", article_texts={"a0": "US stub."},
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
