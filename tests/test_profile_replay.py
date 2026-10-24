"""Offline tests for the profile replay script (no model calls)."""

import argparse
import json

import pytest

import src.engine.profiles as profiles
from src.engine.profiles import VersionedProfile
from src.evals import profile_replay as pr

ENTITY = pr.ReplayEntity("Abu Zubaydah", "person", ("Abu Zubaydah", "Zubaydah"))


def _article(aid, content, date, title=""):
    return {"id": aid, "content": content, "published_date": date, "title": title}


def test_count_mentions_does_not_double_count_overlapping_terms():
    assert pr.count_mentions("Abu Zubaydah was held. ZUBAYDAH said", ENTITY.terms) == 2


def test_select_articles_orders_by_date_and_splits_passing_mentions():
    articles = [
        _article("late", "Zubaydah ... Zubaydah", 300),
        _article("early", "Abu Zubaydah and Zubaydah again", 100),
        _article("passing", "a brief mention of Zubaydah", 200),
        _article("titled", "Zubaydah once", 250, title="Abu Zubaydah's lawyers"),
        _article("none", "nothing relevant", 50),
    ]
    chosen, passing = pr.select_articles(articles, ENTITY, max_articles=10)
    assert [a["id"] for a in chosen] == ["early", "titled", "late"]
    assert [a["id"] for a in passing] == ["passing"]
    chosen, _ = pr.select_articles(articles, ENTITY, max_articles=2)
    assert [a["id"] for a in chosen] == ["early", "titled"]


def test_parse_entity_spec():
    spec = pr.parse_entity_spec("Camp X | location | Camp X, Camp Ex")
    assert spec == pr.ReplayEntity("Camp X", "location", ("Camp X", "Camp Ex"))
    with pytest.raises(argparse.ArgumentTypeError):
        pr.parse_entity_spec("only a name")


@pytest.fixture
def fake_profiles(monkeypatch):
    """Replace the model calls with a deterministic fake that records its calls."""
    calls = []

    def fake_create(entity_type, name, text, article_id, domain, model):
        calls.append(("create", article_id, model))
        profile = {"text": f"{name}^[{article_id}]", "sources": [article_id]}
        versioned = VersionedProfile()
        versioned.add_version(profile_data=profile, trigger_article_id=article_id)
        return profile, versioned, []

    def fake_update(entity_type, name, profile, versioned, text, article_id, **kw):
        calls.append(("update", article_id, kw["model"]))
        if "FAIL" in text:  # mimic update_profile falling back to the old profile
            return profile, versioned, []
        new = {
            "text": profile["text"] + f" more^[{article_id}]",
            "sources": profile["sources"] + [article_id],
        }
        versioned.add_version(profile_data=new, trigger_article_id=article_id)
        return new, versioned, [{"valid": True}]

    monkeypatch.setattr(profiles, "create_profile", fake_create)
    monkeypatch.setattr(profiles, "update_profile", fake_update)
    return calls


def test_replay_saves_each_step_and_resumes(tmp_path, fake_profiles):
    chosen = [{"id": f"a{i}"} for i in range(1, 4)]
    texts = {"a1": "x", "a2": "FAIL", "a3": "y"}
    entity_dir = str(tmp_path)

    assert pr.replay_entity(ENTITY, chosen[:2], texts, entity_dir, "m", "d") == 2
    step2 = json.loads((tmp_path / "step_002.json").read_text())
    assert step2["unchanged"] is True  # the failed update is flagged
    assert step2["article"] == {"id": "a2"}

    # A later run with more articles picks up from step 2: no repeated calls
    assert pr.replay_entity(ENTITY, chosen, texts, entity_dir, "m", "d") == 3
    assert [c[:2] for c in fake_profiles] == [
        ("create", "a1"),
        ("update", "a2"),
        ("update", "a3"),
    ]
    step3 = json.loads((tmp_path / "step_003.json").read_text())
    assert step3["unchanged"] is False
    assert step3["profile"]["sources"] == ["a1", "a3"]
    assert len(step3["profile_versions"]["versions"]) == 2
    assert all(c[2] == "m" for c in fake_profiles)


def test_resolve_mode_prefers_explicit_model():
    from src.utils.llm import GenerationMode

    assert profiles._resolve_mode("gemini", "ollama/qwen") == GenerationMode.LOCAL
    assert profiles._resolve_mode("ollama", "openai/gpt") == GenerationMode.CLOUD
    assert profiles._resolve_mode("gemini", None) == GenerationMode.CLOUD
    assert profiles._resolve_mode("ollama", None) == GenerationMode.LOCAL


def test_replay_refuses_to_resume_a_different_article_order(tmp_path, fake_profiles):
    texts = {"a1": "x", "a2": "y", "b1": "z"}
    pr.replay_entity(
        ENTITY, [{"id": "a1"}, {"id": "a2"}], texts, str(tmp_path), "m", "d"
    )
    with pytest.raises(ValueError, match="step 1 for article a1"):
        pr.replay_entity(ENTITY, [{"id": "b1"}], texts, str(tmp_path), "m", "d")
    # A shorter run over the same order is fine (nothing left to do)
    assert pr.replay_entity(ENTITY, [{"id": "a1"}], texts, str(tmp_path), "m", "d") == 2


def test_profile_draft_uses_the_profile_token_limit(monkeypatch):
    seen = {}

    def fake_cloud(messages, response_model, model, max_tokens, temperature):
        seen["max_tokens"] = max_tokens
        return None

    monkeypatch.setattr(profiles, "cloud_generation", fake_cloud)
    from src.utils.llm import GenerationMode

    profiles._generate_initial_profile_text([], GenerationMode.CLOUD, "openai/x")
    assert seen["max_tokens"] == profiles.PROFILE_MAX_TOKENS > 2048
