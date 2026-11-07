"""Stub profiles for once-mentioned entities (src/engine/profile_stubs.py)."""

from contextlib import ExitStack
from typing import Dict, List
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pytest

from src.engine import EntityMerger, MatchCheckResult
from src.engine.profile_stubs import (
    build_stub_profile,
    is_stub,
    mention_sentences,
    split_sentences,
    stubs_enabled,
)
from tests.test_entity_merger_merge_smoke import (
    StubEmbeddingManager,
    create_profile_stub,
    make_empty_entities,
    update_profile_stub,
)

ART1 = (
    "Lawyers met on Monday. Khalid Shaikh Mohammed appeared in court in a "
    "white turban. The judge, Col. Pohl, delayed the hearing.\n\n"
    "Mohammed did not speak. Prosecutors said the trial could start in 2027."
)
ART2 = "Khalid Shaikh Mohammed asked for a new lawyer, his defense team said."


def test_split_sentences_handles_quotes_and_paragraphs():
    text = 'He said "no." Then he left.\n\nNew paragraph here'
    assert split_sentences(text) == [
        'He said "no."',
        "Then he left.",
        "New paragraph here",
    ]


def test_split_sentences_keeps_titles_and_initials_inside_a_sentence():
    text = "The judge, Col. James L. Pohl, ruled. The U.S. Navy agreed."
    assert split_sentences(text) == [
        "The judge, Col. James L. Pohl, ruled.",
        "The U.S. Navy agreed.",
    ]


def test_mention_sentences_full_name_first_then_surname():
    names = ["Khalid Shaikh Mohammed"]
    assert mention_sentences(ART1, "people", names) == [
        "Khalid Shaikh Mohammed appeared in court in a white turban.",
        "Mohammed did not speak.",
    ]
    # Only people get the surname fallback
    assert mention_sentences(ART1, "organizations", ["Shaikh Mohammed Group"]) == []
    assert mention_sentences(ART1, "people", names, limit=1) == [
        "Khalid Shaikh Mohammed appeared in court in a white turban."
    ]


def test_mention_sentences_skips_surname_when_a_relative_shares_it():
    text = (
        "Marine Col. Keith Parrella called a recess. Parrella was airlifted "
        "from the base.\n\n“He needs surgery,” Jennifer Parrella said."
    )
    assert mention_sentences(text, "people", ["Jennifer Parrella"]) == [
        "“He needs surgery,” Jennifer Parrella said."
    ]
    # A rank or a sentence ending before the surname is not another person
    text = "Lawyers spoke in Kenya.\n\nWiltsie argued. Attorney Ronald Wiltsie sat."
    assert mention_sentences(text, "people", ["Ronald Wiltsie"]) == [
        "Wiltsie argued.",
        "Attorney Ronald Wiltsie sat.",
    ]


def test_mention_sentences_ignores_aliases_broader_than_the_name():
    text = "Christo changed Miami forever. Monica Harvey, Miami Shores"
    names = ["Miami Shores", "Miami"]
    assert mention_sentences(text, "locations", names) == [
        "Monica Harvey, Miami Shores"
    ]
    # An alias with words of its own still counts
    text = "The Central Intelligence Agency declined. The CIA said nothing."
    names = ["Central Intelligence Agency", "CIA"]
    assert len(mention_sentences(text, "organizations", names)) == 2


def test_mention_sentences_matches_whole_words_only():
    text = "The Pentagon spoke. Pentagonal shapes are odd."
    assert mention_sentences(text, "organizations", ["Pentagon"]) == [
        "The Pentagon spoke."
    ]


def test_build_stub_profile_cites_every_block():
    stub = build_stub_profile(
        "people", "Khalid Shaikh Mohammed", {"tags": ["detainee"]}, ART1, "a1"
    )
    assert is_stub(stub)
    assert stub["sources"] == ["a1"]
    assert stub["tags"] == ["detainee", "person"]
    assert stub["text"].startswith("### Mentions")
    for para in stub["text"].split("\n\n")[1:]:
        assert para.endswith("^[a1]")


def test_event_stub_starts_with_extracted_description():
    stub = build_stub_profile(
        "events",
        ("Arraignment of KSM", "2008-06-05"),
        {"description": "KSM was arraigned at Camp Justice."},
        "Nothing here names it.",
        "a9",
    )
    assert stub["text"] == "### Summary\n\nKSM was arraigned at Camp Justice.^[a9]"


def test_stub_with_no_matching_sentence_still_has_a_citation():
    stub = build_stub_profile("locations", ("Camp X", "camp"), {}, "Unrelated.", "a3")
    assert stub["text"] == "Camp X is mentioned in this article.^[a3]"


def test_stubs_env_switch(monkeypatch):
    monkeypatch.delenv("HINBOX_PROFILE_STUBS", raising=False)
    assert stubs_enabled()
    monkeypatch.setenv("HINBOX_PROFILE_STUBS", "off")
    assert not stubs_enabled()


def test_stub_flag_survives_arrow_and_full_profiles_read_as_not_stubs():
    rows = [{"profile": {"text": "full"}}, {"profile": {"text": "s", "stub": True}}]
    back = pa.Table.from_pylist(rows).to_pylist()
    assert [is_stub(r["profile"]) for r in back] == [False, True]


# ── Through the merger ──


def _merge(
    entities,
    article_id: str,
    content: str,
    article_texts: Dict[str, str],
    names: List[str],
    create: MagicMock,
    update: MagicMock,
):
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
        return EntityMerger("people").merge_entities(
            [{"name": n} for n in names],
            entities,
            article_id=article_id,
            article_title="T",
            article_url="u",
            article_published_date="2025-02-01",
            article_content=content,
            extraction_timestamp="2025-02-02T00:00:00Z",
            model_type="gemini",
            similarity_threshold=0.5,
            domain="guantanamo",
            article_texts=article_texts,
        )


@pytest.fixture
def mocks():
    return (
        MagicMock(side_effect=create_profile_stub),
        MagicMock(side_effect=update_profile_stub),
    )


def test_first_mention_writes_a_stub_without_model_calls(monkeypatch, mocks):
    monkeypatch.delenv("HINBOX_PROFILE_STUBS", raising=False)
    create, update = mocks
    entities = make_empty_entities()
    stats = _merge(
        entities, "a1", ART1, {"a1": ART1}, ["Khalid Shaikh Mohammed"], create, update
    )
    ksm = entities["people"]["Khalid Shaikh Mohammed"]
    assert stats.new == 1 and create.call_count == 0 and update.call_count == 0
    assert is_stub(ksm["profile"])
    assert ksm["profile_versions"]["versions"] == []
    assert ksm["profile_embedding"]  # the stub text is embedded like any profile


def test_second_mention_creates_from_first_article_then_updates(monkeypatch, mocks):
    monkeypatch.delenv("HINBOX_PROFILE_STUBS", raising=False)
    create, update = mocks
    entities = make_empty_entities()
    texts = {"a1": ART1, "a2": ART2}
    _merge(entities, "a1", ART1, texts, ["Khalid Shaikh Mohammed"], create, update)
    stats = _merge(
        entities, "a2", ART2, texts, ["Khalid Shaikh Mohammed"], create, update
    )
    ksm = entities["people"]["Khalid Shaikh Mohammed"]
    assert stats.merged == 1 and stats.failed_profiles == 0
    # Created from article 1's own text, then updated with article 2
    assert create.call_count == 1
    assert create.call_args.args[:4] == ("person", "Khalid Shaikh Mohammed", ART1, "a1")
    assert update.call_count == 1
    assert update.call_args.args[4:6] == (ART2, "a2")
    assert not is_stub(ksm["profile"])
    assert ksm["profile"]["sources"] == ["a1", "a2"]
    assert ksm["profile_versions"]["current_version"] == 2
    assert [a["article_id"] for a in ksm["articles"]] == ["a1", "a2"]


def test_missing_first_article_updates_the_stub_itself(monkeypatch, mocks):
    monkeypatch.delenv("HINBOX_PROFILE_STUBS", raising=False)
    create, update = mocks
    entities = make_empty_entities()
    _merge(entities, "a1", ART1, {}, ["Khalid Shaikh Mohammed"], create, update)
    _merge(entities, "a2", ART2, {}, ["Khalid Shaikh Mohammed"], create, update)
    assert create.call_count == 0
    assert is_stub(update.call_args.args[2])
    assert not is_stub(entities["people"]["Khalid Shaikh Mohammed"]["profile"])


def test_failed_update_after_fill_keeps_the_stub(monkeypatch, mocks):
    monkeypatch.delenv("HINBOX_PROFILE_STUBS", raising=False)
    create, _ = mocks
    entities = make_empty_entities()
    texts = {"a1": ART1, "a2": ART2}
    # The real update_profile hands back the profile it got when the call fails
    failing = MagicMock(side_effect=lambda _t, _n, p, v, *a, **k: (p, v, []))
    _merge(entities, "a1", ART1, texts, ["Khalid Shaikh Mohammed"], create, failing)
    stats = _merge(
        entities, "a2", ART2, texts, ["Khalid Shaikh Mohammed"], create, failing
    )
    ksm = entities["people"]["Khalid Shaikh Mohammed"]
    assert stats.failed_profiles == 1
    assert is_stub(ksm["profile"])  # tried again in full when a2 is rerun
    assert [a["article_id"] for a in ksm["articles"]] == ["a1"]


def test_call_count_with_and_without_stubs(monkeypatch):
    """Three people, one of them in two articles: 2 profile calls instead of 4."""

    def run() -> int:
        create = MagicMock(side_effect=create_profile_stub)
        update = MagicMock(side_effect=update_profile_stub)
        entities = make_empty_entities()
        texts = {"a1": ART1, "a2": ART2, "a3": "Alice Brown filed a motion."}
        # Orthogonal vectors keep the three people apart
        vecs = {
            "Khalid Shaikh Mohammed": [1.0, 0, 0],
            "Col. Pohl": [0, 1.0, 0],
            "Alice Brown": [0, 0, 1.0],
        }
        for aid, names in (
            ("a1", ["Khalid Shaikh Mohammed", "Col. Pohl"]),
            ("a2", ["Khalid Shaikh Mohammed"]),
            ("a3", ["Alice Brown"]),
        ):
            for name in names:
                manager = StubEmbeddingManager(vec=vecs[name])
                with patch(
                    "src.engine.mergers.get_embedding_manager", return_value=manager
                ):
                    _merge_one(entities, aid, texts[aid], texts, name, create, update)
        assert len(entities["people"]) == 3
        return create.call_count + update.call_count

    monkeypatch.setenv("HINBOX_PROFILE_STUBS", "off")
    assert run() == 4
    monkeypatch.setenv("HINBOX_PROFILE_STUBS", "on")
    assert run() == 2


def _merge_one(entities, aid, content, texts, name, create, update):
    match = MatchCheckResult(is_match=True, confidence=0.95, reason="same")
    with ExitStack() as stack:
        stack.enter_context(
            patch("src.engine.mergers.cloud_model_check_match", return_value=match)
        )
        stack.enter_context(patch("src.engine.mergers.create_profile", create))
        stack.enter_context(patch("src.engine.profile_stubs.create_profile", create))
        stack.enter_context(patch("src.engine.profile_stubs.update_profile", update))
        EntityMerger("people").merge_entities(
            [{"name": name}],
            entities,
            article_id=aid,
            article_title="T",
            article_url="u",
            article_published_date="2025-02-01",
            article_content=content,
            extraction_timestamp="2025-02-02T00:00:00Z",
            model_type="gemini",
            similarity_threshold=0.9,
            domain="guantanamo",
            article_texts=texts,
        )


def test_merge_eval_can_compare_against_stubs(tmp_path):
    import pyarrow.parquet as pq

    from src.evals.datasets import EvalItem, stub_existing_profiles

    path = tmp_path / "articles.parquet"
    pq.write_table(pa.Table.from_pylist([{"id": "a1", "content": ART1}]), path)
    items = [
        EvalItem(
            "x",
            True,
            {
                "existing_name": "Khalid Shaikh Mohammed",
                "existing_profile_text": "KSM is a detainee.^[a1]",
                "entity_type": "people",
            },
        ),
        EvalItem(
            "y",
            False,
            {
                "existing_name": "Bob",
                "existing_profile_text": "Bob.^[gone]",
                "entity_type": "people",
            },
        ),
    ]
    out, swapped = stub_existing_profiles(items, str(path))
    assert swapped == 1
    assert out[0].inputs["existing_profile_text"].startswith("### Mentions")
    assert out[1].inputs["existing_profile_text"] == "Bob.^[gone]"
