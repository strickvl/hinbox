"""Tagging notes as about the entity or not, and what uses the tags.

Made-up pages only (the repository is public).
"""

import json

import pyarrow as pa
import pyarrow.parquet as pq

from src.engine.edit_guards import guard_edits
from src.engine.note_relevance import (
    NoteVerdict,
    NoteVerdicts,
    notes,
    plan,
    read_reply,
    run_note_relevance,
)
from src.engine.profile_edits import ProfileEdit
from src.frontend.profile_render import render_profile
from src.utils.corrections import sentence_key
from src.utils.file_ops import load_existing_entities
from src.utils.note_tags import MODEL_KEY, TAGS_KEY, off_topic_keys, stored_tags
from src.utils.profile_text import repair_text

# A made-up city page: one note about the city, two about a man arrested there
CITY = (
    "Port Elm is a harbour city in the north.^[a1]\n\n"
    "## Arrests\n\n"
    "Police arrested Tom Vale in Port Elm in 2004.^[a2] "
    "Tom Vale was 19 years old when he was arrested.^[a2]\n\n"
    "Tom Vale later studied law in another country.^[a3]\n"
)
ABOUT = "Police arrested Tom Vale in Port Elm in 2004."
AGE = "Tom Vale was 19 years old when he was arrested."
LAW = "Tom Vale later studied law in another country."


def _reply(*rows):
    return NoteVerdicts(
        verdicts=[
            NoteVerdict(sentence=n, starts_with=w, about_what=a, verdict=v)
            for n, w, a, v in rows
        ]
    )


# --- the no-model text repair -------------------------------------------


def test_escaped_newlines_become_real_ones():
    text = "Ann Lee was a judge.^[a1]\\n\\n### Later life\\nShe retired.^[a2]"
    assert repair_text(text) == (
        "Ann Lee was a judge.^[a1]\n\n### Later life\nShe retired.^[a2]"
    )
    assert repair_text("Tabbed\\tword.^[a1]") == "Tabbed word.^[a1]"


def test_a_doubled_marker_is_dropped():
    assert repair_text("She retired.^[a2].^[a2] Next.^[a3]") == (
        "She retired.^[a2]. Next.^[a3]"
    )
    assert repair_text("She retired.^[a2]^[a2]^[a2]") == "She retired.^[a2]"
    # Two different sources are two citations, kept
    assert repair_text("She retired.^[a2]^[a3]") == "She retired.^[a2]^[a3]"
    assert repair_text("") == ""


def test_saved_profiles_are_mended_as_a_run_loads_them(tmp_path):
    row = {
        "name": "Ann Lee",
        "type": "judge",
        "profile": {
            "text": "Ann Lee was a judge.^[a1]\\n\\n### Later life\\nShe retired.^[a2]"
        },
    }
    pq.write_table(pa.Table.from_pylist([row]), tmp_path / "people.parquet")
    people = load_existing_entities(str(tmp_path))["people"]
    text = next(iter(people.values()))["profile"]["text"]
    assert "\\n" not in text and "\n\n### Later life\n" in text


# --- tagging ------------------------------------------------------------


def test_notes_carry_their_heading():
    items = notes(CITY)
    assert [i["key"] for i in items] == [
        sentence_key("Port Elm is a harbour city in the north."),
        sentence_key(ABOUT),
        sentence_key(AGE),
        sentence_key(LAW),
    ]
    assert [i["heading"] for i in items] == ["", "Arrests", "Arrests", "Arrests"]


def test_a_reply_whose_numbering_slipped_is_not_used():
    sentences = [ABOUT, AGE]
    reply = _reply(
        (1, "Police arrested Tom", "the arrest in Port Elm", "about"),
        (2, "Police arrested Tom", "Tom Vale's age", "other"),  # wrong first words
    )
    tags = read_reply(reply, sentences)
    assert tags[0] == {"verdict": "about", "about": "the arrest in Port Elm"}
    assert tags[1] is None
    # Unknown verdicts, numbers out of range and repeats are dropped
    reply = _reply(
        (1, "", "x", "maybe"),
        (3, "", "x", "about"),
        (2, "", "x", "Other."),
        (2, "", "y", "about"),
    )
    assert read_reply(reply, sentences) == [None, {"verdict": "other", "about": "x"}]


def _page(text=CITY, **profile):
    return {"name": "Port Elm", "type": "city", "profile": {"text": text, **profile}}


def test_the_pass_tags_each_note_once_and_stores_json():
    calls = []

    def ask(msgs):
        calls.append(msgs)
        if "Arrests" in msgs[-1]["content"]:
            return _reply(
                (1, "Police arrested Tom", "the arrest", "about"),
                (2, "Tom Vale was", "Tom Vale's age", "other"),
                (3, "Tom Vale later", "Tom Vale's studies", "other"),
            )
        return _reply((1, "Port Elm is", "the city", "about"))

    entities = {"locations": {("Port Elm", "city"): _page()}}
    counts = run_note_relevance(entities, "ollama/test", local=True, ask=ask)
    assert len(calls) == 2  # one call per heading
    assert counts["about"] == 2 and counts["other"] == 2 and counts["failed_calls"] == 0
    profile = entities["locations"][("Port Elm", "city")]["profile"]
    assert isinstance(profile[TAGS_KEY], str)  # a JSON string, not a struct
    assert profile[MODEL_KEY] == "ollama/test"
    assert sorted(off_topic_keys(profile)) == sorted(
        [sentence_key(AGE), sentence_key(LAW)]
    )
    assert profile["text"] == CITY  # no text changes
    # A second pass asks nothing
    calls.clear()
    run_note_relevance(entities, "ollama/test", local=True, ask=ask)
    assert calls == []


def test_a_failed_call_leaves_its_notes_for_the_next_run():
    def ask(msgs):
        raise RuntimeError("server down")

    entities = {"locations": {("Port Elm", "city"): _page()}}
    counts = run_note_relevance(entities, "ollama/test", local=True, ask=ask)
    assert counts["failed_calls"] == 2
    assert stored_tags(entities["locations"][("Port Elm", "city")]["profile"]) == {}


def test_tags_follow_the_text_and_the_model():
    tags = {
        sentence_key(AGE): {"verdict": "other"},
        "gone sentence": {"verdict": "other"},
    }
    page = _page(**{TAGS_KEY: json.dumps(tags), MODEL_KEY: "m1"})
    todo, kept = plan(page, "m1")
    assert list(kept) == [sentence_key(AGE)]  # a sentence no longer there drops
    assert len(todo) == 3
    todo, kept = plan(page, "m2")  # another model asks again
    assert kept == {} and len(todo) == 4


def test_link_only_hubs_are_not_tagged():
    def ask(msgs):
        raise AssertionError("no call for a hub")

    entities = {"locations": {("Elmland", "country"): _page(linked_only=True)}}
    assert run_note_relevance(entities, "m", local=False, ask=ask)["pages"] == 0


# --- disputes about something else --------------------------------------


def test_a_dispute_about_an_off_topic_note_is_dropped():
    edit = ProfileEdit(
        op="add",
        anchor=AGE,
        text=(
            "Sources disagree on Tom Vale's age: this report says he was 21 "
            "years old when he was arrested, not 19.^[a9]"
        ),
    )
    kept, report = guard_edits(CITY, [edit])
    assert len(kept) == 1 and report["off_topic_disputes"] == []
    kept, report = guard_edits(CITY, [edit], off_topic=[sentence_key(AGE)])
    assert kept == [] and len(report["off_topic_disputes"]) == 1


def test_a_dispute_about_the_entity_itself_is_kept():
    edit = ProfileEdit(
        op="add",
        anchor=ABOUT,
        text="Sources disagree: this report says police arrested Tom Vale in Port Elm in 2005, not 2004.^[a9]",
    )
    kept, report = guard_edits(CITY, [edit], off_topic=[sentence_key(AGE)])
    assert len(kept) == 1 and report["off_topic_disputes"] == []


# --- the reader ---------------------------------------------------------


def test_the_reader_hides_off_topic_notes_without_moving_anything():
    off = [sentence_key(AGE), sentence_key(LAW)]
    plain = render_profile(CITY)
    rp = render_profile(CITY, off_topic=off)
    assert rp.off_count() == 2
    assert [c.text for c in rp.claims] == [c.text for c in plain.claims]
    assert [c.index for c in rp.claims if c.off] == [2, 3]
    assert 'class="s v-supported off" id="claim-2"' in rp.html
    assert '<sup class="c off">' in rp.html
    # The full stop after a hidden note's marker hides with it
    assert rp.html.count('<span class="op">') == 0  # these stops sit before the markers
    rp2 = render_profile(
        "Port Elm is a city^[a1]. Tom Vale was born in 1985^[a2]. More.^[a1]",
        off_topic=[sentence_key("Tom Vale was born in 1985")],
    )
    assert '<span class="op">.</span>' in rp2.html
    assert render_profile(CITY).off_count() == 0


# --- the eval -----------------------------------------------------------


def test_the_eval_scores_caught_and_hidden_facts(tmp_path):
    from src.evals.note_relevance import by_entity, load_labels, score_entity, summarize

    rows = [
        {"entity": "Port Elm", "entity_type": "locations", "kind": "city",
         "heading": "Arrests", "sentence": s + "^[a2]", "gold": g, "confidence": c}
        for s, g, c in [(ABOUT, "about", "sure"), (AGE, "other", "sure"),
                        (LAW, "other", "sure"), ("Unsure one here.", "other", "unsure")]
    ]  # fmt: skip
    path = tmp_path / "note_labels.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    loaded, skipped = load_labels(str(path))
    assert len(loaded) == 3 and skipped == {"unsure": 1}

    def ask(msgs):  # hides the arrest (a real fact) and catches one of two
        return _reply(
            (1, "Police arrested Tom", "the arrest", "other"),
            (2, "Tom Vale was", "his age", "other"),
            (3, "Tom Vale later", "his studies", "about"),
        )

    records = [r for g in by_entity(loaded) for r in score_entity(g, ask)]
    summary = summarize(records)
    assert summary["caught"] == {"n": 2, "hit": 1, "share": 0.5}
    assert summary["hidden_about"] == {"n": 1, "hit": 1, "share": 1.0}
    assert summary["errors"] == 0


def test_tagging_stops_when_the_model_server_is_unreachable(monkeypatch):
    import httpx
    import pytest

    from src.engine.note_relevance import relevance_pass
    from src.utils import server_health

    monkeypatch.setenv("HINBOX_MAX_CONNECTION_FAILURES", "2")
    monkeypatch.setenv("HINBOX_NOTE_RELEVANCE", "on")
    server_health.reset()
    calls = []

    def ask(msgs):
        calls.append(1)
        server_health.record(httpx.ConnectError("no route"))
        raise RuntimeError("Connection error")

    entities = {"locations": {("Port Elm", "city"): _page()}}
    counts = run_note_relevance(entities, "ollama/test", local=True, ask=ask)
    assert counts["failed_calls"] == 2  # one call per heading was made
    assert len(calls) == 2
    entities["locations"][("Port Elm", "city")]["profile"].pop("note_relevance_json")
    calls.clear()
    monkeypatch.setattr("src.engine.note_relevance.make_asker", lambda m, local: ask)
    with pytest.raises(server_health.ModelServerUnreachable):
        relevance_pass(entities, "local")
    assert calls == []  # the breaker was already open: no call made
    server_health.reset()
