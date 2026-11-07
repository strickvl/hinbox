"""Tests for the shortened copy of long profiles sent to the model."""

from typing import List

from src.engine import profile_edits
from src.engine.profile_edits import ProfileEdit, apply_edits
from src.engine.profile_view import (
    embedding_text,
    profile_view,
    trimmed_note,
)


def _hub_profile(sections: int = 12, paras: int = 4) -> str:
    """A long profile: an opening paragraph, then sections of cited paragraphs."""
    lines: List[str] = [
        "The United States holds detainees at Guantánamo Bay, Cuba.^[a0]",
        "",
    ]
    for s in range(sections):
        lines += [f"## Topic {s}", ""]
        for p in range(paras):
            lines += [
                f"In {1990 + s} the government discussed matter{s}x{p} at length, "
                f"with officials reviewing filler text about routine business.^[a{s}{p}]",
                "",
            ]
    lines += [
        "## Hunger Strike",
        "",
        "In 2013 Pentagon officials counted 106 hunger strikers at Camp Delta.^[hs1]",
        "",
    ]
    return "\n".join(lines)


ARTICLE = (
    "The hunger strike at Camp Delta grew in 2013, the Pentagon said, as "
    "lawyers described force-feeding of strikers."
)


def test_short_profile_is_unchanged():
    text = "Short profile.^[a1]\n\n## Background\n\nMore.^[a2]"
    assert profile_view(text, ARTICLE, 16000) == text
    assert trimmed_note(text, text) == ""


def test_zero_budget_turns_it_off():
    text = _hub_profile()
    assert profile_view(text, ARTICLE, 0) == text


def test_long_profile_keeps_opening_headings_and_related_passage():
    text = _hub_profile()
    budget = 1500
    view = profile_view(text, ARTICLE, budget)
    assert len(view) < len(text)
    assert len(view) <= budget
    assert view.startswith("The United States holds detainees")
    for s in range(12):
        assert f"## Topic {s}" in view
    # the passage about the article's subject is picked over filler
    assert "106 hunger strikers at Camp Delta" in view
    assert "not shown" in view
    assert trimmed_note(text, view).startswith("(This profile is long")


def test_view_keeps_original_order():
    text = _hub_profile()
    view = profile_view(text, ARTICLE, 1500)
    positions = [
        text.find(line) for line in view.split("\n\n") if "not shown" not in line
    ]
    assert positions == sorted(positions)
    assert -1 not in positions


def test_long_paragraph_is_cut_into_sentence_runs():
    sentences = [
        f"Sentence {i} talks about ordinary things here.^[b{i}]" for i in range(80)
    ]
    sentences[60] = "The Camp Delta hunger strike reached 106 strikers in 2013.^[hit]"
    text = "Opening line.^[o]\n\n## Body\n\n" + " ".join(sentences)
    view = profile_view(text, ARTICLE, 1200)
    assert "reached 106 strikers in 2013.^[hit]" in view
    assert len(view) <= 1200


def test_edit_anchored_on_shown_sentence_applies_to_full_text():
    text = _hub_profile()
    view = profile_view(text, ARTICLE, 1500)
    anchor = (
        "In 2013 Pentagon officials counted 106 hunger strikers at Camp Delta.^[hs1]"
    )
    assert anchor in view
    new_text, report = apply_edits(
        text,
        [
            ProfileEdit(
                op="add", anchor=anchor, text="Lawyers described force-feeding.^[n1]"
            )
        ],
    )
    assert report["applied"] == 1
    # every passage of the full profile is still there, view or not
    for line in text.splitlines():
        assert line in new_text
    assert "Lawyers described force-feeding.^[n1]" in new_text


def test_heading_edit_goes_to_end_of_full_section():
    text = _hub_profile()
    new_text, _ = apply_edits(
        text, [ProfileEdit(op="add", anchor="## Topic 3", text="Added.^[n2]")]
    )
    section = new_text.split("## Topic 3")[1].split("## Topic 4")[0]
    assert section.rstrip().endswith("Added.^[n2]")
    assert "matter3x3" in section


def test_edit_prompt_shows_view_and_note(monkeypatch):
    monkeypatch.setattr(profile_edits, "PROFILE_VIEW_CHARS", 1500)
    text = _hub_profile()
    messages = profile_edits._messages(
        "organizations", "United States", text, ARTICLE, "new1", "guantanamo"
    )
    user = messages[1]["content"]
    assert "(This profile is long" in user
    assert "106 hunger strikers" in user
    assert len(user) < len(text)


def test_edit_prompt_unchanged_when_off(monkeypatch):
    monkeypatch.setattr(profile_edits, "PROFILE_VIEW_CHARS", 0)
    text = _hub_profile()
    user = profile_edits._messages(
        "organizations", "United States", text, ARTICLE, "new1", "guantanamo"
    )[1]["content"]
    assert text in user
    assert "This profile is long" not in user


def test_embedding_text_keeps_opening():
    text = _hub_profile(sections=80)
    out = embedding_text(text)
    assert len(out) < len(text)
    assert out.startswith("The United States holds detainees")


def test_match_check_sends_a_cut_existing_profile(monkeypatch):
    from unittest.mock import patch

    from src.engine import match_checker

    monkeypatch.setattr(match_checker, "MATCH_PROFILE_CHARS", 1500)
    monkeypatch.setattr(match_checker, "_MATCH_MEMO_ENABLED", False)
    text = _hub_profile()
    sent = {}

    def fake_generation(messages, **kwargs):
        sent["user"] = messages[-1]["content"]
        return "[[ ## reasoning ## ]]\nSame.\n\n[[ ## is_match ## ]]\nTrue\n\n[[ ## confidence ## ]]\n0.9"

    with patch.object(match_checker, "text_generation", fake_generation):
        result = match_checker.local_model_check_match(
            "U.S.", "United States", ARTICLE, text, entity_type="locations"
        )
    assert result.is_match is True
    assert "106 hunger strikers" in sent["user"]
    assert "not shown" in sent["user"]
    assert len(sent["user"]) < len(text)


def test_profile_starting_with_a_heading_keeps_its_first_paragraph():
    text = "### Overview\nThe United States runs the prison.^[a0]\n\n" + _hub_profile()
    view = profile_view(text, ARTICLE, 1500)
    assert view.startswith("### Overview\n\nThe United States runs the prison.^[a0]")


def test_profile_that_is_one_long_paragraph_is_cut_without_error():
    text = " ".join(f"Sentence {i} says routine things.^[c{i}]" for i in range(900))
    assert len(text) > 16000
    view = profile_view(text, ARTICLE, 16000)
    assert view.startswith("Sentence 0 says routine things.^[c0]")
    assert len(view) <= 16000
