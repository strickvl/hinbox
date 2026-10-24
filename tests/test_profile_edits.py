"""Offline tests for applying profile edits (no model calls)."""

import src.engine.profile_edits as pe
from src.engine.profile_edits import ProfileEdit, ProfileEditList, apply_edits

TEXT = (
    "## Background\n"
    "Nashiri was captured in Dubai in 2002.^[a1] He was moved to Guantanamo in 2006.^[a2]\n"
    "\n"
    "## Legal Proceedings\n"
    "He was arraigned in 2011.^[a3]\n"
)


def _edit(op, anchor, text):
    return ProfileEdit(op=op, anchor=anchor, text=text)


def test_add_after_a_sentence_keeps_everything_else():
    new, report = apply_edits(
        TEXT,
        [
            _edit(
                "add",
                "Nashiri was captured in Dubai in 2002.^[a1]",
                "One report says 2003.^[a9]",
            )
        ],
    )
    assert "2002.^[a1] One report says 2003.^[a9] He was moved" in new
    assert new.replace(" One report says 2003.^[a9]", "") == TEXT
    assert report["applied"] == 1


def test_add_under_a_heading_goes_at_the_end_of_that_section():
    new, _ = apply_edits(TEXT, [_edit("add", "## Background", "He used aliases.^[a9]")])
    assert "in 2006.^[a2]\nHe used aliases.^[a9]\n\n## Legal Proceedings" in new
    new, _ = apply_edits(
        TEXT, [_edit("add", "## Legal Proceedings", "Trial set.^[a9]")]
    )
    assert new.endswith("arraigned in 2011.^[a3]\nTrial set.^[a9]\n")


def test_replace_puts_back_citations_the_model_dropped():
    new, _ = apply_edits(
        TEXT,
        [
            _edit(
                "replace",
                "He was arraigned in 2011.^[a3]",
                "He was arraigned in November 2011.^[a9]",
            )
        ],
    )
    assert "He was arraigned in November 2011.^[a9]^[a3]" in new
    assert "He was arraigned in 2011." not in new


def test_anchor_found_despite_reflowed_whitespace():
    new, report = apply_edits(
        TEXT,
        [_edit("add", "He was moved to  Guantanamo\nin 2006.^[a2]", "Fact.^[a9]")],
    )
    assert "in 2006.^[a2] Fact.^[a9]" in new and report["applied"] == 1


def test_unplaced_add_is_appended_and_unplaced_replace_is_skipped():
    new, report = apply_edits(
        TEXT,
        [
            _edit("add", "A sentence that isn't there.", "Kept anyway.^[a9]"),
            _edit("replace", "Also not there.", "Dropped.^[a9]"),
            _edit("add", "END", "## Associates\nKhallad.^[a9]"),
        ],
    )
    assert new.startswith(TEXT.rstrip())
    assert "Kept anyway.^[a9]" in new and "Dropped" not in new
    assert new.rstrip().endswith("## Associates\nKhallad.^[a9]")
    assert report["appended"] == 1 and report["applied"] == 1
    assert report["skipped"][0]["why"] == "not found"


def test_no_edits_leaves_the_text_alone():
    assert apply_edits(TEXT, [])[0] == TEXT


def test_edit_profile_adds_the_article_to_sources(monkeypatch):
    seen = {}

    def fake_generate(messages, response_model, model, max_tokens):
        seen["model"], seen["system"] = model, messages[0]["content"]
        return ProfileEditList(edits=[_edit("add", "END", "New.^[a9]")])

    monkeypatch.setattr(pe, "cloud_generation", fake_generate)
    profile = {"text": TEXT, "sources": ["a1", "a2", "a3"], "tags": ["x"]}
    out = pe.edit_profile(
        "person", "Nashiri", profile, "article", "a9", model="openai/x"
    )
    assert out["sources"] == ["a1", "a2", "a3", "a9"] and out["tags"] == ["x"]
    assert out["text"].rstrip().endswith("New.^[a9]")
    assert seen["model"] == "openai/x" and "^[a9]" in seen["system"]


def test_adds_after_the_same_sentence_keep_their_order():
    anchor = "He was moved to Guantanamo in 2006.^[a2]"
    new, _ = apply_edits(
        TEXT,
        [
            _edit("add", anchor, "Hearings were suspended in 2012.^[a9]"),
            _edit("add", anchor, "They resumed in 2013.^[a9]"),
        ],
    )
    assert (
        "2006.^[a2] Hearings were suspended in 2012.^[a9] They resumed in 2013.^[a9]\n"
        in new
    )


def test_edit_profile_keeps_the_disagreements_it_found(monkeypatch):
    def fake_generate(messages, response_model, model, max_tokens):
        return ProfileEditList(
            disagreements=[
                pe.Disagreement(
                    profile_sentence="He was moved to Guantanamo in 2006.^[a2]",
                    article_says="moved in 2007",
                )
            ],
            edits=[
                _edit(
                    "add",
                    "He was moved to Guantanamo in 2006.^[a2]",
                    "Another report says 2007.^[a9]",
                )
            ],
        )

    monkeypatch.setattr(pe, "cloud_generation", fake_generate)
    out = pe.edit_profile(
        "person", "N", {"text": TEXT, "sources": []}, "x", "a9", model="openai/x"
    )
    assert out["disagreements"][0]["article_says"] == "moved in 2007"
    assert "2006.^[a2] Another report says 2007.^[a9]" in out["text"]
