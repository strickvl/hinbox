"""Offline tests for applying profile edits (no model calls)."""

import pytest
from pydantic import ValidationError

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


def test_an_anchor_copied_without_its_citation_keeps_the_citation():
    # In a local replay Qwen quoted sentences without their "^[id]", the add
    # went in between, and the old sentence lost its citation
    anchors = ["Nashiri was captured in Dubai in 2002", "He was arraigned in 2011."]
    new, _ = apply_edits(
        TEXT,
        [
            _edit("add", anchors[0], "One report says 2003.^[a9]"),
            _edit("add", anchors[0], "Another says 2001.^[a8]"),
            _edit("add", anchors[1], "He pleaded not guilty.^[a7]"),
        ],
    )
    assert "2002.^[a1] One report says 2003.^[a9] Another says 2001.^[a8] He was" in new
    assert "in 2011.^[a3] He pleaded not guilty.^[a7]\n" in new


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


def test_edit_list_accepts_lists_sent_as_json_strings():
    parsed = ProfileEditList.model_validate(
        {
            "disagreements": "[]",
            "new_facts": '["He was arraigned in 2011."]',
            "edits": '[{"op": "add", "anchor": "END", "text": "Arraigned.^[a9]"}]',
        }
    )
    assert parsed.new_facts == ["He was arraigned in 2011."]
    assert parsed.edits[0].text == "Arraigned.^[a9]"


def test_edit_list_joins_one_array_per_line():
    # Bolt's shape on 3 of 4 crashed test cases: valid arrays, one per line
    parsed = ProfileEditList.model_validate(
        {"new_facts": '["Fact one."]\n["Fact two."]\n', "edits": "[]"}
    )
    assert parsed.new_facts == ["Fact one.", "Fact two."]


def test_edit_list_repairs_swapped_closing_quote_and_brace():
    # Bolt closed the last edit with `^[a9]}"]` instead of `^[a9]"}]`
    edits = (
        '[{"op": "add", "anchor": "END", "text": "One.^[a9]"}, '
        '{"op": "add", "anchor": "END", "text": "Two.^[a9]}"]'
    )
    parsed = ProfileEditList.model_validate({"edits": edits})
    assert [e.text for e in parsed.edits] == ["One.^[a9]", "Two.^[a9]"]


def test_edit_list_repairs_dropped_closing_quote():
    # Bolt ended the last edit with `^[a9]}]`: the text's closing quote is gone
    edits = (
        '[{"op": "add", "anchor": "END", "text": "One.^[a9]"}, '
        '{"op": "add", "anchor": "END", "text": "Two.^[a9]}]'
    )
    parsed = ProfileEditList.model_validate({"edits": edits})
    assert [e.text for e in parsed.edits] == ["One.^[a9]", "Two.^[a9]"]


def test_edit_list_repairs_missing_final_bracket():
    # Bolt stopped at `^[a9]"}` without closing the array
    edits = '[{"op": "add", "anchor": "END", "text": "Two.^[a9]"}'
    parsed = ProfileEditList.model_validate({"edits": edits})
    assert [e.text for e in parsed.edits] == ["Two.^[a9]"]


def test_unreadable_new_facts_dont_sink_good_edits():
    parsed = ProfileEditList.model_validate(
        {"new_facts": "He was arraigned", "edits": []}
    )
    assert parsed.new_facts == ["He was arraigned"]


def test_unreadable_edits_still_fail():
    with pytest.raises(ValidationError):
        ProfileEditList.model_validate({"edits": '[{"op": "add", "anchor"'})


def test_local_models_get_the_local_prompt(monkeypatch):
    seen = {}

    def fake_generate(messages, response_model, model, max_tokens):
        seen["system"] = messages[0]["content"]
        return ProfileEditList()

    monkeypatch.setattr(pe, "local_generation", fake_generate)
    monkeypatch.setattr(pe, "cloud_generation", fake_generate)
    pe.edit_profile("person", "N", {"text": TEXT}, "x", "a9", model="ollama/q")
    assert "(local models)" in seen["system"] and "{" not in seen["system"]
    pe.edit_profile("person", "N", {"text": TEXT}, "x", "a9", model="openai/x")
    assert "(local models)" not in seen["system"]


# --- the date check after the edits (edit_date_check.py) ---------------------

MOVED = "Nashiri was moved to Guantánamo in September 2006 by order of Bush.^[a1]"
MOVED_ARTICLE = (
    "The hearing went on. Bush ordered Nashiri moved to Guantánamo in "
    "September 2005 for a military trial."
)


def _date_model(edit_list, verdict="conflict", fail=False):
    from src.engine.date_disputes import DateVerdicts

    calls = []

    def fake_generate(messages, response_model, model, max_tokens):
        calls.append(response_model.__name__)
        if response_model is DateVerdicts:
            if fail:
                raise RuntimeError("server down")
            return DateVerdicts.model_validate(
                {"verdicts": [{"pair": 1, "event": "his transfer", "verdict": verdict,
                               "sentence": "Nashiri was moved in September 2005."}]}
            )  # fmt: skip
        return edit_list

    return fake_generate, calls


def test_the_date_check_adds_a_conflict_the_model_missed(monkeypatch):
    monkeypatch.setenv("HINBOX_EDIT_DATE_CHECK", "on")
    fake, calls = _date_model(ProfileEditList(edits=[]))
    monkeypatch.setattr(pe, "cloud_generation", fake)
    out = pe.edit_profile(
        "person", "Nashiri", {"text": MOVED}, MOVED_ARTICLE, "a9", model="openai/x"
    )
    assert out["text"] == (
        MOVED + " Another report differs: Nashiri was moved in September 2005.^[a9]"
    )
    assert out["edit_report"]["date_check"]["date_verdicts"] == ["conflict"]
    assert calls == ["ProfileEditList", "DateVerdicts"]


def test_the_date_check_is_on_for_cloud_models_only_by_default(monkeypatch):
    from src.engine.edit_date_check import edit_date_check_enabled

    monkeypatch.delenv("HINBOX_EDIT_DATE_CHECK", raising=False)
    assert edit_date_check_enabled(local=False)
    assert not edit_date_check_enabled(local=True)  # Qwen not measured yet
    monkeypatch.setenv("HINBOX_EDIT_DATE_CHECK", "on")
    assert edit_date_check_enabled(local=True)
    monkeypatch.setenv("HINBOX_EDIT_DATE_CHECK", "off")
    assert not edit_date_check_enabled(local=False)


def test_the_date_check_can_be_turned_off(monkeypatch):
    monkeypatch.setenv("HINBOX_EDIT_DATE_CHECK", "off")
    fake, calls = _date_model(ProfileEditList(edits=[]))
    monkeypatch.setattr(pe, "cloud_generation", fake)
    out = pe.edit_profile(
        "person", "Nashiri", {"text": MOVED}, MOVED_ARTICLE, "a9", model="openai/x"
    )
    assert out["text"] == MOVED and calls == ["ProfileEditList"]
    assert "date_check" not in out["edit_report"]


def test_the_date_check_leaves_a_disagreement_the_model_listed(monkeypatch):
    monkeypatch.setenv("HINBOX_EDIT_DATE_CHECK", "on")
    monkeypatch.setenv("HINBOX_DISPUTE_CHECK", "off")  # its own call, not this one
    listed = ProfileEditList(
        disagreements=[{"profile_sentence": MOVED, "article_says": "September 2005"}],
        edits=[_edit("add", MOVED, "A later report gives 2005 instead.^[a9]")],
    )
    fake, calls = _date_model(listed)
    monkeypatch.setattr(pe, "cloud_generation", fake)
    out = pe.edit_profile(
        "person", "Nashiri", {"text": MOVED}, MOVED_ARTICLE, "a9", model="openai/x"
    )
    assert "Another report differs" not in out["text"]
    assert out["edit_report"]["date_check"]["date_pairs"] == 0
    assert "DateVerdicts" not in calls


def test_a_sentence_disputed_before_is_not_disputed_again(monkeypatch):
    monkeypatch.setenv("HINBOX_EDIT_DATE_CHECK", "on")
    fake, calls = _date_model(ProfileEditList(edits=[]))
    monkeypatch.setattr(pe, "cloud_generation", fake)
    first = pe.edit_profile(
        "person", "Nashiri", {"text": MOVED}, MOVED_ARTICLE, "a9", model="openai/x"
    )
    second = pe.edit_profile(
        "person", "Nashiri", first, MOVED_ARTICLE, "a10", model="openai/x"
    )
    assert second["text"].count("Another report differs") == 1
    assert second["edit_report"]["date_check"]["date_pairs"] == 0
    assert calls.count("DateVerdicts") == 1  # only the first article was judged


def test_only_a_real_dispute_marks_a_sentence_as_disputed(monkeypatch):
    from src.engine.edit_date_check import _is_dispute

    assert _is_dispute("Another report differs: Nashiri was moved in 2005.^[a9]")
    assert _is_dispute("A later report gives the year as 2005 instead.^[a9]")
    assert _is_dispute("The sources disagree on the year.^[a9]")
    assert not _is_dispute("His lawyers instead sought a civilian trial.^[a2]")
    assert not _is_dispute("The two men disagree about the plan.^[a2]")
    # ...so the sentence before an unrelated "instead" is still checked
    monkeypatch.setenv("HINBOX_EDIT_DATE_CHECK", "on")
    fake, calls = _date_model(ProfileEditList(edits=[]))
    monkeypatch.setattr(pe, "cloud_generation", fake)
    profile = {"text": MOVED + " His lawyers instead sought a civilian trial.^[a2]"}
    out = pe.edit_profile(
        "person", "Nashiri", profile, MOVED_ARTICLE, "a9", model="openai/x"
    )
    assert "Another report differs" in out["text"] and "DateVerdicts" in calls


def test_a_failed_date_check_fails_the_update(monkeypatch):
    """So the pipeline leaves the article unlinked and tries it again."""
    monkeypatch.setenv("HINBOX_EDIT_DATE_CHECK", "on")
    fake, _ = _date_model(
        ProfileEditList(edits=[_edit("add", "END", "He was tried.^[a9]")]), fail=True
    )
    monkeypatch.setattr(pe, "cloud_generation", fake)
    with pytest.raises(RuntimeError, match="server down"):
        pe.edit_profile(
            "person", "Nashiri", {"text": MOVED}, MOVED_ARTICLE, "a9", model="openai/x"
        )


def test_a_failed_date_check_keeps_the_old_profile_and_article_unlinked(monkeypatch):
    from src.engine.profiles import VersionedProfile, update_profile

    monkeypatch.setenv("HINBOX_EDIT_DATE_CHECK", "on")
    fake, _ = _date_model(ProfileEditList(edits=[]), fail=True)
    monkeypatch.setattr(pe, "cloud_generation", fake)
    old = {"text": MOVED, "sources": ["a1"]}
    new, _, _ = update_profile(
        "people", "Nashiri", old, VersionedProfile(), MOVED_ARTICLE, "a9",
        model="openai/x", approach="edit",
    )  # fmt: skip
    assert new["text"] == MOVED and "a9" not in new.get("sources", [])
