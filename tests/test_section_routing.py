"""Routing new facts to the starter headings of a reorganised profile
(src/engine/section_routing.py, profile_edits.apply_edits).

Scenario: al Nashiri's profile has Background and "Military commission or
legal case" but no "Detention at Guantánamo" section. An article about his
cell conditions should start that section, in its place in the list, not go
to the end of the profile or into the legal section.
"""

from src.engine import profile_edits
from src.engine.profile_edits import ProfileEdit, apply_edits
from src.engine.section_routing import (
    insert_missing_heading,
    routing_note,
    routing_sections,
)

STARTER = [
    "Background",
    "Capture and CIA custody",
    "Detention at Guantánamo",
    "Military commission or legal case",
    "Health and treatment",
]
TEXT = (
    "Abd al Rahim al Nashiri is a Saudi detainee.^[a0]\n\n"
    "### Background\n\nHe was born in Mecca.^[a0]\n\n"
    "### Military commission or legal case\n\n"
    "#### Capital charges\n\nHe faces the death penalty.^[a1]\n\n"
    "### Legal mail\n\nGuards read his mail.^[a2]\n"
)
NEW = "He was moved to Camp 7 in 2006.^[a9]"


def _add(anchor: str) -> ProfileEdit:
    return ProfileEdit(op="add", anchor=anchor, text=NEW)


def test_an_add_on_a_missing_starter_heading_creates_it_in_list_order():
    out, report = apply_edits(TEXT, [_add("## Detention at Guantánamo")], STARTER)
    assert report["applied"] == 1 and report["new_headings"] == [
        "Detention at Guantánamo"
    ]
    detention = out.index("### Detention at Guantánamo")
    assert (
        out.index("### Background") < detention < out.index("### Military commission")
    )
    assert out.index(NEW) > detention
    assert out.replace(f"### Detention at Guantánamo\n\n{NEW}\n\n", "") == TEXT


def test_a_heading_after_every_starter_section_goes_before_added_headings():
    out, _ = apply_edits(TEXT, [_add("### Health and treatment")], STARTER)
    assert out.index("#### Capital charges") < out.index("### Health and treatment")
    assert out.index("### Health and treatment") < out.index("### Legal mail")


def test_a_bare_or_annotated_starter_name_counts_as_a_heading_anchor():
    for anchor in (
        "Detention at Guantánamo",
        "Detention at Guantánamo: camps and cells",
    ):
        out, report = apply_edits(TEXT, [_add(anchor)], STARTER)
        assert "### Detention at Guantánamo" in out and report["appended"] == 0


def test_an_add_on_a_parent_heading_stays_out_of_its_sub_sections():
    parent = _add("### Military commission or legal case")
    out, report = apply_edits(TEXT, [parent], STARTER)
    assert "new_headings" not in report
    assert out.index("### Military commission") < out.index(NEW)
    assert out.index(NEW) < out.index("#### Capital charges")
    # Without starter headings it still goes after the sub-sections
    flat, _ = apply_edits(TEXT, [parent])
    assert flat.index("He faces the death penalty") < flat.index(NEW)


def test_an_add_on_a_sub_heading_goes_at_its_end():
    out, _ = apply_edits(TEXT, [_add("#### Capital charges")], STARTER)
    assert out.index("He faces the death penalty") < out.index(NEW)
    assert out.index(NEW) < out.index("### Legal mail")


def test_without_starter_headings_nothing_changes():
    out, report = apply_edits(TEXT, [_add("## Detention at Guantánamo")])
    assert "### Detention at Guantánamo" not in out and report["appended"] == 1
    assert insert_missing_heading(TEXT, "Background", NEW, STARTER) is None
    assert insert_missing_heading(TEXT, "Prison life", NEW, STARTER) is None


def test_routing_can_be_turned_off(monkeypatch):
    monkeypatch.setenv("HINBOX_SECTION_ROUTING", "off")
    done = {"text": TEXT, "reorganised": "done"}
    assert routing_sections(done, "people", "detainee", "guantanamo") is None


def test_routing_is_only_for_reorganised_profiles():
    assert routing_sections({"text": TEXT}, "people", "detainee", "guantanamo") is None
    starter, notes = routing_sections(
        {"text": TEXT, "reorganised": "done"}, "people", "detainee", "guantanamo"
    )
    assert "Detention at Guantánamo" in starter and notes
    assert (
        routing_sections(
            {"text": TEXT, "reorganised": "refused"}, "people", "detainee", "guantanamo"
        )
        is None
    )


def test_routing_note_marks_headings_the_profile_lacks():
    note = routing_note(STARTER, {}, "al Nashiri", TEXT)
    assert "- Background\n" in note
    assert "- Detention at Guantánamo (not in the profile yet)" in note
    assert "legal heading even when it is about detention" in note


def test_routing_note_lists_headings_with_what_they_are_for():
    note = routing_note(
        STARTER[:2], {"Background": "life before capture"}, "al Nashiri", TEXT
    )
    assert "- Background: life before capture\n- Capture and CIA custody" in note
    assert "al Nashiri" in note


def test_edit_profile_sends_the_note_and_routes_adds(monkeypatch):
    seen = {}

    def fake_generate(messages, response_model, model, max_tokens):
        seen["system"] = messages[0]["content"]
        return response_model(edits=[_add("### Detention at Guantánamo")])

    monkeypatch.setattr(profile_edits, "cloud_generation", fake_generate)
    profile = {"text": TEXT, "sources": ["a0"], "reorganised": "done"}
    out = profile_edits.edit_profile(
        "people", "al Nashiri", profile, "article", "a9", model="m", local=False,
        guard=False, kind="detainee",
    )  # fmt: skip
    assert "## Where new text goes" in seen["system"]
    assert "### Detention at Guantánamo" in out["text"]
    assert out["edit_report"]["routed"] is True
    flat = profile_edits.edit_profile(
        "people", "al Nashiri", {"text": TEXT, "sources": ["a0"]}, "article", "a9",
        model="m", local=False, guard=False, kind="detainee",
    )  # fmt: skip
    assert "## Where new text goes" not in seen["system"]
    assert flat["edit_report"]["routed"] is False


def test_two_bare_adds_on_a_missing_heading_both_go_under_it():
    second = ProfileEdit(
        op="add", anchor="Detention at Guantánamo", text="He fasted.^[a8]"
    )
    out, report = apply_edits(TEXT, [_add("Detention at Guantánamo"), second], STARTER)
    assert report["applied"] == 2 and report["new_headings"] == [
        "Detention at Guantánamo"
    ]
    assert "\n### Detention at Guantánamo\n" in out
    assert out.index("### Detention at Guantánamo") < out.index("He fasted.")
    assert out.index("He fasted.") < out.index("### Military commission")


def test_an_add_on_a_heading_copied_with_its_mark_creates_it():
    anchor = "### Detention at Guantánamo (not in the profile yet)"
    out, report = apply_edits(TEXT, [_add(anchor)], STARTER)
    assert report["new_headings"] == ["Detention at Guantánamo"]
    assert "(not in the profile yet)" not in out
    assert out.index("### Detention at Guantánamo") < out.index(
        "### Military commission"
    )


def test_an_end_add_opening_with_an_existing_heading_goes_into_that_section():
    # Local replay: Qwen added on END "## Military commission or legal case\n\n
    # ### Hearing procedures ..." and KSM got a second legal section at the end
    add = ProfileEdit(op="add", anchor="END", text=(
        "## Military commission or legal case\n\n### Video testimony\n\n" + NEW
    ))  # fmt: skip
    out, report = apply_edits(TEXT, [add], STARTER)
    assert out.count("Military commission or legal case") == 1
    # A sub-heading of its own is dropped: code, not the model, makes them
    assert "Video testimony" not in out and report["appended"] == 0
    assert out.index("### Military commission") < out.index(NEW)
    assert out.index(NEW) < out.index("#### Capital charges")
    # Without starter headings it is kept, at the level under the section
    flat, _ = apply_edits(TEXT, [add])
    assert "\n#### Video testimony\n" in flat


def test_an_end_add_opening_with_a_missing_starter_heading_creates_it():
    add = ProfileEdit(op="add", anchor="END", text="## Detention at Guantánamo\n" + NEW)
    out, report = apply_edits(TEXT, [add], STARTER)
    assert report["new_headings"] == ["Detention at Guantánamo"]
    assert (
        "\n### Detention at Guantánamo\n" in out
        and "## Detention" not in out.replace("### Detention", "")
    )
    assert out.index("### Background") < out.index("### Detention at Guantánamo")
    assert out.index("### Detention at Guantánamo") < out.index(
        "### Military commission"
    )


def test_a_bare_sub_heading_anchor_is_not_glued_onto_the_heading_line():
    # Luna replay: anchor "Nashiri Proceedings" (no "#") gave the heading
    # "#### Nashiri Proceedings Pohl assigned himself ..."
    for starter in (STARTER, None):
        out, _ = apply_edits(TEXT, [_add("Capital charges")], starter)
        assert "\n#### Capital charges\n" in out
        assert out.index("He faces the death penalty") < out.index(NEW)
        assert out.index(NEW) < out.index("### Legal mail")


def test_an_end_add_opening_with_an_unknown_heading():
    add = ProfileEdit(op="add", anchor="END", text="## Legal Proceedings\n" + NEW)
    out, report = apply_edits(TEXT, [add], STARTER)
    assert "Legal Proceedings" not in out and NEW in out
    assert report["dropped_headings"] == ["## Legal Proceedings"]
    flat, _ = apply_edits(TEXT, [add])
    assert "\n### Legal Proceedings\n" in flat and "\n## Legal" not in flat
