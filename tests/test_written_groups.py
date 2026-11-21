"""A section's notes grouped and sorted by code, missed notes written back
in their place, and the people checks' institution and captor rules. Made-up
notes only (the repository is public)."""

from src.engine.written_checks import missing_details
from src.engine.written_groups import event_when, group_notes, groups_block, place_after
from src.engine.written_page import Written
from src.engine.written_plan import _surnames
from tests.test_written_conflicts import _note


def test_notes_are_sorted_by_the_date_they_name_and_grouped():
    notes = [
        _note(1, "In May 2009, Tom Vale said he was innocent.", "2012-01-01"),
        _note(2, "Police arrested Tom Vale in March 2004.", "2010-01-01"),
        _note(3, "Ann Roe was released.", "2006-07-01"),
        _note(4, "Ann Roe asked for her papers.", "2006-09-01"),
        _note(5, "Lee Moss moved away.", "2008-01-01"),
    ]
    assert event_when(notes[0]) == (2009, 5) and event_when(notes[2]) == (2006, 7)
    groups = group_notes(notes, "Port Elm")
    assert [[n.n for n in g] for g in groups] == [[2], [3, 4], [5], [1]]
    text = groups_block(groups, lambda g: " ".join(f"N{n.n}" for n in g))
    assert text.startswith("Group 1:\nN2") and "Group 2:\nN3 N4" in text


def test_a_group_holds_at_most_three_notes():
    notes = [_note(i, f"Tom Vale spoke, note {i}.", "2005-05-06") for i in range(1, 6)]
    assert [len(g) for g in group_notes(notes, "Port Elm")] == [3, 2]


def test_a_filled_sentence_goes_after_the_sentence_telling_the_note_before():
    out = [
        Written("Opening.", [9]),
        Written("In 2004 police came.", [2], "Arrests", 1),
        Written("In 2009 they left.", [4], "Arrests", 2),
        Written("Harbour.", [1], "Harbour", 3),
    ]
    order = {2: 0, 3: 1, 4: 2}
    assert place_after(out, "Arrests", [3], order) == 1  # between 2004 and 2009
    assert place_after(out, "Arrests", [7], {7: 0, 2: 1}) == 0  # before the first
    assert place_after(out, "Arrests", [8], order) == 2  # unknown: the end


def test_institutions_and_hyphenated_spellings_are_not_unnamed_people():
    text = "The Periodic Review Board and the Associated Press named Saad al-Dhuby."
    assert _surnames(text, "Port Elm") == {"dhuby"}
    assert _surnames("Saad al Dhuby left.", "Port Elm") == {"dhuby"}


def test_the_captor_rule_accepts_him_on_the_captives_own_page():
    notes = ["Northern Alliance forces captured him in 2001."]
    sentence = "Northern Alliance forces captured Tom Vale in 2001."
    assert missing_details(sentence, notes, "Tom Vale") == []
    # On another page the note must name the captive
    assert missing_details(sentence, notes, "Port Elm")
