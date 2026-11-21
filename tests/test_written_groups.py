"""A section's notes grouped and sorted by code, missed notes written back
in their place, and the people checks' institution and captor rules. Made-up
notes only (the repository is public)."""

from src.engine.written_checks import drop_repeats, missing_details
from src.engine.written_groups import (
    event_when,
    group_notes,
    groups_block,
    place_after,
    sentence_when,
    text_when,
)
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
    assert event_when(notes[0]) == (2009, 5, 0) and event_when(notes[2]) == (2006, 7, 0)
    groups = group_notes(notes, "Port Elm")
    assert [[n.n for n in g] for g in groups] == [[2], [3, 4], [5], [1]]
    text = groups_block(groups, lambda g: " ".join(f"N{n.n}" for n in g))
    assert text.startswith("Group 1:\nN2") and "Group 2:\nN3 N4" in text


def test_a_group_holds_at_most_three_notes():
    notes = [_note(i, f"Tom Vale spoke, note {i}.", "2005-05-06") for i in range(1, 6)]
    assert [len(g) for g in group_notes(notes, "Port Elm")] == [3, 2]


def test_full_dates_sort_by_month_and_day():
    assert text_when("captured on Sept. 11, 2002, in a raid") == (2002, 9, 11)
    assert text_when("on September 10-11, 2002") == (2002, 9, 10)
    assert text_when("in April 2003") == (2003, 4, 0) and text_when("no date") is None
    notes = [
        _note(1, "Police captured Tom Vale on September 11, 2002.", "2010-01-01"),
        _note(2, "Police arrested Ann Roe in February 2002.", "2010-01-01"),
        _note(3, "Lee Moss was caught on April 29, 2003.", "2010-01-01"),
        _note(4, "Bo Day was caught on March 5, 2003.", "2010-01-01"),
    ]
    flat = [n.n for g in group_notes(notes, "Port Elm") for n in g]
    assert flat == [2, 1, 4, 3]


def test_a_filled_sentence_goes_by_its_own_date_else_after_the_note_before():
    notes = {
        n.n: n
        for n in [
            _note(2, "Police came.", "2004-03-01"),
            _note(3, "Tom Vale spoke.", "2006-03-01"),
            _note(4, "They left.", "2009-03-01"),
        ]
    }
    when = sentence_when(notes)
    out = [
        Written("Opening.", [9]),
        Written("In 2004 police came.", [2], "Arrests", 1),
        Written("They left in 2009.", [4], "Arrests", 2),
        Written("Harbour.", [1], "Harbour", 3),
    ]
    order = {2: 0, 3: 1, 4: 2}
    at = lambda text, ns, o=order: place_after(  # noqa: E731
        out, "Arrests", Written(text, ns, "Arrests"), o, when
    )
    assert at("Tom Vale spoke.", [3]) == 1  # between 2004 and 2009, by note order
    assert at("In June 2017, Tom Vale spoke.", [3]) == 2  # by its date, last
    assert at("In 2001, Tom Vale spoke.", [4]) == 0  # before the first
    assert at("Tom Vale spoke.", [8]) == 2  # notes not in the order: the end


def test_one_fact_told_twice_in_other_words_is_dropped():
    out = [
        Written("In January 2016, Tom Vale was moved to Elmland at the direct request of its government; officials identified Tom Vale as 36 years old.", [1], "Moves", 1),
        Written("The Elmland government directly requested the move of Tom Vale in January 2016; officials identified him as 36 years old.", [1, 2], "Moves", 1),
    ]  # fmt: skip
    drop_repeats(out)
    assert out[1].dropped and not out[0].dropped


def test_a_number_inside_another_number_is_not_in_the_notes():
    notes = [
        "The bomb killed 10 people and wounded three on Aug. 5, 2003.",
        "2014-02-02",
    ]
    assert "14" in missing_details("The bomb killed 14 people.", notes, "Port Elm")
    assert (
        missing_details("The bomb killed 10 people in 2003.", notes, "Port Elm") == []
    )


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


def test_the_page_review_drops_a_bare_repeat_and_tells_a_contradiction():
    from src.engine.written_checks import SentenceChecks
    from src.engine.written_review import Review, review_page
    from src.engine.written_revise import Fill

    notes = [
        _note(1, "Port Elm is a harbour city.", "2003-01-02"),
        _note(2, "Police held Tom Vale 183 days in Port Elm.", "2004-01-01"),
        _note(
            3, "A 2014 report says Tom Vale was not beaten in Port Elm.", "2014-12-01"
        ),
        _note(
            4, "An earlier account says guards beat Tom Vale in Port Elm.", "2009-01-01"
        ),
    ]
    out = [
        Written("Port Elm is a harbour city.", [1]),
        Written("Police held Tom Vale for 183 days.", [2], "Arrests", 1),
        Written("A 2014 report says Tom Vale was not beaten.", [3], "Arrests", 1),
        Written("Guards beat Tom Vale, an earlier account says.", [4], "Prison", 2),
        Written("Tom Vale was held for 183 days.", [2], "Prison", 2),
        Written("Tom Vale was held for 183 days by Ann Roe.", [2], "Prison", 2),
    ]
    both = "A 2014 report says Tom Vale was not beaten; an earlier account says guards beat him."

    def ask(msgs, model, max_tokens):
        if model is Review:
            return Review.model_validate({
                "repeats": [{"sentence": 5, "repeats": 2}, {"sentence": 6, "repeats": 2}],
                "contradictions": [{"first": 4, "second": 3, "about": "beating"}]})  # fmt: skip
        if model is Fill:
            return Fill.model_validate({"sentences": [{"text": both, "notes": [3, 4]}]})
        return SentenceChecks.model_validate({"checks": [
            {"sentence": 1, "starts_with": "A 2014 report", "problem": "none",
             "verdict": "supported"}]})  # fmt: skip

    report = {"calls": 0}
    res = review_page(out, notes, "Port Elm", "place", ask, report)
    kept = [w.text for w in res if not w.dropped]
    assert "Tom Vale was held for 183 days." not in kept  # a bare repeat
    assert "Tom Vale was held for 183 days by Ann Roe." in kept  # adds a name
    assert both in kept and kept.index(both) == 2  # in the earlier one's place
    assert not any("Guards beat" in t for t in kept)
    assert report["review"]["repeats"] == 1 and report["review"]["contradictions"] == 1


def test_a_lost_identity_sentence_is_put_back_first():
    from src.engine.written_order import ensure_identity

    notes = [_note(1, "Tom Vale is a lawyer in Port Elm.", "2005-05-06")]
    out = [Written("His career is described.", [1], summary=True, dropped="talks about the page"),
           Written("He taught law.", [1], "Career", 1)]  # fmt: skip
    assert ensure_identity(out, notes, "Tom Vale", "a lawyer in Port Elm", [1])
    assert out[0].text == "Tom Vale is a lawyer in Port Elm." and out[0].notes == [1]
    assert not ensure_identity(out, notes, "Tom Vale", "a lawyer", [1])  # already named
    lost = [Written("He taught law.", [1], "Career", 1)]
    assert not ensure_identity(lost, notes, "Tom Vale", "a lawyer in Elmland", [1])


def test_an_articles_date_on_a_continuing_state_is_caught():
    from src.engine.written_checks import _ABOUT_PAGE, _LASTING

    assert _LASTING.match("In November 2018, Tom Vale was serving as the judge.")
    assert _LASTING.match(
        "In June 2017, Tom Vale became the first man resettled there."
    )
    assert _ABOUT_PAGE.search("Sections cover staffing and conditions.")
    assert _ABOUT_PAGE.search("His career and later firing are described.")
