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


def test_the_page_review_judges_pairs_code_picks_and_rewrites_small():
    from src.engine.written_checks import SentenceChecks
    from src.engine.written_review import Verdicts, candidate_pairs, review_page
    from src.engine.written_revise import Revisions

    notes = [
        _note(1, "Port Elm is a harbour city.", "2003-01-02"),
        _note(2, "Police held Tom Vale 183 days in the Port Elm jail.", "2004-01-01"),
        _note(
            3, "A 2014 report says Tom Vale was not beaten in Port Elm.", "2014-12-01"
        ),
        _note(
            4, "An earlier account says guards beat Tom Vale in Port Elm.", "2009-01-01"
        ),
    ]
    out = [
        Written("Port Elm is a harbour city.", [1]),
        Written("Police held Tom Vale for 183 days in the city jail.", [2], "Arrests", 1),
        Written("A 2014 report says guards did not beat Tom Vale in the city jail.", [3], "Arrests", 1),
        Written("Guards beat Tom Vale in the city jail.", [4], "Prison", 2),
        Written("Police held Tom Vale for 183 days.", [2], "Prison", 2),
        Written("Police held Tom Vale for 183 days in the city jail, far from home.", [2], "Prison", 2),
    ]  # fmt: skip
    kept = [w for w in out if not w.dropped]
    pairs = candidate_pairs(kept, "Port Elm")
    assert (2, 3) in pairs and (1, 4) in pairs and (1, 5) in pairs
    assert all(i < j and kept[i].section for i, j in pairs)  # never the opening
    other = "Another account says guards beat Tom Vale in the city jail."

    def ask(msgs, model, max_tokens):
        if model is Verdicts:
            body = msgs[1]["content"]
            verdicts = []
            for k, (i, j) in enumerate(pairs, 1):
                v = {(2, 3): "conflict", (1, 4): "same", (1, 5): "same"}.get(
                    (i, j), "different"
                )
                verdicts.append({"pair": k, "starts_with": kept[j].text, "verdict": v})
            assert "Pair 1:" in body
            return Verdicts.model_validate({"pairs": verdicts})
        if model is Revisions:
            revs = []
            for k, part in enumerate(msgs[1]["content"].split("Sentence ")[1:], 1):
                if part.split(": ", 1)[1].startswith("Guards beat"):
                    revs.append(
                        {"sentence": k, "starts_with": "Guards beat Tom", "text": other}
                    )
                else:  # the trim gives up
                    revs.append(
                        {"sentence": k, "starts_with": "Police held Tom", "text": ""}
                    )
            return Revisions.model_validate({"revisions": revs})
        return SentenceChecks.model_validate({"checks": [
            {"sentence": 1, "starts_with": other, "problem": "none", "verdict": "supported"}]})  # fmt: skip

    report = {"calls": 0}
    res = review_page(out, notes, "Port Elm", "place", ask, report)
    kept_now = [w.text for w in res if not w.dropped]
    assert "Police held Tom Vale for 183 days." not in kept_now  # adds nothing
    # It adds "far from home": the trim gave up, so it stays as it was
    assert (
        "Police held Tom Vale for 183 days in the city jail, far from home." in kept_now
    )
    assert (
        other in kept_now and "Guards beat Tom Vale in the city jail." not in kept_now
    )
    assert (
        "A 2014 report says guards did not beat Tom Vale in the city jail." in kept_now
    )
    assert report["review"]["dropped"] == 1 and report["review"]["restored"] == 1


def test_a_lost_identity_sentence_is_put_back_first():
    from src.engine.written_order import ensure_identity

    notes = [_note(1, "Tom Vale is a lawyer in Port Elm.", "2005-05-06")]
    out = [Written("His career is described.", [1], summary=True, dropped="talks about the page"),
           Written("He taught law.", [1], "Career", 1)]  # fmt: skip
    assert ensure_identity(out, notes, "Tom Vale", "A lawyer in Port Elm", [1])
    assert out[0].text == "Tom Vale is a lawyer in Port Elm." and out[0].notes == [1]
    assert not ensure_identity(out, notes, "Tom Vale", "a lawyer", [1])  # already named
    lost = [Written("He taught law.", [1], "Career", 1)]
    assert not ensure_identity(lost, notes, "Tom Vale", "a judge in Elmland", [1])
    # The whole identity fails the check; its first part is used
    assert ensure_identity(lost, notes, "Tom Vale", "a lawyer and Elmland judge", [1])
    assert lost[0].text == "Tom Vale is a lawyer."


def test_an_articles_date_on_a_continuing_state_is_caught():
    from src.engine.written_checks import _ABOUT_PAGE, _LASTING

    assert _LASTING.match("In November 2018, Tom Vale was serving as the judge.")
    assert _LASTING.match(
        "In June 2017, Tom Vale became the first man resettled there."
    )
    assert _ABOUT_PAGE.search("Sections cover staffing and conditions.")
    assert _ABOUT_PAGE.search("His career and later firing are described.")
