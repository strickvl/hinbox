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
    # Taken as the plan wrote it, whatever the code check would say
    lost = [Written("He taught law.", [1], "Career", 1)]
    assert ensure_identity(lost, notes, "Tom Vale", "a lawyer in Elmland", [1])
    assert lost[0].text == "Tom Vale is a lawyer in Elmland."


def test_a_first_sentence_the_checks_dropped_comes_back_when_the_identity_says_it():
    from src.engine.written_order import keep_identity

    notes = [_note(1, "Port Elm police arrested Tom Vale.", "2005-05-06")]
    out = [Written("Port Elm is a city in Elmland.", [1], dropped="not in its notes: Elmland"),
           Written("Police arrested Tom Vale.", [1], "Arrests", 1)]  # fmt: skip
    assert not keep_identity(out, notes, "Port Elm", "a harbour city")
    assert keep_identity(out, notes, "Port Elm", "a city in Elmland")
    assert not out[0].dropped


def test_an_articles_date_on_a_continuing_state_is_caught():
    from src.engine.written_checks import _ABOUT_PAGE, _LASTING

    assert _LASTING.match("In November 2018, Tom Vale was serving as the judge.")
    assert _LASTING.match(
        "In June 2017, Tom Vale became the first man resettled there."
    )
    assert _ABOUT_PAGE.search("Sections cover staffing and conditions.")
    assert _ABOUT_PAGE.search("His career and later firing are described.")
    assert _ABOUT_PAGE.search("They also recount Ann Roe's move.")
    from src.engine.written_checks import code_check

    empty = Written("In March 2013, the counts differed from those in April.", [1], "S")
    code_check(
        [empty],
        [_note(1, "Counts were 24 in March and 84 in April.", "2013-04-01")],
        "Port Elm",
    )
    assert empty.dropped.startswith("says only that counts differ")


def test_dispute_sentences_whose_accounts_agree_are_found():
    from src.engine.written_conflicts import _agree

    assert _agree(
        "He was appointed on April 3, 2017, although another account gives only April 2017."
    )
    assert _agree(
        "One dates it to a raid on Sept. 10-11, 2002, while another account places it in Karachi without giving a date."
    )
    assert _agree(
        "The charges said it killed 10 Indonesians and a Dutch citizen, while three reports said 11 people died."
    )
    assert _agree(
        "The Pentagon gave his age as 36, while another report described him as in his mid-30s."
    )
    assert not _agree(
        "One account dates it to Sept. 11, 2002; three later reports give April 29, 2003."
    )


def test_neighbours_saying_one_thing_and_ranked_first_names():
    from src.engine.written_checks import drop_neighbours

    out = [
        Written("In January 2018, he struck down Tom Vale's $1,000 fine.", [1], "Rulings", 1),
        Written("In January 2018, Ann Roe struck down the $1,000 fine imposed on Gen. Tom Vale.", [2], "Rulings", 1),
        Written("In March 2018, she left.", [3], "Rulings", 1),
    ]  # fmt: skip
    assert drop_neighbours(out) == 1 and out[0].dropped and out[1].notes == [2, 1]
    notes = ["House said the strike grew.", "David Frakt spoke."]
    assert missing_details(
        "Rear Adm. David House said the strike grew.", notes, "Port Elm"
    )
    assert (
        missing_details("President Tom Vale spoke.", ["Vale spoke."], "Port Elm") == []
    )


def test_each_section_comes_from_the_draft_that_scores_best():
    from src.engine.written_draws import pick_sections, section_score

    notes = [
        _note(1, "Port Elm is a harbour city.", "2003-01-02"),
        _note(2, "Police arrested Tom Vale in March 2004.", "2004-03-05"),
        _note(3, "Police arrested Ann Roe in May 2005.", "2005-05-06"),
        _note(4, "The harbour opened in 1990.", "2006-01-01"),
    ]
    good = [
        Written("Police arrested Tom Vale in March 2004.", [2], "Arrests", 1),
        Written("Police arrested Ann Roe in May 2005.", [3], "Arrests", 1),
    ]
    bad = [
        Written("He was arrested in May 2005.", [3], "Arrests", 1),
        Written("One account says Tom Vale was held in 2004; another account says 2005.", [2], "Arrests", 1),
    ]  # fmt: skip
    assert section_score(good, [2, 3], notes, [], "Port Elm") == 0
    assert section_score(bad, [2, 3], notes, [], "Port Elm") >= 6
    d1 = (
        [Written("Port Elm is a harbour city.", [1])]
        + bad
        + [Written("The harbour opened in 1990.", [4], "Harbour", 2)]
    )
    d2 = [Written("Port Elm is a city.", [1])] + good
    parts = [
        {"heading": "Arrests", "notes": [2, 3]},
        {"heading": "Harbour", "notes": [4]},
    ]
    report = {}
    page = pick_sections([d1, d2], parts, notes, [], "Port Elm", report)
    texts = [w.text for w in page]
    assert texts[1:3] == [good[0].text, good[1].text]  # arrests from draft 2
    assert texts[-1] == "The harbour opened in 1990."  # harbour from draft 1
    assert report["draws"]["chosen"] == {"Arrests": 1, "Harbour": 0}
    assert len({w.para for w in page if w.section}) == 2


def test_the_section_score_counts_repeats_lost_content_and_only_wrong_dates():
    from src.engine.written_draws import drop_across, section_score

    notes = [
        _note(1, "Tom Vale said in a talk that the city was calm.", "2016-10-02"),
        _note(2, "Ann Roe was held 183 days in 2003.", "2016-10-02"),
        _note(3, "Lee Moss was resettled in January 2016.", "2017-06-01"),
    ]
    # A remark dated by its article: right, no penalty
    said = [Written("In October 2016, Tom Vale said the city was calm.", [1], "S")]
    assert section_score(said, [1], notes, [], "Port Elm") == 0
    # An event its note dates otherwise, given the article's date
    moved = [Written("In June 2017, Lee Moss was resettled.", [3], "S")]
    assert section_score(moved, [3], notes, [], "Port Elm") >= 3
    # 183 twice with no shared year; note 2's 2003 never said
    twice = [Written("Ann Roe was held for 183 days.", [2], "S"),
             Written("Guards kept Ann Roe 183 days.", [2], "S")]  # fmt: skip
    assert section_score(twice, [2], notes, [], "Port Elm") == 3
    # Note 3 planned, uncited, its person nowhere: 1 + 2 (its 2016 is said)
    assert section_score(said, [1, 3], notes, [], "Port Elm") == 3
    page = [Written("Ann Roe was held 183 days in 2003.", [2], "A"),
            Written("Ann Roe was held for 183 days in 2003.", [2], "B")]  # fmt: skip
    assert drop_across(page, "Port Elm") == 1 and page[1].dropped


def test_sentences_leaning_on_nothing_before_them_cost_the_section():
    from src.engine.written_draws import _leans_on_nothing

    def w(t: str) -> Written:
        return Written(t, [1], "S")

    dated = w("Police arrested Tom Vale in March 2004.")
    plain = w("Police arrested Tom Vale.")
    assert not _leans_on_nothing(dated, w("That month, Vale was charged."), "Port Elm")
    assert _leans_on_nothing(plain, w("That month, Vale was charged."), "Port Elm")
    memo = w("A memo set out the rules.")
    assert not _leans_on_nothing(memo, w("The memo also barred visits."), "Port Elm")
    assert _leans_on_nothing(plain, w("The memo also barred visits."), "Port Elm")
    assert not _leans_on_nothing(plain, w("He was later freed."), "Port Elm")
    assert _leans_on_nothing(memo, w("He was later freed."), "Port Elm")


def test_ties_go_to_fewer_repeats_and_lost_marks_count_against_a_draft():
    from src.engine.written_draws import _lost_against, pick_sections

    notes = [
        _note(1, "Port Elm is a harbour city.", "2003-01-02"),
        _note(2, "Police arrested Tom Vale in March 2004.", "2004-03-05"),
    ]
    one = [Written("Port Elm is a harbour city.", [1]),
           Written("Police arrested Tom Vale in March 2004.", [2], "Arrests", 1)]  # fmt: skip
    two = [Written("Port Elm is a harbour city.", [1]),
           Written("Police arrested Tom Vale in March 2004.", [2], "Arrests", 1),
           Written("In March 2004 police arrested Tom Vale at home.", [2], "Arrests", 1)]  # fmt: skip
    report = {}
    parts = [{"heading": "Arrests", "notes": [2]}]
    pick_sections([two, one], parts, notes, [], "Port Elm", report)
    assert report["draws"]["chosen"] == {"Arrests": 1}  # the one saying it once
    lost = [Written("Police arrested a man.", [2], "Arrests", 1)]
    split = [{"Arrests": one[1:]}, {"Arrests": two[1:]}, {"Arrests": lost}]
    costs = _lost_against(split, "Arrests", "Port Elm")
    assert costs[0] == 0 and costs[2] >= 4  # 2004 and Vale lost


def test_a_planned_conflict_whose_sides_add_up_is_dropped():
    from src.engine.written_plan import Conflict, clean_conflicts

    c = Conflict(
        about="the dead", one_side="10 Elmish and a Dutch citizen were killed",
        one_notes=[1], other_side="11 people were killed", other_notes=[2],
    )  # fmt: skip
    assert clean_conflicts([c], {1, 2}) == []
    real = Conflict(
        about="the arrest", one_side="arrested in March 2004", one_notes=[1],
        other_side="arrested in May 2005", other_notes=[2],
    )  # fmt: skip
    assert len(clean_conflicts([real], {1, 2})) == 1


def test_planned_conflicts_are_confirmed_and_sections_picked_by_a_call():
    from src.engine.written_pick import (
        ConflictVerdicts,
        Pick,
        ask_pick,
        confirm_conflicts,
    )
    from src.engine.written_plan import Conflict

    notes = {
        1: _note(1, "On April 1, 11 captives were tube-fed.", "2013-04-02"),
        2: _note(2, "On April 21, 16 captives were tube-fed.", "2013-04-22"),
        3: _note(3, "Tom Vale was arrested in March 2004.", "2004-03-05"),
        4: _note(4, "Tom Vale was arrested in May 2005.", "2005-05-06"),
    }
    counts = Conflict(about="tube-fed", one_side="11", one_notes=[1],
                      other_side="16", other_notes=[2])  # fmt: skip
    arrest = Conflict(about="arrest", one_side="March 2004", one_notes=[3],
                      other_side="May 2005", other_notes=[4])  # fmt: skip
    seen = []

    def ask(msgs, model, max_tokens):
        seen.append(msgs[1]["content"])
        if model is ConflictVerdicts:
            return model.model_validate({"items": [
                {"item": 1, "fact": "count", "verdict": "compatible"},
                {"item": 2, "fact": "arrest date", "verdict": "contradict"}]})  # fmt: skip
        return Pick.model_validate({"why": "keeps the date", "best": 2})

    report = {}
    block = lambda ns: "\n".join(f"N{n.n}: {n.text}" for n in ns)  # noqa: E731
    kept = confirm_conflicts(
        [counts, arrest], notes, "Port Elm", "place", ask, report, block
    )
    assert kept == [arrest] and len(report["conflicts_dropped"]) == 1
    assert "N1: On April 1" in seen[0]  # each side shown with its notes

    def broken(msgs, model, max_tokens):
        raise RuntimeError("down")

    assert confirm_conflicts(
        [counts], notes, "Port Elm", "place", broken, {}, block
    ) == [counts]
    texts = ["Vale was arrested.", "Vale was arrested in March 2004."]
    assert (
        ask_pick(texts, "Arrests", [notes[3]], "Port Elm", "place", ask, report, block)
        == 1
    )
    assert report["picks"]["Arrests"]["best"] == 1
    assert (
        ask_pick(["Same.", "Same."], "A", [], "Port Elm", "place", ask, {}, block)
        is None
    )
    assert ask_pick(texts, "A", [], "Port Elm", "place", broken, {}, block) is None


def test_the_identity_sentence_takes_a_or_an_by_sound_and_the_for_agencies():
    from src.engine.written_order import ensure_identity

    notes = [_note(1, "The Elm Agency is an Elmish spy service.", "2005-05-06")]
    out = [Written("It ran prisons.", [1], "Prisons", 1)]
    assert ensure_identity(out, notes, "Elm Agency", "U.S. intelligence agency", [1])
    assert out[0].text == "The Elm Agency is a U.S. intelligence agency."
    out = [Written("It ran prisons.", [1], "Prisons", 1)]
    assert ensure_identity(out, notes, "Tom Vale", "officer in the navy", [1])
    assert out[0].text == "Tom Vale is an officer in the navy."


def test_a_comparison_note_is_cut_to_the_articles_side():
    from src.engine.written_page import article_side

    assert (
        article_side(
            "However, the new article reports that Tom Vale couriered $500 to Elmtown,"
            " differing from the profile's description of him collecting funds."
        )
        == "Tom Vale couriered $500 to Elmtown."
    )
    assert (
        article_side(
            "However, the article reports that the recess lasted 35 minutes, rather"
            " than the recess occurring after 30 minutes."
        )
        == "The recess lasted 35 minutes."
    )
    assert article_side("The profile was wrong about it.") == ""
    plain = "Tom Vale chose to plead rather than go to trial."
    assert article_side(plain) == plain  # no comparison: kept whole


def test_an_articles_date_on_a_one_time_event_dated_otherwise_is_caught():
    from src.engine.written_checks import _LASTING, code_check

    assert _LASTING.match(
        "In June 2017, Ann Roe, an Elmish national, and Lee Moss became the first men resettled there."
    )
    notes = [
        _note(1, "Ann Roe was also fired alongside Tom Vale.", "2018-05-02"),
        _note(2, "Ann Roe was fired in February 2018.", "2018-02-06"),
    ]
    w = Written("In May 2018, Ann Roe was also fired alongside Tom Vale.", [1], "S")
    code_check([w], notes, "Port Elm")
    assert w.dropped.startswith("check: it puts the article's date on a one-time")
    alone = Written("In May 2018, Ann Roe was also fired alongside Tom Vale.", [1], "S")
    code_check([alone], notes[:1], "Port Elm")
    assert not alone.dropped  # no other note dates it: the article's date stands


def test_the_comparison_cut_keeps_the_articles_own_rather_than():
    from src.engine.written_page import article_side

    assert article_side(
        "However, the article reports that Tom Vale was fired on May 5, 2018, citing"
        " his manner rather than the plea talks"
    ) == (
        "Tom Vale was fired on May 5, 2018, citing his manner rather than the plea talks."
    )
    assert (
        article_side(
            "However, the article reports that Tom Vale carried $500 that killed 11 people,"
            " indicating a discrepancy in the figures compared to the charges which alleged 12"
        )
        == "Tom Vale carried $500 that killed 11 people."
    )
    assert (
        article_side(
            "The sources disagree on the venue: the profile states Elmtown, while the"
            " article states Ann Roe was tried in Port Elm"
        )
        == "Ann Roe was tried in Port Elm."
    )
    assert (
        article_side(
            "However, the article reports the jail holds 41 men, and notes that Lee Moss spoke"
        )
        == "The jail holds 41 men, and Lee Moss spoke."
    )


def test_statements_on_the_day_keep_their_article_date():
    from src.engine.written_checks import code_check

    notes = [_note(1, "Tom Vale said the men had expected the order.", "2018-02-02")]
    said = Written(
        "In February 2018, Tom Vale said the men had expected the order.", [1], "S"
    )
    code_check([said], notes, "Port Elm")
    assert not said.dropped
    state = [_note(2, "Tom Vale was directly involved in the talks.", "2018-11-02")]
    was = Written(
        "In November 2018, Tom Vale was directly involved in the talks.", [2], "S"
    )
    code_check([was], state, "Port Elm")
    assert was.dropped.startswith("check: it puts the article's date on a lasting")


def test_repeated_clauses_and_report_wrapped_repeats_are_cut():
    from src.engine.written_clauses import drop_repeat_clauses, drop_report_repeats

    page = [
        Written("On November 3, Tom Vale suspended Ann Roe's 21-day sentence.", [1], "S"),
        Written(
            "In February 2018, Tom Vale backed the judge, and suspended Ann Roe's sentence.",
            [2], "S",
        ),
        Written("Ann Roe, Lee Moss, and Sam Hart were held.", [3], "S"),
    ]  # fmt: skip
    assert len(drop_repeat_clauses(page, "Port Elm")) == 1
    assert page[1].text == "In February 2018, Tom Vale backed the judge."
    assert page[2].text == "Ann Roe, Lee Moss, and Sam Hart were held."
    fired = [
        Written("In February 2018, Tom Vale and Ann Roe were dismissed.", [1], "S"),
        Written("In May 2018, reports confirmed that Ann Roe was dismissed alongside Tom Vale.", [2], "S"),
        Written("A May 2018 report said Lee Moss was named judge.", [3], "S"),
    ]  # fmt: skip
    assert len(drop_report_repeats(fired, "Port Elm")) == 1
    assert fired[1].dropped and not fired[2].dropped


def test_weekdays_and_yearless_months_take_their_articles_date():
    from src.engine.written_dates import own_dates

    notes = [
        _note(1, "As of Wednesday, 100 men were on hunger strike.", "2013-04-26"),
        _note(2, "In September, a prisoner was found dead.", "2013-04-02"),
        _note(3, "The Pentagon's November policy banned artwork.", "2018-02-10"),
        _note(4, "Police arrested Tom Vale in May 2004.", "2004-05-09"),
    ]
    page = [
        Written("As of Wednesday, 100 men were on hunger strike.", [1], "S"),
        Written("In September, a prisoner was found dead.", [2], "S"),
        Written("The Pentagon's November policy banned artwork.", [3], "S"),
        Written("Police arrested Tom Vale in May 2004.", [4], "S"),
        Written("Tom Vale may appeal.", [4], "S"),
    ]
    changed = own_dates(page, notes)
    assert page[0].text == "As of April 24, 2013, 100 men were on hunger strike."
    assert page[1].text == "In September 2012, a prisoner was found dead."
    assert page[2].text == "The Pentagon's November 2017 policy banned artwork."
    assert page[3].text == "Police arrested Tom Vale in May 2004."
    assert page[4].text == "Tom Vale may appeal." and len(changed) == 3


def test_a_section_sentence_adding_nothing_is_dropped_and_struck_notes_obey():
    from src.engine.written_clauses import drop_section_repeats
    from src.engine.written_page import page_notes

    page = [
        Written("In April 2013, most men awaiting transfer were Elmish, including Tom Vale.", [1], "S"),
        Written("In April 2013, most of the men awaiting transfer were Elmish, including Vale.", [2], "S"),
        Written("In April 2013, Tom Vale testified that he gathered experts and said a call came first.", [3], "S"),
    ]  # fmt: skip
    assert drop_section_repeats(page, "Port Elm") == [page[1].text]
    assert not page[2].dropped
    city = {
        "name": "Port Elm",
        "profile": {"text": "Ann Roe is Tom Vale's widow.^[a1] The port is old.^[a2]"},
        "articles": [{"article_id": "a1", "published_date": "2005-01-01"},
                     {"article_id": "a2", "published_date": "2005-01-02"}],
    }  # fmt: skip
    strikes = {
        "Ann Roe is Tom Vale's widow.": {
            "replacement": "Ann Roe is Tom Vale's mother.",
            "replacement_cites": "a2",
        }
    }
    notes = page_notes(city, strikes=strikes)
    assert notes[0].text == "Ann Roe is Tom Vale's mother." and notes[0].articles == [
        "a2"
    ]
    assert len(page_notes(city, strikes={"The port is old.": {}})) == 1
