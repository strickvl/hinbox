"""The written page's last stages: disagreements told, sources not narrated,
two people's dates not taken for a disagreement. Made-up notes only (the
repository is public)."""

from src.engine.written_checks import code_check
from src.engine.written_conflicts import tell_conflicts, unnarrate
from src.engine.written_page import Note, Written
from src.engine.written_plan import (
    Conflict,
    LeftOut,
    Plan,
    PlanPart,
    clean_conflicts,
    restore_on_topic,
)
from src.engine.written_revise import Fill, Revisions


def _note(n: int, text: str, date: str = "2005-05-06", tag: str = "about") -> Note:
    return Note(
        n=n,
        key=f"k{n}",
        text=text,
        articles=[f"a{n}"],
        date=date,
        tag=tag,
        keys=[f"k{n}"],
    )


NOTES = [
    _note(1, "Port Elm is a harbour city in Elmland.", "2003-01-02"),
    _note(2, "Police captured Tom Vale in Port Elm on March 5, 2004.", "2004-03-05"),
    _note(3, "Police arrested Tom Vale in Port Elm in May 2005.", "2005-05-06"),
    _note(4, "Ann Roe was named a forever prisoner in 2009.", "2009-02-02"),
    _note(5, "Lee Moss was named a forever prisoner in 2010.", "2010-02-02"),
]
TEXTS = {n.n: n.text for n in NOTES}


def _conflict(one, other, a="2004", b="2005"):
    return Conflict(
        about="x", one_side=a, one_notes=one, other_side=b, other_notes=other
    )


def test_two_peoples_dates_are_no_disagreement():
    kept = clean_conflicts(
        [_conflict([2], [3]), _conflict([4], [5], "2009", "2010")],
        {1, 2, 3, 4, 5}, None, TEXTS, "Port Elm",
    )  # fmt: skip
    assert [c.one_notes for c in kept] == [[2]]


def test_an_untold_disagreement_gets_a_passage_after_its_first_sentence():
    written = [
        Written("Port Elm is a harbour city in Elmland.", [1]),
        Written("Police arrested Tom Vale in Port Elm in May 2005.", [3], "Arrests", 1),
        Written("The harbour is large.", [1], "Arrests", 1),
    ]
    conflict = _conflict([2], [3]).model_dump()
    seen = []

    def ask(msgs, model, max_tokens):
        seen.append(msgs[0]["content"])
        if model is Fill:
            return Fill.model_validate({"sentences": [{
                "text": "One account dates Tom Vale's capture to March 5, 2004; another "
                "gives May 2005.", "notes": [2, 3]}]})  # fmt: skip
        return model.model_validate({"checks": []})

    report = {}
    out = tell_conflicts(written, NOTES, "Port Elm", "place", [conflict], ask, report)
    assert "disagree" in seen[0] and "Tom Vale" not in seen[0][:40]
    assert out[2].text.startswith("One account dates") and out[2].section == "Arrests"
    assert not out[2].dropped and report["conflicts_untold"] == 0
    # Told already: no call
    seen.clear()
    tell_conflicts(out, NOTES, "Port Elm", "place", [conflict], ask, {})
    assert seen == []


def test_narrating_sentences_are_rewritten_or_kept_as_they_were():
    narr = Written(
        "In May 2005, a news account said police arrested Tom Vale in Port Elm.",
        [3], "Arrests", 1,
    )  # fmt: skip
    told = Written(
        "One account dates Tom Vale's capture to March 5, 2004; another gives May 2005.",
        [2, 3], "Arrests", 1,
    )  # fmt: skip
    conflict = _conflict([2], [3]).model_dump()

    def ask(msgs, model, max_tokens):
        if model is Revisions:
            return Revisions.model_validate({"revisions": [{
                "sentence": 1, "starts_with": "In May 2005,",
                "text": "In May 2005, police arrested Tom Vale in Port Elm."}]})  # fmt: skip
        return model.model_validate({"checks": []})

    report = {}
    out = unnarrate([narr, told], NOTES, "Port Elm", [conflict], ask, report)
    assert report["narrating"] == 1  # the disagreement may name its accounts
    assert (
        narr.dropped
        and out[1].text == "In May 2005, police arrested Tom Vale in Port Elm."
    )
    assert not told.dropped

    def fails(msgs, model, max_tokens):
        raise RuntimeError("down")

    again = Written(narr.revised or narr.text, [3], "Arrests", 1)
    out = unnarrate([again], NOTES, "Port Elm", [], fails, {})
    assert out == [again] and not again.dropped  # narrated beats lost


def test_an_article_date_on_an_older_report_is_flagged():
    notes = [_note(1, "The Senate report said the agency held 119 men.", "2018-04-02")]
    w = Written(
        "In April 2018, the Senate report said the agency held 119 men.", [1], "S"
    )
    ok = Written("In April 2018, the agency said it held 119 men.", [1], "S")
    code_check([w, ok], notes, "The agency")
    assert "older report" in w.dropped and not ok.dropped


def test_a_note_naming_the_persons_surname_goes_back_in():
    by_n = {
        1: _note(1, "Tom Vale was born in Port Elm."),
        2: _note(2, "His lawyer said Vale was a cook, not a fighter.", tag="context"),
        3: _note(3, "Ann Roe ran the harbour.", tag="other"),
    }
    plan = Plan(
        identity="", why_it_matters="", opening_notes=[1],
        parts=[PlanPart(heading="Life", covers="", notes=[1])],
        left_out=[LeftOut(note=2, why="someone else"), LeftOut(note=3, why="someone else")],
    )  # fmt: skip
    restore_on_topic(plan, by_n, "Tom Vale", "person")
    assert plan.parts[0].notes == [1, 2] and [o.note for o in plan.left_out] == [3]


def test_a_rewrite_that_drops_a_named_source_or_adds_a_day_is_refused():
    from src.engine.written_conflicts import lost_or_added

    old = "Tom Vale was arrested in May 2005, the Elm Red Cross reported; another account agreed."
    assert lost_or_added(old, "Tom Vale was arrested in May 2005.")
    assert not lost_or_added(
        old, "Tom Vale was arrested in May 2005, the Elm Red Cross reported."
    )
    assert lost_or_added(
        "A 2014 account described guards feeding men.",
        "On May 19, 2014, guards fed men.",
    )


def test_two_peoples_dates_in_one_note_are_still_no_disagreement():
    texts = {
        6: "Tom Vale was lured from Port Elm to Hilltown in July 2003.",
        7: "Lee Moss, a Hilltown man captured in Port Elm in May 2002, like Tom Vale was named a forever prisoner.",
    }
    c = _conflict([6], [7], "July 2003", "May 2002")
    assert clean_conflicts([c], {6, 7}, None, texts, "Port Elm") == []
    # One man's two dates, another named beside him: kept
    texts = {
        8: "Police captured Tom Vale in Port Elm on March 5, 2004.",
        9: "Police arrested Ann Roe and Tom Vale in Port Elm in May 2005.",
    }
    assert clean_conflicts([_conflict([8], [9])], {8, 9}, None, texts, "Port Elm")


def test_brackets_and_abbreviations():
    from src.engine.written_page import split_sentences, strip_refs

    assert strip_refs("[Clarke] said it was \"hard for [the men]\" there [N3].") == (
        'Clarke said it was "hard for [the men]" there.', [3])  # fmt: skip
    assert (
        strip_refs("It ended. [This note adds no distinct fact, so it is left out.]")[0]
        == "It ended."
    )
    assert split_sentences("He overturned Brig. Gen. John Baker's conviction. It ended.") == [
        "He overturned Brig. Gen. John Baker's conviction.", "It ended."]  # fmt: skip


def test_two_notes_left_out_as_repeating_each_other_bring_one_back():
    from src.engine.written_plan import restore_repeats

    by_n = {
        1: _note(1, "Tom Vale was born in Port Elm."),
        2: _note(2, "Guards called Tom Vale a troublemaker at Hilltown."),
        3: _note(3, "Tom Vale was called a troublemaker by guards at Hilltown."),
    }
    plan = Plan(
        identity="", why_it_matters="", opening_notes=[1],
        parts=[PlanPart(heading="Life", covers="", notes=[1])],
        left_out=[LeftOut(note=2, why="repeats 3"), LeftOut(note=3, why="repeats 2")],
    )  # fmt: skip
    restore_repeats(plan, by_n, "Tom Vale")
    assert plan.parts[0].notes == [1, 2] and [o.note for o in plan.left_out] == [3]


def test_one_side_stated_as_fact_beside_the_disagreement_goes():
    from src.engine.written_conflicts import drop_one_sided

    c = Conflict(about="capture", one_side="Sept. 11, 2002", one_notes=[2],
                 other_side="April 29, 2003", other_notes=[3]).model_dump()  # fmt: skip
    told = Written("One account dates Tom Vale's capture to Sept. 11, 2002; two "
                   "later reports give April 29, 2003.", [2, 3], "Arrests")  # fmt: skip
    plain = Written("A raid on September 11 captured Tom Vale.", [2], "Arrests")
    other = Written(
        "Police arrested Tom Vale and Ann Roe on April 29, 2003.", [3], "Arrests"
    )
    drop_one_sided([told, plain, other], [c])
    assert plain.dropped and not other.dropped and not told.dropped  # Ann Roe kept


def test_a_named_captor_must_stand_beside_a_force_word_in_the_notes():
    from src.engine.written_checks import missing_details

    note = ["Lee Moss was captured in Port Elm, Elmland, in May 2002."]
    assert missing_details(
        "In May 2002, Elmlandish forces captured Lee Moss.", note, "Port Elm"
    )
    note = ["Elmland's security forces arrested Lee Moss in May 2002."]
    assert not missing_details(
        "Elmlandish forces arrested Lee Moss in May 2002.", note, "X"
    )


def test_the_old_profiles_disagreement_sentences_are_not_notes():
    from src.engine.written_page import _OLD_DISPUTE

    assert _OLD_DISPUTE.search("Sources disagree on when Tom Vale was captured.")
    assert _OLD_DISPUTE.search(
        "One article identifies the man as Lee Moss rather than Tom Vale."
    )
    assert not _OLD_DISPUTE.search("Police arrested Tom Vale rather than Lee Moss.")


def test_notes_the_plan_forgot_go_back_in():
    from src.engine.written_plan import restore_orphans

    by_n = {
        1: _note(1, "Port Elm is a harbour city."),
        2: _note(2, "Police captured Tom Vale in Port Elm in March 2004."),
        3: _note(3, "Police arrested Tom Vale again in May 2005."),
        4: _note(4, "Port Elm's harbour has two piers."),
    }
    plan = Plan(
        identity="", why_it_matters="", opening_notes=[1],
        parts=[PlanPart(heading="City", covers="", notes=[1]),
               PlanPart(heading="Arrests", covers="", notes=[2])],
        left_out=[LeftOut(note=4, why="no fact")],
    )  # fmt: skip
    restore_orphans(plan, by_n)
    assert plan.parts[1].notes == [2, 3] and plan.parts[0].notes == [1]


def test_two_accounts_with_different_counts_are_not_unnarrated_and_quotes_close():
    from src.engine.written_conflicts import _two_counts
    from src.engine.written_page import strip_refs

    assert _two_counts(
        "It killed 11 and injured more than 100, one report said; another reported more than 80."
    )
    assert not _two_counts("In May 2005, a news account said police arrested Tom Vale.")
    assert (
        strip_refs("He said to “let them write it down.")[0]
        == "He said to “let them write it down.”"
    )


def test_a_first_sentence_a_note_gives_is_kept_against_the_model_check():
    from src.engine.written_order import keep_identity

    first = Written("Port Elm is a harbour city in Elmland.", [2], "", 0)
    first.dropped = "check: notes don't say it is a harbour city"
    assert keep_identity([first], NOTES, "Port Elm")
    assert not first.dropped and first.notes == [1]
    made_up = Written("Port Elm is the capital of Elmland.", [2], "", 0)
    made_up.dropped = "check: not in the notes"
    assert not keep_identity([made_up], NOTES, "Port Elm") and made_up.dropped


def test_filled_sentences_move_into_date_order():
    from src.engine.written_order import date_sort

    a = Written("In May 2007, Tom Vale said one thing.", [2], "Life", 1)
    b = Written("In February 2018, Tom Vale said another.", [3], "Life", 1)
    c = Written("In June 2017, Tom Vale said a third.", [3], "Life", 1000, filled=True)
    written = [a, b, c]
    assert date_sort(written, NOTES) == 1
    assert written == [a, c, b] and c.para == 1


def test_passive_captors_page_talk_and_named_sources():
    from src.engine.written_checks import missing_details
    from src.engine.written_conflicts import lost_or_added

    note = ["Lee Moss was captured in Port Elm in May 2002."]
    assert missing_details(
        "In May 2002, Lee Moss was captured there by Elmlandish forces.",
        note,
        "Port Elm",
    )
    w = Written(
        "Tom Vale was held there; his transfer is not detailed here.", [2], "Arrests"
    )
    code_check([w], NOTES, "Port Elm")
    assert w.dropped.startswith("check: it talks about the page")
    # A rewrite may drop a word that names no source
    old = "Tom Vale was taken on Sept. 11, 2002; another report describes Elm forces taking him."
    assert not lost_or_added(
        old, "Tom Vale was taken on Sept. 11, 2002, by Elm forces."
    )


def test_notes_are_filed_into_the_plans_sections_one_batch_at_a_time():
    from src.engine.written_file import Filing, file_notes
    from src.engine.written_page import notes_block

    plan = Plan(
        identity="", why_it_matters="", opening_notes=[1],
        parts=[PlanPart(heading="Arrests", covers="who was held", notes=[2]),
               PlanPart(heading="Prisoners", covers="", notes=[3])],
        left_out=[LeftOut(note=4, why="someone else")],
    )  # fmt: skip

    def ask(msgs, model, max_tokens):
        assert model is Filing and "- Arrests: who was held" in msgs[0]["content"]
        return Filing.model_validate({"notes": [
            {"note": 2, "starts_with": "Police captured Tom", "section": "Arrests"},
            {"note": 3, "starts_with": "Police arrested Tom", "section": "arrests"},
            {"note": 4, "starts_with": "Ann Roe was", "section": "Prisoners"},
            {"note": 5, "starts_with": "Wrong words here", "section": "Arrests"},
        ]})  # fmt: skip

    report = {}
    file_notes(plan, NOTES, "Port Elm", "place", ask, report, notes_block)
    assert [p.heading for p in plan.parts] == ["Arrests", "Prisoners"]
    assert plan.parts[0].notes == [2, 3] and plan.parts[1].notes == [4]
    assert plan.left_out == [] and report["filing"]["moved"] == 2


def test_fuller_names_and_lasting_facts():
    from src.engine.written_conflicts import lost_or_added

    assert not lost_or_added("An account said Vale was held.", "Tom Vale was held.")
    notes = [_note(1, "Tom Vale did not want his name published.", "2009-05-02")]
    w = Written("In May 2009, Tom Vale did not want his name published.", [1], "Life")
    code_check([w], notes, "Tom Vale")
    assert "lasting fact" in w.dropped


def test_filing_keeps_events_at_a_place_and_a_persons_own_words():
    from src.engine.written_file import must_keep

    assert must_keep(
        _note(1, "Tom Vale was killed in Port Elm in 2002."), "Port Elm", "place (city)"
    )
    assert not must_keep(
        _note(1, "Port Elm has two piers."), "Port Elm", "place (city)"
    )
    assert must_keep(
        _note(1, "Vale does not want his name published."), "Tom Vale", "person"
    )
    assert must_keep(_note(1, "In 2016, Vale blamed the ban."), "Tom Vale", "person")
    assert not must_keep(
        _note(1, "Ann Roe said the camp was quiet."), "Tom Vale", "person"
    )


def test_a_persons_sections_follow_their_life():
    from src.engine.written_plan import life_order

    by_n = {
        1: _note(1, "Tom Vale was captured in 2001."),
        2: _note(2, "Tom Vale, captured in 2001, was sent to Elmland in 2016."),
        3: _note(3, "Elmland's court ruled on his transfer in 2017."),
        4: _note(4, "A 2007 profile called Tom Vale a troublemaker."),
        5: _note(5, "His lawyer said he was friendly."),
    }
    parts = [PlanPart(heading="Capture", covers="", notes=[1]),
             PlanPart(heading="Elmland", covers="", notes=[2, 3]),
             PlanPart(heading="Portraits", covers="", notes=[4, 5])]  # fmt: skip
    assert [p.heading for p in life_order(parts, by_n)] == [
        "Capture",
        "Portraits",
        "Elmland",
    ]


def test_a_captor_must_be_in_the_captives_own_note():
    from src.engine.written_checks import missing_details

    notes = [
        "Lee Moss was captured in Port Elm in May 2002.",
        "Elmlandish forces captured Ann Roe in Port Elm in June 2002.",
    ]
    s = "In May, Elmlandish forces captured Lee Moss there; in June, they captured Ann Roe."
    assert missing_details(s, notes, "Port Elm") == ["Elmlandish forces"]
    assert not missing_details(
        "In June, Elmlandish forces captured Ann Roe there.", notes, "Port Elm"
    )


def test_a_date_against_no_date_is_no_disagreement():
    c = Conflict(about="capture", one_side="Sept. 10-11, 2002", one_notes=[2],
                 other_side="a series of raids, no date given", other_notes=[3])  # fmt: skip
    assert clean_conflicts([c], {2, 3}) == []


def test_unwritten_lists_planned_notes_no_sentence_tells():
    from src.engine.written_revise import unwritten

    written = [
        Written(
            "Police captured Tom Vale in Port Elm on March 5, 2004.", [2], "Arrests"
        )
    ]
    parts = [{"heading": "Arrests", "notes": [2, 3]}]
    assert unwritten(written, NOTES, "Port Elm", parts) == [3]
