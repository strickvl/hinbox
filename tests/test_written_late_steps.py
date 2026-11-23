"""The last steps of a written page: dating "As of" days, the date and actor
check, and notes telling one fact joined before planning. Made-up notes
only (the repository is public)."""

from src.engine.written_page import Written
from tests.test_written_conflicts import _note


def test_as_of_days_take_their_year():
    from src.engine.written_dates import own_dates

    notes = [_note(1, "As of April 24, 100 men struck.", "2013-04-26")]
    page = [Written("As of April 24, 100 men struck.", [1], "S")]
    own_dates(page, notes)
    assert page[0].text == "As of April 24, 2013, 100 men struck."


def test_when_who_mends_a_misdated_sentence_and_drops_a_wrong_actor():
    from src.engine.written_when_who import WhenWhoChecks, usable_fix, when_who

    notes = [
        _note(1, "The 2014 Senate report found the program cruel.", "2018-04-02"),
        _note(2, "Judge Tom Vale ruled the motion moot.", "2018-05-01"),
    ]
    page = [
        Written("In April 2018, the Senate report found the program cruel.", [1], "S"),
        Written("Lawyer Ann Roe ruled the motion moot.", [2], "S"),
    ]

    def ask(msgs, model, max_tokens):
        assert model is WhenWhoChecks and "Its notes:" in msgs[1]["content"]
        return WhenWhoChecks.model_validate({"checks": [
            {"sentence": 1, "starts_with": "In April 2018", "when": "wrong", "who": "right",
             "problem": "the report is from 2014",
             "fixed": "In 2014, the Senate report found the program cruel."},
            {"sentence": 2, "starts_with": "Lawyer Ann Roe", "when": "no date", "who": "wrong",
             "problem": "the judge ruled", "fixed": "Lawyer Ann Roe's judge, Sam Lee, ruled it."},
        ]})  # fmt: skip

    report: dict = {}
    when_who(page, notes, "Port Elm", ask, report)
    assert page[0].text == "In 2014, the Senate report found the program cruel."
    assert not page[1].dropped  # no date question about it, so not asked
    assert len(report["when_who"]) == 1 and report["when_who_asked"] == 1
    assert report["calls"] == 1
    # a year the date pass worked out, a date the note writes and "said" stay
    note = [
        "On Monday, the court met. In September, a man died. It quoted a November 2003 cable."
    ]
    assert not usable_fix(
        "The court met on September 24, 2018.",
        "The court met on Monday.",
        note,
        "Port Elm",
        [],
        ["September", "24", "2018"],
    )
    assert not usable_fix(
        "A November 2003 cable was quoted.", "A cable was quoted.", note, "Port Elm", []
    )
    assert not usable_fix(
        "In 2012, officials said the jail closed in 2003.",
        "In 2012, the jail closed in 2003.",
        ["The jail closed in 2003."],
        "Port Elm",
        [],
    )
    assert usable_fix(
        "In June 2017, Tom Vale was resettled.",
        "Tom Vale was resettled.",
        ["Tom Vale was resettled in January 2016."],
        "Port Elm",
        [],
    )
    # a fix that rewrites the sentence is refused
    assert not usable_fix(
        "Tom Vale ruled the motion moot.",
        "The motion, a long one, was ruled moot by the court after a hearing.",
        ["Tom Vale ruled the motion moot."],
        "Port Elm",
        [],
    )


def test_a_month_joined_to_a_dated_month_takes_no_article_year():
    from src.engine.written_dates import own_dates

    notes = [_note(1, "He was waterboarded in May and July 2003.", "2015-06-02")]
    page = [Written("He was waterboarded in May and July 2003.", [1], "S")]
    own_dates(page, notes)
    assert page[0].text == "He was waterboarded in May and July 2003."


def test_only_sentences_with_a_date_question_go_to_the_date_check():
    from src.engine.written_when_who import suspect

    by_n = {
        1: _note(1, "He was waterboarded in May and July 2003.", "2015-06-02"),
        2: _note(2, "In September, a man died.", "2013-04-02"),
    }
    assert "2015" in suspect(
        Written("He was waterboarded in May 2015 and July 2003.", [1], "S"), by_n
    )
    # a year the date pass worked out where the note gives none is not asked about
    assert suspect(Written("In September 2012, a man died.", [2], "S"), by_n) == ""
    assert suspect(Written("That month, a man died.", [2], "S"), by_n)


def test_month_year_pairs_are_asked_about_and_that_month_is_named_in_code():
    from src.engine.written_when_who import (
        WhenWhoChecks,
        month_years,
        suspect,
        when_who,
    )

    assert month_years("in May and July 2003") == {("May", "2003"), ("July", "2003")}
    by_n = {
        1: _note(
            1,
            "In June 2015, lawyers said he was hurt in May and July 2003.",
            "2015-06-02",
        )
    }
    assert "May 2015" in suspect(
        Written("He was hurt in May 2015 and July 2003.", [1], "S"), by_n
    )
    assert suspect(Written("He was hurt in May and July 2003.", [1], "S"), by_n) == ""

    notes = [_note(2, "Spokesman Tom Vale said the copy was kept.", "2016-12-30")]
    page = [Written("That month, spokesman Tom Vale said the copy was kept.", [2], "S")]

    def ask(msgs, model, max_tokens):
        return WhenWhoChecks.model_validate({"checks": [
            {"sentence": 1, "starts_with": "That month, spokesman", "when": "right",
             "who": "right", "problem": "The month is December 2016", "fixed": ""},
        ]})  # fmt: skip

    report: dict = {}
    when_who(page, [notes[0]], "Port Elm", ask, report)
    assert (
        page[0].text == "In December 2016, spokesman Tom Vale said the copy was kept."
    )


def test_date_check_guards_follow_month_year_pairs_and_article_years():
    from src.engine.written_when_who import WhenWhoChecks, when_who

    def run(text, note, problem, fixed, date="2015-06-02", own_dates=()):
        notes = [_note(1, note, date)]
        page = [Written(text, [1], "S")]

        def ask(msgs, model, max_tokens):
            return WhenWhoChecks.model_validate({"checks": [{
                "sentence": 1, "starts_with": " ".join(text.split()[:3]), "when": "wrong",
                "who": "right", "problem": problem, "fixed": fixed}]})  # fmt: skip

        report = {"own_dates": [list(p) for p in own_dates]}
        when_who(page, notes, "Port Elm", ask, report)
        return page[0].text

    # Khan: the date pass's "May 2015" goes, though 2015 is in the note
    khan = "Lawyers said he was hurt in May 2015 and July 2003."
    assert run(
        khan,
        "In June 2015, lawyers said he was hurt in May and July 2003.",
        "May 2015 should be May 2003",
        "Lawyers said he was hurt in May and July 2003.",
        own_dates=[("Lawyers said he was hurt in May and July 2003.", khan)],
    ) == ("Lawyers said he was hurt in May and July 2003.")
    # Trivett: "That month" is not named from a January article when the note dates the act
    assert run(
        "That month, Tom Vale also signed a directive.",
        "Tom Vale signed a Nov. 17 directive.",
        "The directive was signed November 17, 2017; the article is January 2018",
        "In January 2018, Tom Vale also signed a directive.",
        date="2018-01-19",
    ).startswith("That month")
    # Cowan: the year of an article is kept on what was reported then
    cowan = "Ann Roe reported in January 2015 that he was a client."
    assert (
        run(
            cowan,
            "Ann Roe, who met him in 2009, said he was a client.",
            "the report is from 2015?",
            "Ann Roe reported that he was a client.",
            date="2015-01-10",
        )
        == cowan
    )


def test_notes_telling_one_fact_are_joined_only_when_nothing_is_lost():
    from src.engine.written_same import PairsSame, holds, join_same

    notes = [
        _note(
            1, "On March 4, 2010, Judge Tom Vale ruled the motion moot.", "2010-03-05"
        ),
        _note(2, "Judge Tom Vale ruled the motion moot.", "2010-03-09"),
        _note(
            3,
            "On March 4, 2010, Judge Tom Vale ruled the motion moot and fined Ann Roe.",
            "2010-03-06",
        ),
        _note(
            4,
            "On March 9, 2010, Judge Tom Vale ruled a second motion moot.",
            "2010-03-10",
        ),
    ]
    for i, n in enumerate(notes):
        n.articles, n.keys = [f"a{i + 1}"], [f"k{i + 1}"]
    asked = []

    def ask(msgs, model, max_tokens):
        assert model is PairsSame
        asked.append(msgs[1]["content"])
        n = msgs[1]["content"].count("Pair ")
        return PairsSame.model_validate({"pairs": [
            {"pair": k, "tells": "a ruling", "verdict": "same"} for k in range(1, n + 1)
        ]})  # fmt: skip

    report: dict = {}
    out = join_same(notes, "Port Elm", "place", ask, report)
    texts = [n.text for n in out]
    # 2 goes into 1, 1 into 3; 4 (another day) is held by none, so it stays
    assert texts == [notes[2].text, notes[3].text]
    assert out[0].articles == ["a3", "a1", "a2"] or set(out[0].articles) == {
        "a1",
        "a2",
        "a3",
    }
    assert out[0].date == "2010-03-06" and len(report["same_joined"]) == 2
    assert not holds(
        notes[0].text, notes[2].text, "Port Elm"
    )  # "Ann Roe" would be lost


def test_two_notes_that_each_add_a_detail_are_written_as_one_only_if_nothing_is_lost():
    from src.engine.written_same import JoinedNotes, PairsSame, join_same, joined_ok

    a = "Tom Vale suspended Ann Roe's 21-day sentence pending a review."
    b = "On November 3, 2017, Tom Vale suspended the sentence, telling Roe by letter."
    good = "On November 3, 2017, Tom Vale suspended Ann Roe's 21-day sentence pending a review, telling Roe by letter."
    assert joined_ok(good, a, b, "Port Elm")
    assert not joined_ok(
        "Tom Vale suspended Ann Roe's sentence.", a, b, "Port Elm"
    )  # lost 21, November
    assert not joined_ok(
        good.replace("Tom Vale", "Judge Sam Lee"), a, b, "Port Elm"
    )  # a new name
    notes = [_note(1, a, "2017-11-05"), _note(2, b, "2017-11-04")]
    for i, n in enumerate(notes):
        n.articles, n.keys = [f"a{i + 1}"], [f"k{i + 1}"]

    def ask(msgs, model, max_tokens):
        if model is PairsSame:
            return PairsSame.model_validate({"pairs": [{"pair": 1, "verdict": "same"}]})
        assert model is JoinedNotes and "A: Tom Vale" in msgs[1]["content"]
        return JoinedNotes.model_validate({"joined": [{"pair": 1, "text": good}]})

    report: dict = {}
    out = join_same(notes, "Port Elm", "place", ask, report, write=True)
    assert [n.text for n in out] == [good] and set(out[0].articles) == {"a1", "a2"}
    assert out[0].date == "2017-11-04" and len(report["same_written"]) == 1


def test_the_review_judges_every_pair_in_batches():
    from src.engine import written_review as wr
    from src.engine.written_review import PairVerdict, Verdicts, review_page

    notes = [_note(k, f"Tom Vale was held {k + 10} days in Port Elm jail.") for k in range(1, 16)]  # fmt: skip
    out = [Written(f"Tom Vale was held {k + 10} days in the Port Elm jail.", [k], f"S{k % 5}", 1) for k in range(1, 15)]  # fmt: skip
    twice = "Tom Vale was held 11 days in the Port Elm jail."
    out.append(Written(twice, [15], "Later", 2))
    calls = []

    def ask(msgs, model, max_tokens):
        assert model is Verdicts
        parts = msgs[1]["content"].split("Pair ")[1:]
        calls.append(len(parts))
        return Verdicts(pairs=[
            PairVerdict(pair=k, starts_with="Tom Vale was",
                        verdict="same" if part.count(twice) == 2 else "different")
            for k, part in enumerate(parts, 1)
        ])  # fmt: skip

    report = {"calls": 0}
    res = review_page(out, notes, "Port Elm", "place", ask, report, "review_final")
    assert len(calls) > 1 and max(calls) <= wr.BATCH  # more pairs than one call takes
    assert report["review_final"]["dropped"] == 1
    assert [w.text for w in res if not w.dropped].count(twice) == 1


def test_a_fact_told_twice_with_a_detail_each_is_told_once():
    from src.engine.written_checks import SentenceChecks
    from src.engine.written_fold import Fold, Folds, fold_pairs, foldable, keeps_both

    notes = [
        _note(1, "On Feb. 5, 2018, Tom Vale was fired as head of the Port Elm court.", "2018-02-06"),
        _note(2, "Ann Roe signed the order removing Tom Vale on Feb. 3, 2018.", "2018-02-07"),
    ]  # fmt: skip
    a = Written("On February 5, 2018, Tom Vale was fired as head of the court.", [1], "Court", 1)  # fmt: skip
    b = Written("Ann Roe signed the order removing Vale as head of the court.", [2], "Later", 2)  # fmt: skip
    good = "On February 5, 2018, Tom Vale was fired as head of the court, by an order Ann Roe signed."
    assert keeps_both(good, a.text, b.text)
    assert not keeps_both(
        "Tom Vale was fired as head of the court.", a.text, b.text
    )  # lost the date and Roe
    # a speaker or a group's members are never folded (round 38)
    assert not foldable(a, Written("A spokesman said Vale was fired.", [2], "S"))
    assert not foldable(a, Written("Judges, including Vale, were fired.", [2], "S"))

    def ask(msgs, model, max_tokens):
        if model is Folds:
            assert (
                "A: On February 5" in msgs[1]["content"]
                and "Feb. 3, 2018" in msgs[1]["content"]
            )
            return Folds(folds=[Fold(pair=1, text=good)])
        return SentenceChecks.model_validate({"checks": [
            {"sentence": 1, "starts_with": good, "problem": "none", "verdict": "supported"}]})  # fmt: skip

    report = {"calls": 0}
    out = fold_pairs([a, b], [(a, b)], notes, "Port Elm", ask, report)
    kept = [w for w in out if not w.dropped]
    assert [w.text for w in kept] == [good] and kept[0].notes == [1, 2]
    assert kept[0].section == "Court" and report["folded"] == [[good, a.text, b.text]]

    # a fold the check refuses leaves both sentences as they were
    a2, b2 = (Written(w.text, w.notes, w.section, w.para) for w in (a, b))

    def refuse(msgs, model, max_tokens):
        if model is Folds:
            return Folds(folds=[Fold(pair=1, text=good)])
        return SentenceChecks.model_validate({"checks": [
            {"sentence": 1, "starts_with": good, "problem": "Roe did not sign it", "verdict": "wrong"}]})  # fmt: skip

    out = fold_pairs([a2, b2], [(a2, b2)], notes, "Port Elm", refuse, {"calls": 0})
    assert [w.text for w in out if not w.dropped] == [a.text, b.text]


def test_notes_about_one_event_are_planned_into_one_part():
    from src.engine.written_events import PairEvents, cross_pairs, gather_events
    from src.engine.written_plan import Plan, PlanPart

    notes = [
        _note(1, "Tom Vale was fired from the Port Elm court after a dispute, a report said.", "2018-03-10"),
        _note(2, "On Feb. 5, 2018, Tom Vale was fired as head of the Port Elm court.", "2018-02-06"),
        _note(3, "Tom Vale ruled on the Port Elm court budget in 2017.", "2017-05-01"),
        _note(4, "The court building in Port Elm was painted.", "2016-01-01"),
        _note(5, "Tom Vale is a judge who was fired from the Port Elm court in 2018.", "2018-04-01"),
    ]  # fmt: skip
    by_n = {n.n: n for n in notes}
    plan = Plan(identity="a court", why_it_matters="", parts=[
        PlanPart(heading="Budget", covers="", notes=[3, 1]),
        PlanPart(heading="Firing", covers="", notes=[2]),
        PlanPart(heading="Building", covers="", notes=[4]),
        PlanPart(heading="Who", covers="", notes=[5]),
    ], left_out=[], opening_notes=[])  # fmt: skip
    pairs = cross_pairs(plan.parts, by_n, "Port Elm")
    assert (1, 2) in pairs and all({a, b} != {1, 3} for a, b in pairs)  # same part
    asked = []

    def ask(msgs, model, max_tokens):
        assert model is PairEvents
        parts = msgs[1]["content"].split("Pair ")[1:]
        asked.extend(parts)
        return PairEvents.model_validate({"pairs": [
            {"pair": k, "verdict": "one" if p.count("fired") >= 2 else "two",
             "home": "A" if 'A (section "Firing")' in p else "B",
             "a_mainly": "yes", "b_mainly": "yes"}
            for k, p in enumerate(parts, 1)
        ]})  # fmt: skip

    report = {"calls": 0}
    gather_events(plan, by_n, "Port Elm", "court", ask, report)
    assert not any("is a judge" in p for p in asked)  # who someone is never moves
    # the note joins the section the model names, though it is the later one
    assert [(p.heading, p.notes) for p in plan.parts] == [
        ("Budget", [3]), ("Firing", [2, 1]), ("Building", [4]), ("Who", [5])
    ]  # fmt: skip
    assert report["events_moved"] == [[notes[0].text, notes[1].text, "Firing"]]

    # a note mainly about something else stays where it is
    plan.parts = [PlanPart(heading="Budget", covers="", notes=[3, 1]),
                  PlanPart(heading="Firing", covers="", notes=[2])]  # fmt: skip

    def passing(msgs, model, max_tokens):
        parts = msgs[1]["content"].split("Pair ")[1:]
        return PairEvents.model_validate({"pairs": [
            {"pair": k, "verdict": "one", "home": "B", "a_mainly": "no", "b_mainly": "yes"}
            for k, _ in enumerate(parts, 1)
        ]})  # fmt: skip

    gather_events(plan, by_n, "Port Elm", "court", passing, {"calls": 0})
    assert [p.notes for p in plan.parts] == [[3, 1], [2]]


def test_an_opening_note_with_more_than_identity_is_also_filed_in_the_body():
    from src.engine.written_plan import Plan, PlanPart, file_opening_details

    notes = [
        _note(1, "Tom Vale was a Port Elm soldier from Cedar Falls and a father of two, including a son named Abe.", "2010-01-01"),
        _note(2, "Tom Vale died in a firefight near Cedar Falls in July 2002.", "2010-01-02"),
        _note(3, "Tom Vale was a Port Elm soldier.", "2010-01-03"),
    ]  # fmt: skip
    by_n = {n.n: n for n in notes}
    notes += [_note(k, f"Tom Vale's trial in Port Elm heard witness {k}.", "2011-01-01") for k in range(4, 9)]  # fmt: skip
    by_n = {n.n: n for n in notes}
    plan = Plan(identity="a Port Elm soldier", why_it_matters="", parts=[
        PlanPart(heading="Death", covers="", notes=[2]),
        PlanPart(heading="Trial", covers="", notes=[4, 5, 6, 7, 8]),
    ], left_out=[], opening_notes=[1, 3])  # fmt: skip
    assert file_opening_details(plan, by_n, "Tom Vale") == [1]  # 3 says only who he was
    # the part closest per note, not the biggest; first in it
    assert plan.parts[0].notes == [1, 2]


def test_an_opener_on_the_article_month_takes_the_month_its_note_gives():
    from src.engine.written_when_who import retime_openers, suspect

    notes = [
        _note(1, "In May 2018, lawyer Tom Vale told the Port Elm judge the tapes were lost.", "2018-07-20"),
        _note(2, "Ann Roe testified about the March 2002 raid in Port Elm.", "2010-10-05"),
    ]  # fmt: skip
    by_n = {n.n: n for n in notes}
    told = Written("In July 2018, lawyer Tom Vale told the judge the tapes were lost.", [1], "S")  # fmt: skip
    said = Written("In October 2010, Ann Roe testified in Port Elm.", [2], "S")
    report = {}
    retime_openers([told, said], by_n, report, "Port Elm")
    assert (
        told.text == "In May 2018, lawyer Tom Vale told the judge the tapes were lost."
    )
    # the note's March 2002 is the raid's, not the testimony's
    assert said.text == "In October 2010, Ann Roe testified in Port Elm."
    assert len(report["openers_retimed"]) == 1
    # a month the notes give only another of that year is asked about
    other = Written("In July 2018, Tom Vale lost the tapes.", [1], "S")
    assert "July 2018" in suspect(other, by_n)


def test_a_sentence_dated_by_a_later_retelling_takes_the_earlier_report():
    from src.engine.written_retell import RetoldChecks, candidates, redate_retold

    notes = [
        _note(1, "Tom Vale struck down Ann Roe's $1,000 fine for contempt at the Port Elm court.", "2018-01-15"),
        _note(2, "Tom Vale on Wednesday struck down the $1,000 contempt fine of Ann Roe.", "2017-11-22"),
        _note(3, "Ann Roe was held in contempt at the Port Elm court.", "2017-11-01"),
    ]  # fmt: skip
    by_n = {n.n: n for n in notes}
    w = Written(
        "In January 2018, Tom Vale struck down Ann Roe's $1,000 fine.", [1], "S"
    )
    dated = Written("In November 2017, Ann Roe was held in contempt.", [3], "S")
    found = candidates([w, dated], by_n, "Port Elm")
    assert [(x[0].text, [e.n for e in x[3]]) for x in found] == [(w.text, [2])]

    def ask(msgs, model, max_tokens):
        assert model is RetoldChecks and "E1 [2017-11-22]" in msgs[1]["content"]
        return RetoldChecks.model_validate(
            {
                "checks": [
                    {
                        "sentence": 1,
                        "same": "yes",
                        "earlier": 1,
                        "when": "November 2017",
                    }
                ]
            }
        )

    report = {}
    redate_retold([w, dated], by_n, "Port Elm", ask, report)
    # the earlier note's text gives no month, so its report's month bounds it
    assert w.text == "By November 2017, Tom Vale struck down Ann Roe's $1,000 fine."
    assert w.notes == [1, 2] and len(report["retold"]) == 1


def test_a_condensed_note_is_read_through_its_own_notes():
    from src.engine.written_page import Note
    from src.engine.written_when_who import retime_openers

    a = _note(11, "In July 2017, prosecutors told the Port Elm judge a secret court was needed.", "2017-08-02")  # fmt: skip
    b = _note(12, "In March 2016 the Port Elm site was closed, and in May 2018 it was razed.", "2018-06-01")  # fmt: skip
    joined = Note(n=1, key="j", text=a.text + " " + b.text, articles=["a11", "a12"],
                  date="2017-08-02", keys=["k11", "k12"])  # fmt: skip
    w = Written("In August 2017, prosecutors told the judge a secret court was needed.", [1], "S")  # fmt: skip
    report = {}
    retime_openers([w], {1: joined}, report, "Port Elm")  # three months: no change
    assert w.text.startswith("In August 2017")
    retime_openers([w], {1: joined}, report, "Port Elm", {"k11": a, "k12": b})
    assert (
        w.text == "In July 2017, prosecutors told the judge a secret court was needed."
    )


def test_writers_see_which_notes_give_no_date_of_their_own():
    from src.engine.written_page import UNDATED, section_block
    from src.engine.written_revise import _note_line

    undated = _note(1, "Tom Vale struck down the fine.", "2018-01-15")
    dated = _note(2, "In November 2017, Tom Vale struck down the fine.", "2018-01-15")
    block = section_block([undated, dated])
    assert (
        block.splitlines()[0]
        == f"N1 (2018-01-15; {UNDATED}): Tom Vale struck down the fine."
    )
    assert block.splitlines()[1].startswith("N2 (2018-01-15): ")
    assert UNDATED in _note_line(undated) and UNDATED not in _note_line(dated)


def test_round_46_date_questions_answers_and_retell_choices():
    from src.engine.written_retell import RetoldChecks, candidates, redate_retold
    from src.engine.written_when_who import (
        WhenWhoChecks,
        suspect,
        usable_fix,
        when_who,
    )

    by_n = {
        1: _note(1, "Lawyer Tom Vale accused the Port Elm agency of lying to courts.", "2014-04-02"),
        2: _note(2, "Ann Roe testified that she questioned Sam Lee at the Port Elm camp in 2003.", "2010-05-04"),
        3: _note(3, "In September, Tom Vale filed a motion.", "2014-04-02"),
    }  # fmt: skip
    # a month no note names and no note's article is from
    loose = Written("In February 2014, Tom Vale accused the agency of lying.", [1], "S")
    assert "February 2014" in suspect(loose, by_n)
    assert suspect(Written("In April 2014, Tom Vale accused it.", [1], "S"), by_n) == ""
    # a month the note names without a year is the date pass's
    assert suspect(Written("In September 2013, Tom Vale filed a motion.", [3], "S"), by_n) == ""  # fmt: skip
    # a year the note puts on what was said, not on the saying
    inner = Written("In 2003, Ann Roe testified that she questioned Sam Lee.", [2], "S")
    assert "2003" in suspect(inner, by_n)
    moved = "Ann Roe testified that she questioned Sam Lee in 2003."
    assert usable_fix(inner.text, moved, [by_n[2].text], "Port Elm", [])

    def ask(msgs, model, max_tokens):
        return WhenWhoChecks.model_validate({"checks": [
            {"sentence": 1, "starts_with": "In February 2014", "when": "right", "who": "right"},
            {"sentence": 2, "starts_with": "In 2003, Ann", "when": "wrong", "who": "right",
             "problem": "the year is the questioning's", "fixed": moved},
        ]})  # fmt: skip

    report: dict = {}
    when_who([loose, inner], list(by_n.values()), "Port Elm", ask, report)
    assert inner.text == moved
    assert [a["done"] for a in report["when_who_answers"]] == ["left", "mended"]

    # the retell step looks from each note of the opener's month and the
    # model names which earlier note reports the act
    notes = [
        _note(4, "Tom Vale struck down Ann Roe's $1,000 fine for contempt at the Port Elm court.", "2018-01-15"),
        _note(5, "Tom Vale suspended Ann Roe's 21-day contempt sentence at the Port Elm court.", "2017-11-03"),
        _note(6, "Tom Vale on Wednesday struck down the $1,000 contempt fine of Ann Roe.", "2017-11-22"),
    ]  # fmt: skip
    rb = {n.n: n for n in notes}
    w = Written(
        "In January 2018, Tom Vale struck down Ann Roe's $1,000 fine.", [4], "S"
    )
    ((_, _, own, earlier),) = candidates([w], rb, "Port Elm")
    # the suspended sentence shares only names with what the sentence tells
    assert [n.n for n in own] == [4] and [e.n for e in earlier] == [6]
    said = Written("In January 2018, Tom Vale said he struck down the $1,000 fine.", [4], "S")  # fmt: skip
    assert candidates([said], rb, "Port Elm") == []  # a statement keeps its date

    def retell(msgs, model, max_tokens):
        assert model is RetoldChecks and "E1 [2017-11-22]" in msgs[1]["content"]
        return RetoldChecks.model_validate({"checks": [
            {"sentence": 1, "same": "yes", "earlier": 1, "when": "November 2017"}]})  # fmt: skip

    report = {}
    redate_retold([w], rb, "Port Elm", retell, report)
    assert w.text == "By November 2017, Tom Vale struck down Ann Roe's $1,000 fine."
    assert w.notes == [4, 6] and report["retell_answers"][0]["picked"] == 1


def test_a_mend_keeps_a_year_beside_a_month_of_its_own_article():
    from src.engine.written_when_who import WhenWhoChecks, when_who

    notes = [_note(1, "The 2003 Port Elm bombing killed 12. Charges were sworn out on June 20.", "2017-06-29")]  # fmt: skip
    w = Written("Charges were sworn out on June 20, 2017, over the Port Elm bombing.", [1], "S")  # fmt: skip

    def ask(msgs, model, max_tokens):
        return WhenWhoChecks.model_validate({"checks": [
            {"sentence": 1, "starts_with": "Charges were sworn", "when": "wrong", "who": "right",
             "fixed": "Charges were sworn out on June 20, over the Port Elm bombing."}]})  # fmt: skip

    report: dict = {}
    when_who([w], notes, "Port Elm", ask, report)
    assert w.text.startswith("Charges were sworn out on June 20, 2017,")


def test_a_bare_month_takes_the_year_of_its_own_months_article():
    from src.engine.written_dates import own_dates

    notes = [
        _note(1, "Ann Roe taught art classes at Port Elm.", "2018-02-10"),
        _note(2, "Ann Roe said guards wore orange at Port Elm.", "2018-06-20"),
        _note(3, "Ann Roe spoke at Port Elm.", "2016-04-02"),
    ]
    page = [
        Written("In June, Ann Roe said guards wore orange.", [1, 2], "S"),
        Written("In September, Ann Roe spoke again.", [1, 3], "S"),
    ]
    own_dates(page, notes)
    assert page[0].text == "In June 2018, Ann Roe said guards wore orange."
    assert page[1].text == "In September, Ann Roe spoke again."  # two years: left


def test_a_name_starting_with_a_month_is_not_dated():
    from src.engine.written_dates import own_dates

    notes = [
        _note(1, "Ann Roe spoke on Mayra Bell's behalf at Port Elm.", "2016-05-02")
    ]
    page = [Written("Ann Roe spoke on Mayra Bell's behalf.", [1], "S")]
    own_dates(page, notes)
    assert page[0].text == "Ann Roe spoke on Mayra Bell's behalf."


def test_an_undated_act_on_its_article_month_says_by_then():
    from src.engine.written_bound import Bounds, bound_openers, candidates

    notes = [
        _note(1, "Tom Vale struck down Ann Roe's $1,000 fine at Port Elm.", "2018-01-15"),
        _note(2, "Tom Vale said on Tuesday the court would sit.", "2018-01-17"),
        _note(3, "Ann Roe was moved to Camp Six at Port Elm.", "2018-01-20"),
    ]  # fmt: skip
    by_n = {n.n: n for n in notes}
    fine = Written("In January 2018, Tom Vale struck down Ann Roe's $1,000 fine.", [1], "S")  # fmt: skip
    said = Written("In January 2018, Tom Vale said the court would sit.", [2], "S")
    moved = Written("In January 2018, Ann Roe was moved to Camp Six.", [3], "S")
    assert [c[0] for c in candidates([fine, said, moved], by_n)] == [fine, moved]

    def ask(msgs, model, max_tokens):
        assert model is Bounds
        return Bounds.model_validate({"checks": [
            {"sentence": 1, "told": "older",
             "text": "By January 2018, Tom Vale had struck down Ann Roe's $1,000 fine."},
            {"sentence": 2, "told": "older", "text": "By January 2018, Ann Roe was moved to Camp 6."},
        ]})  # fmt: skip

    report: dict = {}
    bound_openers([fine, said, moved], by_n, "Port Elm", ask, report)
    assert (
        fine.text == "By January 2018, Tom Vale had struck down Ann Roe's $1,000 fine."
    )
    assert moved.text.startswith("In January 2018")  # "Six" lost: refused
    assert [a["done"] for a in report["bound_answers"]] == ["bounded", "refused"]
