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
