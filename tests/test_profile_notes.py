"""Notes about "the profile" never reach a profile (``src/engine/profile_notes.py``)."""

import datetime as dt
from typing import List

from src.engine.dispute_check import DisputeVerdict, DisputeVerdicts
from src.engine.edit_guards import guard_edits
from src.engine.later_updates import as_update
from src.engine.profile_edits import ProfileEdit
from src.engine.profile_notes import (
    is_empty_note,
    is_profile_note,
    plain_side,
    reworded,
)

COUNT = "As of January 2016, the camp held 93 captives.^[old]"
RULING = "Judge Hale ordered the two officials to testify.^[old]"
PROFILE = f"## Camp\n\n{COUNT}\n\n## Court\n\n{RULING}\n"
DATES = {"old": "2016-01-14", "a9": "2016-03-08"}
MORE = "The article states there are 104 captives, conflicting with the profile's count of 93 captives.^[a9]"
EVENT = (
    "The article states that the hearing was a formal arraignment of the five "
    "men, whereas the profile describes it as a first appearance.^[a9]"
)


def _add(text: str, anchor: str = "END") -> ProfileEdit:
    return ProfileEdit(op="add", anchor=anchor, text=text)


def _asker(verdicts: List[DisputeVerdict], calls: List[str]):
    def ask(instructions: str, items: str) -> DisputeVerdicts:
        calls.append(items)
        return DisputeVerdicts(verdicts=verdicts)

    return ask


def test_a_note_names_the_profile_as_a_source():
    assert is_profile_note(MORE) and is_profile_note(EVENT)
    assert is_profile_note("This contradicts the existing profile's account.")
    assert is_profile_note(
        "Sources disagree on his age; the profile states he is 41, while a 2014 "
        "report lists him as 38."
    )
    # A document called a profile (a review board's file on a captive)
    assert not is_profile_note(
        "The profile also acknowledged that he answered the debriefers' questions."
    )
    assert not is_profile_note(
        "The board released the profile, which called him a low-level fighter."
    )
    assert not is_profile_note("The ruling raised the profile of the case.")
    assert not is_profile_note("He was a high-profile captive at the camp.")
    assert not is_profile_note("Sources disagree on the year of his capture.")


def test_plain_side_is_the_articles_half_alone():
    assert plain_side(EVENT) == (
        "The hearing was a formal arraignment of the five men.^[a9]"
    )
    tail = (
        "However, the article reports that the two men were questioned by a "
        "judge and held, contradicting the profile's implication of immediate "
        "release.^[a9]"
    )
    assert plain_side(tail) == "The two men were questioned by a judge and held.^[a9]"
    no_subject = (
        "The captive ended his fast on June 1, a spokesman said, conflicting "
        "with the profile's statement that no captive had ended a fast.^[a9]"
    )
    assert plain_side(no_subject) == (
        "The captive ended his fast on June 1, a spokesman said.^[a9]"
    )
    timing = (
        "The article reports that Judge Hale ordered both men to testify the "
        "week of July 23, rather than the unspecified timing in the profile^[a9]"
    )
    assert plain_side(timing) == (
        "Judge Hale ordered both men to testify the week of July 23.^[a9]"
    )
    dated = (
        "A report from April 19, 2013, states that the general toured the "
        "new wing with two senators, whereas the profile says he never went.^[a9]"
    )
    assert plain_side(dated) == (
        "An April 19, 2013 report states that the general toured the new wing "
        "with two senators.^[a9]"
    )
    # The article's half is too short to stand alone
    short = (
        "Sources disagree on the date of his capture: the profile states "
        "April 2003, while the article states Sept. 11, 2002.^[a9]"
    )
    assert plain_side(short) is None


def test_a_count_is_dated_by_the_article():
    import datetime as dt

    assert plain_side(MORE, dt.date(2016, 3, 8)) == (
        "As of March 2016, there are 104 captives.^[a9]"
    )
    assert plain_side(MORE) == "There are 104 captives.^[a9]"


def test_reworded_never_names_the_profile():
    assert reworded(MORE).endswith(
        "conflicting with another report's count of 93 captives.^[a9]"
    )
    assert reworded("The profile states 2003; the article states 2002.^[a9]") == (
        "Another report states 2003; the article states 2002.^[a9]"
    )


def test_a_newer_count_becomes_a_dated_entry_without_a_model():
    edits, report = guard_edits(
        PROFILE, [_add(MORE, anchor=COUNT)], "2016-03-08", DATES
    )
    assert edits[0].text == "As of March 2016, there are 104 captives.^[a9]"
    assert report["profile_notes"][0][1] == "plain"
    # A hunger-strike count is one the later-update rule already knows
    strike = (
        "The article reports that in March 2016, 84 captives were on hunger "
        "strike, conflicting with the profile's figure of 92 hunger strikers.^[a9]"
    )
    old = "In January 2016, 92 captives were hunger strikers.^[old]"
    edits, report = guard_edits(
        f"{old}\n", [_add(strike, anchor=old)], "2016-03-08", DATES
    )
    assert edits[0].text == "In March 2016, 84 captives were on hunger strike.^[a9]"
    assert report["later_updates"]


def test_without_a_model_the_articles_half_is_kept():
    edits, report = guard_edits(PROFILE, [_add(EVENT)], "2016-03-08", DATES)
    assert edits[0].text == "The hearing was a formal arraignment of the five men.^[a9]"
    assert report["profile_notes"][0][1] == "plain"
    # A half the profile already has adds nothing
    again = (
        "The article states that Judge Hale ordered the two officials to "
        "testify, contradicting the profile's timing.^[a9]"
    )
    edits, _ = guard_edits(PROFILE, [_add(again)], "2016-03-08", DATES)
    assert edits == []
    # What the profile is said to lack is the article's fact
    lacks = "The profile omits that Colonel Reed led the camp in 2015.^[a9]"
    edits, report = guard_edits(PROFILE, [_add(lacks)], "2016-03-08", DATES)
    assert edits[0].text == "Colonel Reed led the camp in 2015.^[a9]"
    assert report["profile_notes"][0][1] == "plain"


def test_the_model_decides_what_a_note_is():
    calls: List[str] = []
    fact = "The hearing was a formal arraignment of the five men"
    verdict = DisputeVerdict(item=1, verdict="different", sentence=fact)
    edits, report = guard_edits(
        PROFILE, [_add(EVENT)], "2016-03-08", DATES, ask=_asker([verdict], calls)
    )
    assert len(calls) == 1 and "the editor's sentence: The article states" in calls[0]
    assert edits[0].text == fact + ".^[a9]"
    assert report["dispute_check"][0][1] == "different"

    same = DisputeVerdict(item=1, verdict="same")
    edits, _ = guard_edits(
        PROFILE, [_add(EVENT)], "2016-03-08", DATES, ask=_asker([same], [])
    )
    assert edits == []


def test_a_real_conflict_is_kept_without_the_profile_as_a_source():
    conflict = DisputeVerdict(item=1, verdict="conflict")
    edits, report = guard_edits(
        PROFILE, [_add(EVENT)], "2016-03-08", DATES, ask=_asker([conflict], [])
    )
    assert edits[0].text == reworded(EVENT)
    assert "whereas another report describes it" in edits[0].text
    assert report["profile_notes"][0][1] == "conflict"


def test_a_failed_check_keeps_the_articles_half():
    def ask(instructions: str, items: str) -> DisputeVerdicts:
        raise RuntimeError("timeout")

    edits, report = guard_edits(PROFILE, [_add(EVENT)], "2016-03-08", DATES, ask=ask)
    assert edits[0].text == "The hearing was a formal arraignment of the five men.^[a9]"
    assert report["profile_notes"][0][1] == "plain"
    assert report["dispute_check"][0][1] == "unchecked"


def test_a_model_sentence_that_still_names_the_profile_is_dropped():
    verdict = DisputeVerdict(
        item=1, verdict="different",
        sentence="The hearing was an arraignment, unlike what the profile says",
    )  # fmt: skip
    edits, _ = guard_edits(
        PROFILE, [_add(EVENT)], "2016-03-08", DATES, ask=_asker([verdict], [])
    )
    assert edits == []


def test_a_half_that_is_not_a_sentence_keeps_a_source_in_front():
    count = (
        "The article reports 41 captives at the camp, conflicting with the "
        "profile's count of 93.^[a9]"
    )
    assert plain_side(count) == "A report notes 41 captives at the camp.^[a9]"
    assert plain_side(count, dt.date(2016, 3, 8)) == (
        "A March 2016 report notes 41 captives at the camp.^[a9]"
    )
    # With "that", or with "was" in it, the half is a sentence
    moved = (
        "The article states the hearing was moved to June, contradicting the "
        "profile's date.^[a9]"
    )
    assert plain_side(moved) == "The hearing was moved to June.^[a9]"


def test_the_half_ends_where_the_comparison_starts():
    figure = (
        "The article states that 41 of the last 60 captives were cleared, a "
        "figure that differs from the profile's count of 43.^[a9]"
    )
    assert plain_side(figure) == "41 of the last 60 captives were cleared.^[a9]"
    two = (
        "The article reports that Judge Hale suspended the fine and that the "
        "ruling was not vacated, contradicting the profile's claim.^[a9]"
    )
    assert plain_side(two) == (
        "Judge Hale suspended the fine and the ruling was not vacated.^[a9]"
    )
    both = (
        "The article states Hale was appointed in May 2015, which is consistent "
        "with the profile's first sentence but conflicts with the profile's "
        "statement that he was appointed in June 2015.^[a9]"
    )
    assert plain_side(both) == "Hale was appointed in May 2015.^[a9]"
    # "which conflicts with" is not "sources disagree": the half stands alone
    edits, report = guard_edits(PROFILE, [_add(both)])
    assert "profile" not in edits[0].text and report["profile_notes"][0][1] == "plain"


def test_a_half_that_leans_on_the_other_half_is_not_cut_loose():
    leans = "The article states that it took place in May, unlike the profile.^[a9]"
    assert plain_side(leans) is None
    edits, report = guard_edits(PROFILE, [_add(leans)])
    assert edits[0].text == reworded(leans)
    assert report["profile_notes"][0][1] == "reworded"


def test_a_note_with_no_fact_from_the_article_is_dropped():
    silent = (
        "The article does not mention the judge's age, whereas the profile "
        "gives two ages.^[a9]"
    )
    thinking = (
        "The article states the camp holds 93 captives, which conflicts with "
        "the profile's figure of 93 (no conflict, so no edit needed).^[a9]"
    )
    assert is_empty_note(silent) and is_empty_note(thinking)
    assert not is_empty_note(MORE)
    for note in (silent, thinking):
        edits, report = guard_edits(PROFILE, [_add(note)], ask=_asker([], []))
        assert edits == [] and report["profile_notes"][0][1] == "empty"


def test_a_date_the_half_already_gives_is_not_said_twice():
    when = dt.date(2017, 1, 9)
    side = "a January 2017 transfer of four captives left 55 at the camp"
    assert as_update("says", side, None, when, "^[a9]") == (
        "A January 2017 transfer of four captives left 55 at the camp.^[a9]"
    )
    early = "in early 2017 he had lost 30 pounds"
    assert as_update("says", early, None, when, "^[a9]") == (
        "In early 2017 he had lost 30 pounds.^[a9]"
    )
    noted = "Hale claimed the power to excuse the lawyers in January 2017"
    assert as_update("notes", noted, None, when, "^[a9]") == (
        "A report notes Hale claimed the power to excuse the lawyers in January "
        "2017.^[a9]"
    )


def test_the_same_note_twice_in_a_reply_is_rewritten_once():
    again = MORE.replace("count of 93 captives", "count of 92 captives")
    edits, report = guard_edits(PROFILE, [_add(f"{MORE} {again} {MORE}")])
    assert edits[0].text == "There are 104 captives.^[a9]"
    assert [how for _, how, _ in report["profile_notes"]] == ["plain", "repeat"]
    assert report["repeats"] == [MORE]
    # ... and the same later state is written once
    edits, report = guard_edits(
        PROFILE, [_add(f"{MORE} {again}", anchor=COUNT)], "2016-03-08", DATES
    )
    assert edits[0].text.count("104 captives") == 1


def test_what_is_left_of_the_comparison_is_cut_from_the_half():
    silent = (
        "The article states that the base has about 5,500 residents, but does "
        "not specify the number of guards as 1,500, conflicting with the "
        "profile's guard count of 1,700.^[a9]"
    )
    assert plain_side(silent) == "The base has about 5,500 residents.^[a9]"
    year = (
        "The article states that the five men arrived in 2006, not 2008, and "
        "were held apart, contradicting the profile's claim.^[a9]"
    )
    assert plain_side(year) == "The five men arrived in 2006, and were held apart.^[a9]"
    # A half about an earlier year is not "as of" the article
    past = (
        "The article states that Colonel Reed was due to retire in May 2014 "
        "and has an extension, unlike the profile.^[a9]"
    )
    assert plain_side(past, dt.date(2016, 7, 1)) == (
        "Colonel Reed was due to retire in May 2014 and has an extension.^[a9]"
    )
    far = (
        "The article states that the camp holds 79 captives, with 26 cleared "
        "for release with security assurances that satisfy the defense "
        "secretary, conflicting with the 31 cleared for release with security "
        "assurances that satisfy the defense secretary cited in the profile.^[a9]"
    )
    assert plain_side(far).startswith("The camp holds 79 captives, with 26 cleared")
    assert "31" not in plain_side(far)


def test_a_half_whose_subject_was_in_the_other_half_is_not_cut_loose():
    he = (
        "The profile states that Hale upheld the fine, while the new article "
        "reports that he suspended the fine pending review.^[a9]"
    )
    assert plain_side(he) is None
    edits, _ = guard_edits(PROFILE, [_add(he)])
    assert edits[0].text == reworded(he)


def test_a_remark_on_the_sentence_before_is_dropped():
    remark = (
        "This contradicts the profile's claim that Hale took the case in July "
        "2014, as Reed was already presiding in May.^[a9]"
    )
    assert is_empty_note(remark)
    edits, report = guard_edits(PROFILE, [_add(remark)])
    assert edits == [] and report["profile_notes"][0][1] == "empty"


def test_as_of_is_not_put_before_a_side_with_its_own_day_or_source():
    when = dt.date(2016, 8, 2)
    day = "the board cleared Reed for release on May 26, 2016, the 35th so far"
    assert as_update("says", day, None, when, "^[a9]").startswith("The board cleared")
    weekday = "17 captives were on hunger strike on Wednesday, October 9, 2013"
    assert as_update("says", weekday, None, when, "^[a9]").startswith("17 captives")
    source = "a later report by a wire service states that no one was injured"
    assert as_update("says", source, None, when, "^[a9]").startswith("A later report")
    today = "there are 265 captives at the camp today"
    assert as_update("says", today, None, when, "^[a9]") == (
        "As of August 2016, there are 265 captives at the camp.^[a9]"
    )
    # A date the side is about still gets one
    planned = "the trial was scheduled to begin on December 4, 2016"
    assert as_update("says", planned, None, when, "^[a9]").startswith("As of August")
    then = "there were 80 captives at that time"
    assert as_update("says", then, None, when, "^[a9]") == (
        "As of August 2016, there were 80 captives.^[a9]"
    )
    bracket = "on Friday evening (March 15, 2013) the spokesman counted 14 men"
    assert as_update("says", bracket, None, when, "^[a9]").startswith("On Friday")


def test_any_mention_of_our_profile_is_a_note():
    for note in (
        "The camp closed in May, a date not specified in the profile.^[a9]",
        "The profile states that Colonel Reed led the camp in 2015.^[a9]",
        "The article states that 44 men were fed by tube, correcting the "
        "profile's figure of 45.^[a9]",
        "A 2016 report puts his age at 36, noting a discrepancy with the "
        "profile's age of 38.^[a9]",
    ):
        assert is_profile_note(note), note
    # A document someone wrote about the entity is not our profile
    for document in (
        "The profile also assessed him as a low-level cook.^[a9]",
        "The profile also states that he answered every question.^[a9]",
        "Ahead of the hearing, the profile described him as 38 years old.^[a9]",
        "A 2015 intelligence profile prepared for the board retreated from the "
        "earlier profile that cast him as a guard.^[a9]",
    ):
        assert not is_profile_note(document), document


def test_the_model_thinking_aloud_is_dropped():
    for debris in (
        "Wait, the profile sentence *is* the article's sentence.",
        "The sentence with 154 is already in the profile?",
        "The profile has 116 and 122.",
        "But the profile *also* has 116 and 122.",
    ):
        assert is_profile_note(debris) and is_empty_note(debris), debris
        edits, report = guard_edits(PROFILE, [_add(debris)])
        assert edits == [] and report["profile_notes"][0][1] == "empty"
    # What the profile is said to state is kept when an article is cited
    states = "The profile states that Colonel Reed led the camp in 2015."
    assert is_empty_note(states, cited=False) and not is_empty_note(states)
    edits, _ = guard_edits(PROFILE, [_add(states)])
    assert edits == []
    edits, _ = guard_edits(PROFILE, [_add(f"{states[:-1]}.^[a9]")])
    assert edits[0].text == "Colonel Reed led the camp in 2015.^[a9]"
    firm = (
        "The profile wrongly calls Harbor Air an engineering firm; Reed was an "
        "engineer employed by the airline.^[a9]"
    )
    assert plain_side(firm) == "Reed was an engineer employed by the airline.^[a9]"


def test_the_profile_in_the_last_clause_or_in_brackets_is_cut():
    cases = {
        "The article reports that Colonel Reed left the camp on May 18, a date "
        "not specified in the profile.^[a9]": "Colonel Reed left the camp on May 18.^[a9]",
        "The article confirms Reed was moved to the base in April 2007, "
        "resolving the previous ambiguity in the profile about the year.^[a9]": (
            "Reed was moved to the base in April 2007.^[a9]"
        ),
        "The article reports that Judge Hale paused the case in January 2015 "
        "(implied by the profile's earlier mention of the order), not March "
        "2017.^[a9]": "Judge Hale paused the case in January 2015.^[a9]",
        'The article states that the men were held "without lawyers for four '
        "years,\" which supports the profile's mention of four years.^[a9]": (
            'The men were held "without lawyers for four years".^[a9]'
        ),
        "The article states that of the 116 captives here, just one was "
        "convicted, conflicting with the profile's count of 10 charged.^[a9]": (
            "Of the 116 captives, just one was convicted.^[a9]"
        ),
        # "have been closed" happened; it is not how things stand
        "The article states that the court struck down the first tribunals, "
        "and that they have been closed once, which occurred before the 2008 "
        "ruling described in the profile.^[a9]": (
            "The court struck down the first tribunals, and they have been "
            "closed once.^[a9]"
        ),
    }
    for note, plain in cases.items():
        assert plain_side(note, dt.date(2016, 3, 8)) == plain, note


def test_the_articles_report_after_the_profiles_claim():
    claim = (
        "The profile's claim that Hale retired in June 2018 conflicts with the "
        "article's report that Hale announced his retirement on August 27, "
        "2018^[old].^[a9]"
    )
    # ... with the article's own citation, not the one copied from the profile
    assert plain_side(claim) == "Hale announced his retirement on August 27, 2018.^[a9]"
    copied = (
        "The article states that an official suspended the sentence after two "
        "days, whereas the profile states that a judge overturned it^[old].^[a9]"
    )
    assert (
        plain_side(copied) == "An official suspended the sentence after two days.^[a9]"
    )
    # A bare count has no sentence of its own: the dispute stays, reworded
    count = (
        "The profile states the camp held 265 men, conflicting with the "
        "article's count of 181 men held there.^[a9]"
    )
    assert plain_side(count) is None and not is_empty_note(count)
    edits, _ = guard_edits(PROFILE, [_add(count)])
    assert edits[0].text.startswith("Another report states the camp held 265 men")


def test_a_side_with_the_article_as_a_verbs_subject_keeps_a_source():
    second = (
        "The profile mentions a July event, but the article clarifies the "
        "handover happened in April 2017, not July, and identifies Reed as the "
        "departing commander, whereas the profile does not specify the month.^[a9]"
    )
    assert plain_side(second) == (
        "A report states that the handover happened in April 2017, and "
        "identifies Reed as the departing commander.^[a9]"
    )
    also = (
        "However, the article states that the players were put in a closet, "
        "but also notes that in one wing each man still has his own player, "
        "contradicting the profile's implication that all were removed.^[a9]"
    )
    assert plain_side(also, dt.date(2016, 3, 8)) == (
        "A March 2016 report states that the players were put in a closet, but "
        "also notes that in one wing each man still has his own player.^[a9]"
    )
    narrator = (
        "The new article refers to her as Dana Moss and states she joined the "
        "team in 2011, contradicting the profile's claim that she joined in "
        "2009^[a9]."
    )
    assert plain_side(narrator) == (
        "A report refers to her as Dana Moss and states she joined the team in "
        "2011.^[a9]"
    )


def test_a_doubled_profile_is_reworded_once():
    doubled = (
        "The article reports the charge, whereas the profile notes the profile "
        "also notes this fact.^[a9]"
    )
    assert reworded(doubled) == (
        "The article reports the charge, whereas another report notes this fact.^[a9]"
    )


def test_a_profile_that_only_lacks_something_disputes_nothing():
    lacks = (
        "The sources disagree on the cost of the camp; while the profile does "
        "not specify a cost, the article states that the government put the "
        "cost at $800,000 per captive a year.^[a9]"
    )
    edits, report = guard_edits(PROFILE, [_add(lacks)])
    assert edits[0].text == (
        "The government put the cost at $800,000 per captive a year.^[a9]"
    )
    assert report["profile_notes"][0][1] == "plain"
    # With a side of its own for the profile it is a dispute; an invented
    # one here (the profile gives no cost), and a bare value is not news
    both = (
        "The sources disagree on the cost of the camp; the profile states "
        "$700,000 a year, while the article states $800,000 a year for each "
        "captive.^[a9]"
    )
    edits, report = guard_edits(PROFILE, [_add(both)])
    assert edits == [] and report["disputes"]


def test_an_invented_dispute_keeps_the_articles_news():
    """No old sentence gives another value, so nothing is disputed, but the
    article's half is still a new fact."""
    old = "A health team found no record of contamination at the camp.^[old]"
    note = (
        "Sources disagree on the status of the inquiry: while the profile "
        "states a health team found no record of contamination, the article "
        "reports that a second team is carrying out a full survey of the "
        "camp.^[a9]"
    )
    edits, report = guard_edits(f"{old}\n", [_add(note)], "2016-03-08", DATES)
    assert edits[0].text == (
        "As of March 2016, a second team is carrying out a full survey of the "
        "camp.^[a9]"
    )
    assert report["profile_notes"][0][1] == "plain" and not report["disputes"]
    # A half that leans on the other half is dropped with the dispute
    leans = (
        "The sources disagree on the nurse's status: the profile states he "
        "went back to work, while the article reports he is facing a board^[a9]."
    )
    edits, report = guard_edits(f"{old}\n", [_add(leans)], "2016-03-08", DATES)
    assert edits == [] and report["disputes"]


def test_a_day_gets_its_year_and_a_weekday_its_date():
    when = dt.date(2017, 12, 14)
    signed = "the statement on Dec. 12 referenced a bill for $124 million"
    assert as_update("says", signed, None, when, "^[a9]") == (
        "The statement on Dec. 12, 2017, referenced a bill for $124 million.^[a9]"
    )
    # A planned day is not when the value held
    planned = "the trial is set to begin on Dec. 12"
    assert as_update("says", planned, None, when, "^[a9]").startswith("As of December")
    friday = (
        "The article reports that as of Friday (January 11, 2013), the base "
        "was holding 166 captives, which conflicts with the profile's "
        "statement that this number was recorded in April.^[a9]"
    )
    assert plain_side(friday, dt.date(2013, 1, 12)) == (
        "As of January 11, 2013, the base was holding 166 captives.^[a9]"
    )


def test_a_relative_day_keeps_the_dated_source():
    when = dt.date(2013, 3, 20)
    monday = "as of Monday, 44 captives were fed by tube"
    assert as_update("says", monday, None, when, "^[a9]") == (
        "A March 2013 report states that as of Monday, 44 captives were fed "
        "by tube.^[a9]"
    )
    office = (
        "The article reports that the office counted 19 deaths on Monday, "
        "contradicting the profile's count of 12.^[a9]"
    )
    assert plain_side(office, when) == (
        "A March 2013 report states that the office counted 19 deaths on Monday.^[a9]"
    )
    # A weekday with its full date says when by itself
    full = "17 captives were on hunger strike on Wednesday, October 9, 2013"
    assert as_update("says", full, None, when, "^[a9]").startswith("17 captives")
    now = "a second team is currently carrying out a survey"
    assert as_update("says", now, None, when, "^[a9]") == (
        "As of March 2013, a second team is carrying out a survey.^[a9]"
    )


def test_two_halves_that_say_the_same_are_dropped():
    same = (
        "Sources disagree on the count; the profile states there are 172 "
        "captives at the camp, while the article states there are 172 captives "
        "at the camp.^[a9]"
    )
    assert is_empty_note(same)
    edits, report = guard_edits(PROFILE, [_add(same)], "2016-03-08", DATES)
    assert edits == [] and report["profile_notes"][0][1] == "empty"
