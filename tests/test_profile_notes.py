"""Notes about "the profile" never reach a profile (``src/engine/profile_notes.py``)."""

import datetime as dt
from typing import List

from src.engine.dispute_check import DisputeVerdict, DisputeVerdicts
from src.engine.edit_guards import guard_edits
from src.engine.profile_edits import ProfileEdit
from src.engine.profile_notes import is_profile_note, plain_side, reworded

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
    # No half stands alone: the sentence stays, without "the profile"
    bare = "The profile omits that Colonel Reed led the camp in 2015.^[a9]"
    edits, report = guard_edits(PROFILE, [_add(bare)], "2016-03-08", DATES)
    assert (
        edits[0].text
        == "Another report omits that Colonel Reed led the camp in 2015.^[a9]"
    )
    assert report["profile_notes"][0][1] == "reworded"


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
    assert report["profile_notes"][0][1] == "unchecked"


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
