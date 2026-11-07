"""The model check on "sources disagree" sentences (``dispute_check.py``) and
the "surname only" note for the match check."""

from typing import List

from src.engine.dispute_check import (
    DisputeVerdict,
    DisputeVerdicts,
    items_prompt,
    replacement,
)
from src.engine.edit_guards import _NARRATOR, guard_edits
from src.engine.merge_candidates import surname_note
from src.engine.profile_edits import ProfileEdit

PROFILE = (
    "### Transfer\n\n"
    "The prisoner was sent home to a rehabilitation center in May 2018.^[dec]\n"
)
ANCHOR = "The prisoner was sent home to a rehabilitation center in May 2018.^[dec]"
DISPUTE = (
    "A February 2018 article said he was slated to go to the rehabilitation center on February 20, 2018; "
    "this differs from the reported May transfer date.^[feb]"
)
DATES = {"dec": "2018-12-21", "feb": "2018-02-11"}


def _asker(verdicts: List[DisputeVerdict], calls: List[str]):
    def ask(instructions: str, items: str) -> DisputeVerdicts:
        calls.append(items)
        return DisputeVerdicts(verdicts=verdicts)

    return ask


def _guard(verdict: DisputeVerdict, calls: List[str]):
    edit = ProfileEdit(op="add", anchor=ANCHOR, text=DISPUTE)
    return guard_edits(
        PROFILE, [edit], "2018-02-11", DATES, ask=_asker([verdict], calls)
    )


def test_newer_dispute_wordings_are_recognised():
    assert _NARRATOR.search("this differs from the reported May transfer date")
    assert _NARRATOR.search("these sources give different timelines for it")
    assert _NARRATOR.search("the sources therefore give different timing")
    # News about people disagreeing is not the profile comparing sources
    assert not _NARRATOR.search("the judge disagreed and moved toward trial")
    assert not _NARRATOR.search("a ban which contradicted American values")


def test_a_later_state_is_rewritten_as_the_articles_dated_fact():
    calls: List[str] = []
    fact = "In February 2018, the prisoner was slated to go home on February 20."
    edits, report = _guard(
        DisputeVerdict(item=1, verdict="later", sentence=fact), calls
    )
    assert len(calls) == 1
    # The model sees the old sentence with its source's date, and the article's
    assert "source dated December 21, 2018" in calls[0]
    assert "dated February 11, 2018" in calls[0]
    assert edits[0].text == fact + "^[feb]"
    assert report["dispute_check"][0][1] == "later"


def test_a_real_conflict_is_kept_as_written():
    edits, _ = _guard(DisputeVerdict(item=1, verdict="conflict"), [])
    assert edits[0].text == DISPUTE


def test_a_restatement_is_dropped():
    edits, report = _guard(DisputeVerdict(item=1, verdict="same"), [])
    assert edits == []
    assert report["dispute_check"][0][1] == "same"


def test_a_failed_check_keeps_the_sentence():
    def ask(instructions: str, items: str) -> DisputeVerdicts:
        raise RuntimeError("timeout")

    edit = ProfileEdit(op="add", anchor=ANCHOR, text=DISPUTE)
    edits, _ = guard_edits(PROFILE, [edit], "2018-02-11", DATES, ask=ask)
    assert edits[0].text == DISPUTE


def test_no_call_without_a_dispute_sentence():
    calls: List[str] = []
    edit = ProfileEdit(
        op="add", anchor=ANCHOR, text="He later took part in an art program.^[feb]"
    )
    guard_edits(PROFILE, [edit], "2018-02-11", DATES, ask=_asker([], calls))
    assert calls == []


def test_a_replacement_that_still_compares_sources_is_dropped():
    verdict = DisputeVerdict(
        item=1, verdict="different", sentence="The stay instead came on Nov. 7."
    )
    assert replacement(verdict, DISPUTE, "^[feb]") is None
    verdict.sentence = "The appeals court stayed the case on Nov. 7, 2018"
    assert (
        replacement(verdict, DISPUTE, "^[feb]")
        == "The appeals court stayed the case on Nov. 7, 2018.^[feb]"
    )


def test_items_prompt_strips_citations():
    text = items_prompt(
        [{"sentence": DISPUTE, "olds": [(ANCHOR, "x")], "article_date": None}]
    )
    assert "^[" not in text and "unknown date" in text


def test_surname_note_only_for_a_unique_surname():
    people = {"Marco Rubio": {}, "Rubio": {}, "Keith Parrella": {}}
    note = surname_note("people", "Rubio", "Marco Rubio", {"people": people})
    assert "only person in the database" in note
    people["Jennifer Parrella"] = {}
    assert not surname_note("people", "Parrella", "Keith Parrella", {"people": people})
    assert not surname_note("people", "Marco", "Marco Rubio", {"people": people})
    assert not surname_note("organizations", "Navy", "U.S. Navy", {"people": people})
