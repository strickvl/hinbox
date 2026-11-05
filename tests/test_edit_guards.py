"""Code checks on the edit-list updater's edits (``src/engine/edit_guards.py``)."""

from src.engine.edit_guards import guard_edits, is_invented_dispute, is_repeat
from src.engine.profile_edits import ProfileEdit

PROFILE = (
    "# Camp 6\n\n## Background\n\n"
    "Camp 6 opened in 2006 as a penitentiary-style building.^[a1] "
    "The camp holds about 120 detainees in communal blocks.^[a2]\n"
)


def _add(text: str, anchor: str = "END") -> ProfileEdit:
    return ProfileEdit(op="add", anchor=anchor, text=text)


def test_a_restated_sentence_is_a_repeat():
    known = ["Camp 6 opened in 2006 as a penitentiary-style building.^[a1]"]
    assert is_repeat("Camp 6 opened in 2006 as a penitentiary building.^[a9]", known)
    # A new number is new detail, not a repeat
    assert not is_repeat(
        "Camp 6 opened in 2006 as a 200-cell penitentiary-style building.^[a9]", known
    )
    # The opposite ruling isn't a repeat, though its long words are the same
    ruling = ["The judge ruled that Nashiri could attend the hearing.^[a1]"]
    opposite = "The judge ruled that Nashiri could not attend the hearing.^[a9]"
    assert not is_repeat(opposite, ruling)
    # A changed year is new, even if the profile gives that year elsewhere
    hearing = ["In 2013 the military judge scheduled Nashiri's hearing.^[a1]"]
    moved = "In 2014 the military judge scheduled Nashiri's hearing.^[a9]"
    assert not is_repeat(moved, hearing)
    # A different person doing the same thing is new
    pohl = [
        "Nashiri's military defense lawyer consulted Judge Pohl about the hearing.^[a1]"
    ]
    spath = "Nashiri's military defense lawyer consulted Judge Spath about the hearing.^[a9]"
    assert not is_repeat(spath, pohl)
    # A changed outcome in a short word is new
    won = [
        "Nashiri's lawyers won the appeal against the military commission ruling.^[a1]"
    ]
    lost = (
        "Nashiri's lawyers lost the appeal against the military commission ruling.^[a9]"
    )
    assert not is_repeat(lost, won)
    # The model's "the article states" wrapping doesn't hide a repeat
    assert is_repeat(
        "The article states that the appeal was won by Nashiri's lawyers.^[a9]", won
    )
    # Timing words count
    before = ["The military lawyers consulted Nashiri before the hearing.^[a1]"]
    after = "The military lawyers consulted Nashiri after the hearing.^[a9]"
    assert not is_repeat(after, before)
    # Different news about the same camp shares too few words
    assert not is_repeat(
        "Guards at Camp 6 moved detainees to single cells after a raid.^[a9]", known
    )


def test_a_dispute_needs_an_old_sentence_with_a_different_value():
    known = ["Camp 6 opened in 2006 as a penitentiary-style building.^[a1]"]
    real = "Sources disagree: this report says Camp 6 opened in 2007, not 2006.^[a9]"
    invented = "Sources disagree on whether guards searched the detainees' Korans.^[a9]"
    assert not is_invented_dispute(real, known)
    assert is_invented_dispute(invented, known)
    # A planted year sharing a single topic word with the profile is kept
    khadr = ["Khadr allegedly threw a grenade in a July 2002 firefight.^[a1]"]
    planted = (
        "The new article says the incident occurred in July 2001, and the "
        "sources disagree on the year of the firefight.^[a9]"
    )
    assert not is_invented_dispute(planted, khadr)
    # A count from a later article is a later state, not a dispute
    strikers = ["In April 2013, 59 prisoners were hunger strikers.^[a1]"]
    later = "Sources disagree on the number of hunger strikers: 106 in July.^[a9]"
    assert is_invented_dispute(later, strikers)
    # The year a source was written isn't a disputed value
    age = ["In 2010 Khadr was 24 years old.^[a1]"]
    dated = "The August 14, 2009 article describes Khadr as 22; sources disagree on his age.^[a9]"
    assert is_invented_dispute(dated, age)
    # A denial the article reports is news, not the profile comparing sources
    denial = "Nashiri's lawyers contradicted reports that he confessed.^[a9]"
    assert not is_invented_dispute(denial, known)
    # A different name, or an undated count, on the same thing is kept
    judge = ["Judge Pohl presided over Nashiri's hearing.^[a1]"]
    other_judge = "Sources disagree on the presiding judge: Judge Spath presided over Nashiri's hearing.^[a9]"
    assert not is_invented_dispute(other_judge, judge)
    sailors = ["The bombing killed 17 sailors.^[a1]"]
    count = "Sources disagree: this report says the bombing killed 18 sailors.^[a9]"
    assert not is_invented_dispute(count, sailors)
    # A dispute the article reports, without "sources", is just a fact
    assert not is_invented_dispute("The lawyer disagreed with the ruling.^[a9]", known)


def test_guard_drops_repeats_and_keeps_the_new_sentence_in_the_same_edit():
    edits = [
        _add(
            "The camp holds about 120 detainees in communal blocks.^[a9] "
            "In 2013 the military moved all of them to single cells.^[a9]",
            anchor="## Background",
        ),
        ProfileEdit(op="replace", anchor="x", text="Camp 6 opened in 2006.^[a1]"),
        _add("Camp 6 opened in 2006 as a penitentiary-style building.^[a9]"),
    ]
    kept, report = guard_edits(PROFILE, edits)
    assert [e.text for e in kept] == [
        "In 2013 the military moved all of them to single cells.^[a9]",
        "Camp 6 opened in 2006.^[a1]",
    ]
    assert len(report["repeats"]) == 2 and report["disputes"] == []


def test_guard_stops_the_same_new_sentence_being_added_twice():
    sentence = "In 2013 the military moved all Camp 6 detainees to single cells.^[a9]"
    kept, report = guard_edits(
        PROFILE, [_add(sentence), _add(sentence, "## Background")]
    )
    assert [e.text for e in kept] == [sentence]
    assert report["repeats"] == [sentence]


def test_a_profile_too_long_to_edit_goes_to_the_fact_list(monkeypatch):
    from src.engine import profiles
    from src.utils.ollama_context import PromptTooLongError

    def too_long(*args, **kwargs):
        raise PromptTooLongError("too long")

    monkeypatch.setattr("src.engine.profile_edits.edit_profile", too_long)
    monkeypatch.setattr(
        profiles, "_fact_profile_update", lambda *a, **k: ({"text": "facts"}, [])
    )
    updated, _ = profiles._edit_profile_update(
        "people", "Nashiri", {"text": "old"}, "article", "a9", model="ollama/q"
    )
    assert updated == {"text": "facts"}
