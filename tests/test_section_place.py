"""Filing profile pieces with the placement-only prompt
(src/engine/section_place.py): in the reorganise, and for each update's new
pieces once a profile has its starter headings.

Scenario: KSM's profile has its detainee headings. An update adds "He was
charged again in 2025" at the end of Background (where the edit model
anchored it); the placement call files it under the legal heading, and code
moves it there word for word.
"""

import re
from typing import Any, Callable, Dict, List

from src.engine import section_place, section_reorganise
from src.engine.section_place import (
    Placement,
    Placements,
    new_pieces,
    place_new,
    place_profile,
)
from src.engine.section_reorganise import reorganise_profile, reorganise_text
from src.engine.section_split import _sentences

STARTER = [
    "Background",
    "Capture and CIA custody",
    "Detention at Guantánamo",
    "Military commission or legal case",
    "Health and treatment",
]
RULES = [
    ("charged", "Military commission or legal case"),
    ("plea", "Military commission or legal case"),
    ("captured", "Capture and CIA custody"),
    ("waterboarded", "Capture and CIA custody"),
    ("moved to Guantánamo", "Detention at Guantánamo"),
    ("heart", "Health and treatment"),
]


def _by_keyword(seen: List[Any], fail: bool = False) -> Callable:
    """A placement asker that files each numbered sentence by a keyword."""

    def ask(msgs: List[Dict[str, str]]) -> Placements:
        seen.append(msgs)
        if fail:
            raise RuntimeError("down")
        out = []
        for n, sentence in re.findall(r"^\[(\d+)\] (.*)$", msgs[1]["content"], re.M):
            heading = next((h for k, h in RULES if k in sentence), "Background")
            out.append(Placement(sentence=int(n), about="", heading=heading))
        return Placements(placements=out)

    return ask


FLAT = (
    "Khalid Sheik Mohammed is a Pakistani detainee.^[a0]\n\n"
    "### Background\n\nHe was born in Kuwait in 1964.^[a0]\n\n"
    "### Legal Proceedings\n\n"
    "He was captured in Rawalpindi in 2003.^[a1] "
    "The CIA waterboarded him 183 times.^[a1]\n\n"
    "He was moved to Guantánamo in 2006.^[a2]\n\n"
    "He was charged in 2008 with planning the 9/11 attacks.^[a2]\n\n"
    "A plea deal was signed in July 2024.^[a3]\n\n"
    "He has a heart condition.^[a4]\n"
)


def test_the_reorganise_files_pieces_with_the_placement_prompt():
    seen: List[Any] = []
    out, report = reorganise_text(
        FLAT, "person", "KSM", STARTER, ask=None, min_chars=100,
        place=_by_keyword(seen),
    )  # fmt: skip
    assert report["status"] == "done" and report["assign"] == "classify"
    assert len(seen) == 1  # 7 pieces: one call of up to 8
    assert "1. Say in a few words" in seen[0][0]["content"]
    assert _sentences(out) == _sentences(FLAT)
    capture = out.split("### Capture and CIA custody")[1].split("###")[0]
    assert "Rawalpindi" in capture and "waterboarded" in capture
    assert "Legal Proceedings" not in out
    assert (
        out.index("### Background") < out.index("### Capture") < out.index("### Health")
    )


def test_the_pipeline_reorganise_uses_it_unless_told_otherwise(monkeypatch):
    monkeypatch.setattr(section_reorganise, "REORGANISE_CHARS", 100)
    seen: List[Any] = []
    monkeypatch.setattr(
        section_place, "make_asker", lambda model, local, step="": _by_keyword(seen)
    )
    args = ("person", "KSM", "detainee", "guantanamo", "m", True)
    out, report = reorganise_profile({"text": FLAT}, *args)
    assert out["reorganised"] == "done" and len(seen) == 1
    monkeypatch.setenv("HINBOX_REORGANISE_ASSIGN", "sections")
    called: List[str] = []

    def old_prompt(model, local, step=""):
        called.append(step)
        return lambda msgs: (_ for _ in ()).throw(RuntimeError("down"))

    monkeypatch.setattr(section_reorganise, "make_asker", old_prompt)
    _, report = reorganise_profile({"text": FLAT}, *args)
    assert called and report["status"] == "failed" and len(seen) == 1


SECTIONED = (
    "Khalid Sheik Mohammed is a Pakistani detainee.^[a0]\n\n"
    "### Background\n\nHe was born in Kuwait in 1964.^[a0]\n\n"
    "### Capture and CIA custody\n\nHe was captured in Rawalpindi in 2003.^[a1]\n\n"
    "### Military commission or legal case\n\n"
    "He was charged in 2008 with planning the 9/11 attacks.^[a2]\n\n"
    "#### Plea deal\n\nA plea deal was signed in July 2024.^[a3]\n"
)


def _place(new: str, ask=None):
    seen: List[Any] = []
    out, report = place_new(
        SECTIONED, new, "person", "KSM", STARTER, {}, ask or _by_keyword(seen)
    )
    return out, report, seen


def test_a_new_piece_under_the_wrong_heading_moves_word_for_word():
    new = SECTIONED.replace(
        "Kuwait in 1964.^[a0]",
        "Kuwait in 1964.^[a0] He was charged again in 2025.^[a5]",
    ).replace(
        "Rawalpindi in 2003.^[a1]",
        "Rawalpindi in 2003.^[a1]\n\nThe CIA waterboarded him 183 times.^[a5]",
    )
    assert [new[s:e] for _, s, e in new_pieces(SECTIONED, new)] == [
        "He was charged again in 2025.^[a5]",
        "The CIA waterboarded him 183 times.^[a5]",
    ]
    out, report, seen = _place(new)
    assert report["status"] == "done" and len(seen) == 1
    assert report["moved"] == {"Background -> Military commission or legal case": 1}
    assert _sentences(out) == _sentences(new)
    assert "Kuwait in 1964.^[a0]\n\n### Capture" in out  # no stray space left
    legal = out.split("### Military commission or legal case")[1]
    # At the end of the legal heading's own text, before its sub-heading
    assert legal.index("charged again") < legal.index("#### Plea deal")
    assert "waterboarded" in out.split("### Capture")[1].split("###")[0]  # stayed


def test_opening_pieces_stay_when_they_say_who_the_person_is():
    new = SECTIONED.replace(
        "a Pakistani detainee.^[a0]",
        "a Pakistani detainee.^[a0] He is an engineer.^[a5] "
        "He was captured with Hawsawi.^[a5]",
    )
    out, report, _ = _place(new)
    assert report["moved"] == {"(opening) -> Capture and CIA custody": 1}
    assert out.startswith(
        "Khalid Sheik Mohammed is a Pakistani detainee.^[a0] He is an engineer.^[a5]\n"
    )


def test_a_heading_left_empty_by_the_move_goes_and_a_missing_one_is_made():
    new = SECTIONED.replace(
        "### Military commission",
        "### Detention at Guantánamo\n\nHe was charged again in 2025.^[a5]\n\n"
        "### Military commission",
    ).replace("1964.^[a0]", "1964.^[a0]\n\nHe has a heart condition.^[a5]")
    out, report, _ = _place(new)
    assert report["status"] == "done"
    assert "### Detention" not in out  # it held only the misfiled piece
    assert "charged again" in out.split("### Military commission or legal case")[1]
    # Health didn't exist: made after the last starter heading the profile has
    assert out.rstrip().endswith(
        "### Health and treatment\n\nHe has a heart condition.^[a5]"
    )


def test_a_failed_call_or_nothing_new_keeps_the_update():
    new = SECTIONED.replace("1964.^[a0]", "1964.^[a0] He was charged.^[a5]")
    out, report, _ = _place(new, _by_keyword([], fail=True))
    assert out == new and report["status"] == "failed"
    out, report, _ = _place(SECTIONED)
    assert out == SECTIONED and report["status"] == "skipped"
    same = SECTIONED.replace("1964.^[a0]", "1964.^[a0] He was a student.^[a5]")
    out, report, _ = _place(same)
    assert out == same and report["status"] == "kept"


def test_only_profiles_reorganised_before_the_update(monkeypatch):
    new = SECTIONED.replace("1964.^[a0]", "1964.^[a0] He was charged.^[a5]")
    seen: List[Any] = []
    args = (SECTIONED, "person", "KSM", "detainee", "guantanamo", "m", True)
    profile, report = place_profile({"text": new}, *args, ask=_by_keyword(seen))
    assert report is None and not seen
    profile, report = place_profile(
        {"text": new, "reorganised": "done"}, *args, ask=_by_keyword(seen)
    )
    assert (
        report["status"] == "done"
        and "charged." in profile["text"].split("### Military")[1]
    )
    monkeypatch.setenv("HINBOX_SECTION_PLACE", "off")
    assert place_profile({"text": new, "reorganised": "done"}, *args)[1] is None


def test_update_profile_files_new_pieces_of_a_reorganised_profile(monkeypatch):
    from src.engine import profiles

    monkeypatch.setenv("HINBOX_SECTION_REORGANISE", "people")
    monkeypatch.setenv("HINBOX_SECTION_SPLIT", "off")
    seen: List[Any] = []
    monkeypatch.setattr(
        section_place, "make_asker", lambda model, local, step="": _by_keyword(seen)
    )
    new = SECTIONED.replace("1964.^[a0]", "1964.^[a0] He was charged.^[a5]")
    monkeypatch.setitem(
        profiles._UPDATERS, "edit",
        lambda *a, **k: ({"text": new, "sources": ["a0", "a5"], "reorganised": "done"}, []),
    )  # fmt: skip
    old = {"text": SECTIONED, "sources": ["a0"], "reorganised": "done"}
    out, _, _ = profiles.update_profile(
        "person", "KSM", old, profiles.VersionedProfile(), "article", "a5",
        model_type="cloud", model="m", approach="edit", kind="detainee",
    )  # fmt: skip
    assert len(seen) == 1 and "charged." in out["text"].split("### Military")[1]
