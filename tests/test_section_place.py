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
    ("engineer", "Background"),
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
    assert "1. Copy its first three words." in seen[0][0]["content"]
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


def test_pieces_stay_out_of_background_and_next_to_what_they_answer():
    new = SECTIONED.replace(
        "planning the 9/11 attacks.^[a2]",
        "planning the 9/11 attacks.^[a2] He is an engineer.^[a5]",
    ).replace(
        "1964.^[a0]",
        "1964.^[a0] However, a later report says he was charged in 2009.^[a5] "
        "As of 2025, he was charged with more counts.^[a5]",
    )
    out, report, seen = _place(new)
    assert len(seen) == 1 and report["pieces"] == 3
    assert out == new and report["status"] == "kept"


def test_a_split_keeps_its_headings_and_files_pieces_again():
    from src.engine.section_split import SectionSplit, SubSection, split_long_sections

    paras = [
        "He was charged in 2008 with planning the 9/11 attacks.^[a1]",
        "At his 2008 arraignment he asked to be put to death.^[a1]",
        "Prosecutors began plea talks in 2017.^[a4]",
        "Pretrial hearings began in 2012 and dealt with CIA torture.^[a2]",
        "A plea deal was signed in July 2024.^[a5]",
        "He gave a hospital an award.^[a6]",
    ]
    text = "### Legal\n\n" + "\n\n".join(paras) + "\n\n### Health\n\nHe is ill.^[a3]\n"
    split = SectionSplit(
        subsections=[
            SubSection(heading="Charges", covers="charges", paragraphs=[1, 2, 3]),
            SubSection(heading="Hearings", covers="hearings", paragraphs=[4]),
            SubSection(heading="Plea talks", covers="plea talks", paragraphs=[5, 6]),
        ]
    )
    seen: List[Any] = []

    def place(msgs):
        seen.append(msgs)
        rules = [("charged", "Charges"), ("arraignment", "Hearings"),
                 ("plea", "Plea talks"), ("Pretrial", "Hearings"),
                 ("award", "None of these")]  # fmt: skip
        found = re.findall(r"^\[(\d+)\] (.*)$", msgs[1]["content"], re.M)
        return Placements(
            placements=[
                Placement(sentence=int(n), about="",
                          heading=next(h for k, h in rules if k in t))
                for n, t in found
            ]
        )  # fmt: skip

    out, reports = split_long_sections(
        text, "person", "KSM", lambda m: split, min_chars=100, place=place
    )
    assert reports[0]["status"] == "split"
    assert reports[0]["refile"] == {"status": "done", "refiled": 3, "stayed": 1}
    assert "- Charges: charges" in seen[0][0]["content"]
    assert 'stays at the top of "Legal"' in seen[0][0]["content"]
    assert _sentences(out) == _sentences(text)
    legal = out.split("### Health")[0]
    assert legal.index("hospital an award") < legal.index("#### Charges")
    plea = legal.split("#### Plea talks")[1]
    assert "plea talks in 2017" in plea and "plea deal" in plea
    assert "arraignment" in legal.split("#### Hearings")[1].split("####")[0]


def test_a_stayed_piece_never_takes_the_uncited_text_above_the_section():
    from src.engine.opening_sort import _claims
    from src.engine.section_split import SectionSplit, SubSection, split_long_sections

    paras = [
        "He was charged in 2008 with planning the 9/11 attacks.^[a1]",
        "At his 2008 arraignment he asked to be put to death.^[a1]",
        "Prosecutors began plea talks in 2017.^[a4]",
        "Pretrial hearings began in 2012 and dealt with CIA torture.^[a2]",
        "A plea deal was signed in July 2024.^[a5]",
        "He gave a hospital an award.^[a6]",
    ]
    # Grounding judges this uncited line with piece 1's citation (a1); the
    # stayed piece 6 going first would hand it to a6
    text = "### Background\n\nHe grew up in Kuwait.\n\n### Legal\n\n" + (
        "\n\n".join(paras) + "\n\n### Health\n\nHe is ill.^[a3]\n"
    )
    split = SectionSplit(
        subsections=[
            SubSection(heading="Charges", covers="charges", paragraphs=[1, 2, 3]),
            SubSection(heading="Plea talks", covers="plea talks", paragraphs=[4, 5, 6]),
        ]
    )

    def place(msgs):
        found = re.findall(r"^\[(\d+)\] (.*)$", msgs[1]["content"], re.M)
        pick = lambda t: (  # noqa: E731
            "None of these"
            if "award" in t
            else "Plea talks"
            if "plea" in t
            else "Charges"
        )
        return Placements(
            placements=[
                Placement(sentence=int(n), about="", heading=pick(t)) for n, t in found
            ]
        )

    out, reports = split_long_sections(
        text, "person", "KSM", lambda m: split, min_chars=100, place=place
    )
    assert reports[0]["status"] == "split"
    assert reports[0]["refile"]["status"] == "kept"
    assert reports[0]["refile"]["reason"] == "citation claims changed"
    assert _claims(out) == _claims(text) and _sentences(out) == _sentences(text)
    legal = out.split("### Legal")[1].split("### Health")[0]
    assert legal.index("#### Charges") < legal.index("charged in 2008")


def test_a_refiling_that_leaves_one_heading_keeps_the_split():
    from src.engine.section_place import refile_split

    plan = [("A", [1, 2]), ("B", [3])]
    one = lambda msgs: Placements(  # noqa: E731
        placements=[Placement(sentence=n, about="", heading="A") for n in (1, 2, 3)]
    )
    out, stay, report = refile_split(
        plan, {}, ["x", "y", "z"], "P", "person", "K", one, True
    )
    assert out == plan and stay == [] and report["status"] == "kept"
    out, _, report = refile_split(
        plan, {}, ["x", "y", "z"], "P", "person", "K", _by_keyword([], fail=True), True
    )
    assert out == plan and report["status"] == "failed"


def test_an_answer_whose_copied_words_are_another_sentences_is_dropped():
    from src.engine.section_place import read_reply

    reply = Placements(
        placements=[
            Placement(
                sentence=1, starts_with="He was born", about="", heading="Background"
            ),
            Placement(
                sentence=2, starts_with="He was born", about="", heading="Background"
            ),
            Placement(
                sentence=3, starts_with="", about="", heading="Health and treatment"
            ),
        ]
    )
    sentences = [
        "He was born in Kuwait.^[a0]",
        "“He was charged.”^[a1]",
        "He is ill.^[a2]",
    ]
    assert read_reply(reply, 3, STARTER, sentences) == [
        "Background", None, "Health and treatment",
    ]  # fmt: skip


def test_the_model_must_fill_the_alignment_and_covers_fields():
    from src.engine.section_split import SubSection

    assert "starts_with" in Placement.model_json_schema()["required"]
    assert "covers" in SubSection.model_json_schema()["required"]
    assert SubSection(heading="A", paragraphs=[1]).covers == ""  # code needn't


def test_split_refiling_is_for_local_models_unless_turned_on(monkeypatch):
    from src.engine.section_place import refile_enabled

    assert refile_enabled(local=True) and not refile_enabled(local=False)
    monkeypatch.setenv("HINBOX_SPLIT_REFILE", "on")
    assert refile_enabled(local=False)
    monkeypatch.setenv("HINBOX_SPLIT_REFILE", "off")
    assert not refile_enabled(local=True)
