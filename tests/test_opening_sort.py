"""Moving facts out of an over-long opening (src/engine/opening_sort.py).

Scenario: in a local replay KSM's opening, the text above his first heading,
grew from 562 to 11,336 characters because new facts were anchored on its
sentences. The model says which opening piece belongs under which heading;
code moves them word for word.
"""

from typing import Any, Dict, List

from src.engine import opening_sort
from src.engine.opening_sort import SORTED_KEY, sort_opening, sort_text
from src.engine.section_routing import routing_note
from src.engine.section_split import SectionSplit, SubSection, _sentences

STARTER = [
    "Background",
    "Capture and CIA custody",
    "Detention at Guantánamo",
    "Military commission or legal case",
]
OPENING = (
    "Khalid Sheik Mohammed is a Pakistani detainee.^[a0] "
    "He is accused of planning the 9/11 attacks.^[a0]\n"
    "Leaked documents give his birthday as 14 April 1965.^[a3]\n"
    "Camp 7 guards moved him in 2016.^[a4]\n"
    "Judge Pohl adjourned the 9/11 case in May 2017.^[a5]"
)
REST = (
    "### Background\n\nHe was born in Kuwait.^[a1]\n\n"
    "### Military commission or legal case\n\nHe was charged in 2008.^[a2]\n\n"
    "#### Plea negotiations\n\nPlea talks began in 2017.^[a2]\n"
)
TEXT = f"{OPENING}\n\n{REST}"
REPLY = SectionSplit(
    subsections=[
        SubSection(heading="Opening", paragraphs=[1, 2]),
        SubSection(heading="Background", paragraphs=[3]),
        SubSection(heading="Detention at Guantánamo", paragraphs=[4]),
        SubSection(heading="Military commission or legal case", paragraphs=[5]),
    ]
)


def _asker(reply: Any, seen: List[List[Dict[str, str]]]):
    def ask(msgs):
        seen.append(msgs)
        if isinstance(reply, Exception):
            raise reply
        return reply

    return ask


def _run(reply: Any, text: str = TEXT):
    seen: List[List[Dict[str, str]]] = []
    out, report = sort_text(
        text, "person", "KSM", STARTER, _asker(reply, seen), min_chars=100
    )
    return out, report, seen


def test_opening_facts_move_word_for_word_into_their_sections():
    out, report, seen = _run(REPLY)
    assert report["status"] == "done" and _sentences(out) == _sentences(TEXT)
    opening = out.split("###")[0]
    assert "Pakistani detainee" in opening and "accused of planning" in opening
    assert "birthday" not in opening and "Camp 7" not in opening
    # Into the end of the section's own text, before its sub-headings
    assert out.index("born in Kuwait") < out.index("birthday") < out.index("### Mil")
    assert out.index("charged in 2008") < out.index("Pohl adjourned")
    assert out.index("Pohl adjourned") < out.index("#### Plea negotiations")
    # A starter heading the profile lacks is made in its place in the list
    assert "### Detention at Guantánamo\n\nCamp 7 guards" in out
    assert out.index("### Background") < out.index("### Detention")
    assert out.index("### Detention") < out.index("### Military")
    assert report["moved"] == {
        "Background": 1, "Detention at Guantánamo": 1,
        "Military commission or legal case": 1,
    }  # fmt: skip
    prompt = seen[0][0]["content"]
    assert "Detention at Guantánamo (not in the profile yet)" in prompt
    assert "Plea negotiations" not in prompt  # only top-level headings


def test_piece_one_a_missing_piece_and_an_unknown_heading_stay():
    reply = SectionSplit(
        subsections=[
            SubSection(heading="Background", paragraphs=[1, 3]),
            SubSection(heading="Prison life", paragraphs=[4]),
        ]
    )
    out, report, _ = _run(reply)
    opening = out.split("###")[0]
    assert "Pakistani detainee" in opening and "Camp 7" in opening
    assert "Pohl adjourned" in opening and "birthday" not in opening
    assert report["moved"] == {"Background": 1}


def test_nothing_moved_a_short_opening_and_a_failed_call_keep_the_text():
    keep = SectionSplit(subsections=[SubSection(heading="Opening", paragraphs=[1])])
    out, report, _ = _run(keep)
    assert out == TEXT and report["status"] == "kept"
    out, report, seen = _run(REPLY, "Short.^[a0]\n\n" + REST)
    assert out.startswith("Short.") and report["status"] == "skipped" and not seen
    out, report, _ = _run(RuntimeError("down"))
    assert out == TEXT and report["status"] == "failed"


def _profile(**extra: Any) -> Dict[str, Any]:
    return {"text": TEXT, "reorganised": "done", **extra}


def test_sort_waits_until_the_opening_grows_and_needs_a_reorganised_profile(
    monkeypatch,
):
    monkeypatch.setattr(opening_sort, "SORT_CHARS", 100)
    seen: List[List[Dict[str, str]]] = []
    ask = _asker(REPLY, seen)
    out, report = sort_opening(_profile(), "people", "KSM", "detainee",
                               "guantanamo", "m", False, ask)  # fmt: skip
    assert report["status"] == "done" and "Camp 7" not in out["text"].split("###")[0]
    assert out[SORTED_KEY] == report["opening_chars"]
    # Kept once at this size: not asked again until it grows 25%
    sized = _profile(**{SORTED_KEY: len(OPENING)})
    same, report = sort_opening(sized, "people", "KSM", "detainee",
                                "guantanamo", "m", False, ask)  # fmt: skip
    assert report is None and same is sized and len(seen) == 1
    flat = {"text": TEXT, "reorganised": "refused"}
    assert (
        sort_opening(flat, "people", "KSM", None, "guantanamo", "m", False, ask)[1]
        is None
    )
    monkeypatch.setenv("HINBOX_OPENING_SORT", "off")
    assert (
        sort_opening(_profile(), "people", "KSM", None, "guantanamo", "m", False, ask)[
            1
        ]
        is None
    )


def test_the_edit_model_is_told_the_opening_is_only_who_they_are():
    note = routing_note(STARTER, {}, "KSM", TEXT)
    assert "text above the first heading only says who KSM is" in note
