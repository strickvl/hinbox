"""Moving misfiled facts out of a long section before it is split
(src/engine/section_sort.py).

Scenario: in the 8 Nov local run Qwen filed Vance Spath's 2018 trial
suspension at the end of "Background"; the split then cut Background into
sub-headings and the suspension sat under "Post-military career".
"""

import json
from typing import Any, Dict, List

from src.engine.section_sort import SORTED_KEY, sort_long_sections, sort_section
from src.engine.section_split import SectionSplit, SubSection, _sentences

STARTER = ["Background", "Role", "Cases and decisions"]
NOTES = {
    "Background": "who they are and their career before the events covered",
    "Role": "the posts and appointments they hold or held",
    "Cases and decisions": "rulings, orders, actions and decisions they took",
}
TEXT = (
    "Vance Spath is an Air Force judge.^[a0]\n\n"
    "## Background\n\n"
    "Spath served as an Air Force prosecutor.^[a1]\n\n"
    "Spath suspended the USS Cole trial in February 2018.^[a2]\n\n"
    "He became an immigration judge in 2018.^[a3]\n\n"
    "Spath issued writs for Eliades and Spears.^[a4]\n\n"
    "## Role\n\nHe was chief judge of the Air Force trial judiciary.^[a5]\n\n"
    "## Cases and decisions\n\nSpath held Gen. Baker in contempt.^[a6]\n\n"
    "### Contempt of Baker\n\nBaker was confined to a trailer.^[a7]\n"
)
REPLY = SectionSplit(
    subsections=[
        SubSection(heading="Background", paragraphs=[1, 3]),
        SubSection(heading="Cases and decisions", paragraphs=[2, 4]),
    ]
)


def _asker(reply: Any, seen: List[List[Dict[str, str]]]):
    def ask(msgs):
        seen.append(msgs)
        if isinstance(reply, Exception):
            raise reply
        return reply

    return ask


def _sort(reply: Any, text: str = TEXT):
    seen: List[List[Dict[str, str]]] = []
    out, report = sort_section(
        text, "Background", "person", "Vance Spath", STARTER, NOTES,
        _asker(reply, seen),
    )  # fmt: skip
    return out, report, seen


def test_court_items_leave_background_word_for_word():
    out, report, seen = _sort(REPLY)
    assert report["status"] == "done" and report["moved"] == {"Cases and decisions": 2}
    assert _sentences(out) == _sentences(TEXT)
    background = out[out.index("## Background") : out.index("## Role")]
    assert "prosecutor" in background and "immigration judge" in background
    assert "suspended" not in background and "writs" not in background
    # Into Cases' own text, after its last citation, before its sub-heading
    assert out.index("Baker in contempt") < out.index("suspended the USS Cole")
    assert out.index("issued writs") < out.index("### Contempt of Baker")
    prompt = seen[0][0]["content"]
    assert 'The section "Background" has grown long' in prompt
    assert "- Cases and decisions: rulings, orders" in prompt
    assert "[2] Spath suspended the USS Cole trial" in seen[0][1]["content"]


def test_pieces_left_out_or_under_unknown_headings_stay():
    reply = SectionSplit(
        subsections=[
            SubSection(heading="Trial matters", paragraphs=[2]),
            SubSection(heading="Role", paragraphs=[3]),
        ]
    )
    out, report, _ = _sort(reply)
    background = out[out.index("## Background") : out.index("## Role")]
    assert "suspended" in background and "writs" in background
    assert "immigration judge" not in background
    assert report["moved"] == {"Role": 1}


def test_nothing_moved_or_a_failed_call_keeps_the_text():
    keep = SectionSplit(subsections=[SubSection(heading="Background", paragraphs=[1])])
    out, report, _ = _sort(keep)
    assert out == TEXT and report["status"] == "kept"
    out, report, _ = _sort(RuntimeError("down"))
    assert out == TEXT and report["status"] == "failed"


def test_a_missing_starter_heading_is_created_for_a_moved_piece():
    text = TEXT.replace(
        "## Role\n\nHe was chief judge of the Air Force trial judiciary.^[a5]\n\n", ""
    )
    reply = SectionSplit(subsections=[SubSection(heading="Role", paragraphs=[3])])
    out, report, seen = _sort(reply, text)
    assert report["status"] == "done"
    assert "## Role\n\nHe became an immigration judge" in out
    assert out.index("## Background") < out.index("## Role") < out.index("## Cases")
    assert (
        "Role: the posts and appointments they hold or held (not in the profile yet)"
        in seen[0][0]["content"]
    )


def _profile(**extra: Any) -> Dict[str, Any]:
    return {"text": TEXT, "reorganised": "done", **extra}


def _run(profile: Dict[str, Any], ask, min_chars: int = 150):
    return sort_long_sections(
        profile, "person", "Vance Spath", None, "guantanamo", ask, min_chars
    )


def test_only_reorganised_profiles_and_long_own_text_and_growth(monkeypatch):
    seen: List[List[Dict[str, str]]] = []
    ask = _asker(REPLY, seen)
    flat, reports = _run({"text": TEXT, "reorganised": "refused"}, ask)
    assert reports == [] and flat["text"] == TEXT
    out, reports = _run(_profile(), ask)
    assert [r["section"] for r in reports] == ["Background"]  # others are short
    assert (
        reports[0]["status"] == "done"
        and "suspended" not in out["text"].split("## Role")[0]
    )
    stored = json.loads(out[SORTED_KEY])
    assert stored["Background"] < 200
    # Asked again only once Background's own text grows 25% past that size
    again, reports = _run(out, ask)
    assert reports == [] and len(seen) == 1
    monkeypatch.setenv("HINBOX_SECTION_SORT", "off")
    assert _run(_profile(), ask)[1] == []


def test_update_profile_sorts_before_it_splits(monkeypatch):
    from src.engine import profiles, section_split

    calls: List[str] = []
    monkeypatch.setattr(section_split, "SPLIT_CHARS", 100)

    def fake_sort(profile, *a, **k):
        calls.append("sort")
        return profile, []

    def fake_split(profile, *a, **k):
        calls.append("split")
        return profile, []

    monkeypatch.setattr("src.engine.section_sort.sort_long_sections", fake_sort)
    monkeypatch.setattr(section_split, "split_profile", fake_split)
    monkeypatch.setenv("HINBOX_SECTION_REORGANISE", "off")
    monkeypatch.setenv("HINBOX_SECTION_SPLIT", "people")
    profiles._split_sections(_profile(), "people", "Vance Spath", "cloud", "m")
    assert calls == ["sort", "split"]
