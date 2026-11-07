"""Reorganising a long flat profile under starter headings (src/engine/section_reorganise.py).

Scenario: KSM's profile grew under the headings its first draft made up
("### Background", "### Legal Proceedings"). Once it is long, the model is
shown the detainee starter headings and says which paragraph goes where;
code moves the paragraphs word for word.
"""

from typing import Any, Dict, List

from src.engine import section_reorganise
from src.engine.section_reorganise import (
    plan_sections,
    reorganise_profile,
    reorganise_text,
    starter_headings,
)
from src.engine.section_split import SectionSplit, SubSection, _sentences

STARTER = [
    "Background",
    "Capture and CIA custody",
    "Detention at Guantánamo",
    "Military commission or legal case",
    "Health and treatment",
]
OPENING = "Khalid Sheik Mohammed is a Pakistani detainee.^[a0]"
PARAS = [
    "He was born in Kuwait in 1964.^[a0]",
    "He was captured in Rawalpindi in 2003.^[a1]",
    "The CIA waterboarded him 183 times.^[a1]",
    "He was moved to Guantánamo in 2006.^[a2]",
    "He was charged in 2008 with planning the 9/11 attacks.^[a2]",
    "A plea deal was signed in July 2024.^[a3]",
    "He has a heart condition.^[a4]",
]
TEXT = (
    f"{OPENING}\n\n### Background\n\n{PARAS[0]}\n\n### Legal Proceedings\n\n"
    + "\n\n".join(PARAS[1:])
    + "\n"
)
GOOD = SectionSplit(
    subsections=[
        SubSection(heading="Military commission or legal case", paragraphs=[5, 6]),
        SubSection(heading="Background", paragraphs=[1]),
        SubSection(heading="Capture and CIA custody", paragraphs=[2, 3]),
        SubSection(heading="Detention at Guantánamo", paragraphs=[4]),
        SubSection(heading="Health and treatment", paragraphs=[7]),
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
    out, report = reorganise_text(
        text, "person", "KSM", STARTER, _asker(reply, seen), min_chars=100
    )
    return out, report, seen


def test_paragraphs_move_word_for_word_under_starter_headings_in_list_order():
    out, report, seen = _run(GOOD)
    assert report["status"] == "done"
    assert len(seen) == 1 and "Capture and CIA custody" in seen[0][0]["content"]
    assert "Legal Proceedings" not in out  # the made-up heading is gone
    assert out.startswith(OPENING + "\n\n### Background\n\n" + PARAS[0])
    heads = [line for line in out.splitlines() if line.startswith("###")]
    assert heads == [f"### {h}" for h in STARTER]  # list order, not reply order
    assert _sentences(out) == _sentences(TEXT)
    for p in PARAS:
        assert out.count(p) == 1


def test_opening_is_not_numbered_and_short_profiles_are_skipped():
    _, _, seen = _run(GOOD)
    assert OPENING not in seen[0][1]["content"]
    out, report = reorganise_text(TEXT, "person", "KSM", STARTER, _asker(GOOD, []))
    assert out == TEXT and report["status"] == "skipped"  # under 8,000 chars


def test_an_empty_starter_heading_is_left_out():
    reply = SectionSplit(
        subsections=[
            SubSection(heading="Background", paragraphs=[1, 2, 3, 4]),
            SubSection(
                heading="military commission or legal case", paragraphs=[5, 6, 7]
            ),
        ]
    )
    out, report, _ = _run(reply)
    assert report["status"] == "done"
    assert "### Military commission or legal case" in out  # starter spelling
    assert "Health and treatment" not in out


def test_a_new_heading_is_kept_when_it_shares_no_word():
    reply = SectionSplit(
        subsections=GOOD.subsections[:4]
        + [SubSection(heading="Civil lawsuits", paragraphs=[7])]
    )
    out, report, _ = _run(reply)
    assert report["status"] == "done"
    assert out.rstrip().endswith("### Civil lawsuits\n\n" + PARAS[6])


def test_a_new_heading_that_shares_a_word_refuses_the_whole_answer():
    reply = SectionSplit(
        subsections=GOOD.subsections[:3]
        + [SubSection(heading="Legal proceedings", paragraphs=[5, 6, 7])]
    )
    out, report, _ = _run(reply)
    assert out == TEXT
    assert report["status"] == "refused" and "overlaps" in report["reason"]


def test_missing_and_repeated_numbers_follow_the_split_rules():
    plan, why = plan_sections(
        SectionSplit(
            subsections=[
                SubSection(heading="Background", paragraphs=[1, 2, 2]),
                SubSection(heading="Health and treatment", paragraphs=[2, 4]),
            ]
        ),
        4,
        STARTER,
    )
    assert why == ""
    # 2 keeps its first place; missing 3 goes with its neighbour 2
    assert plan == [("Background", [1, 2, 3]), ("Health and treatment", [4])]
    plan, why = plan_sections(
        SectionSplit(subsections=[SubSection(heading="Background", paragraphs=[1])]),
        4,
        STARTER,
    )
    assert plan is None and "only 1 of 4" in why


def test_uncited_end_of_the_opening_is_numbered_with_the_rest():
    text = "KSM is a detainee. He is from Kuwait.\n\n### Background\n\n" + "\n\n".join(
        PARAS
    )
    out, report, seen = _run(GOOD, text)
    # the uncited opening joins paragraph 1, so its claim stays whole
    assert "[1] KSM is a detainee." in seen[0][1]["content"]
    assert report["status"] == "done" and _sentences(out) == _sentences(text)


def test_a_failed_call_keeps_the_profile_and_is_tried_again(monkeypatch):
    monkeypatch.setattr(section_reorganise, "REORGANISE_CHARS", 100)
    profile = {"text": TEXT, "sources": ["a0"]}
    out, report = reorganise_profile(
        profile, "person", "KSM", "detainee", "guantanamo", "m", False,
        ask=_asker(RuntimeError("no answer"), []),
    )  # fmt: skip
    assert out is profile and report["status"] == "failed"
    out, report = reorganise_profile(
        profile, "person", "KSM", "detainee", "guantanamo", "m", False,
        ask=_asker(GOOD, []),
    )  # fmt: skip
    assert out["reorganised"] == "done" and "### Capture and CIA custody" in out["text"]
    seen: List[Any] = []
    again, report = reorganise_profile(
        out, "person", "KSM", "detainee", "guantanamo", "m", False,
        ask=_asker(GOOD, seen),
    )  # fmt: skip
    assert again is out and report is None and not seen  # once per entity


def test_fact_row_profiles_are_left_alone(monkeypatch):
    from src.engine.fact_list import FACT_ROWS_KEY

    monkeypatch.setattr(section_reorganise, "REORGANISE_CHARS", 100)
    profile = {"text": TEXT, FACT_ROWS_KEY: "[]"}
    out, report = reorganise_profile(
        profile, "person", "KSM", "detainee", "guantanamo", "m", False,
        ask=_asker(GOOD, []),
    )  # fmt: skip
    assert out is profile and report is None


def test_starter_headings_come_from_the_domain_config():
    config = {
        "profiles": {
            "starter_headings": {
                "people": {"detainee": ["Background", "Capture"], "default": ["Role"]}
            }
        }
    }
    assert starter_headings(config, "person", "detainee") == ["Background", "Capture"]
    assert starter_headings(config, "people", "PersonType.DETAINEE")[0] == "Background"
    assert starter_headings(config, "person", "lawyer") == ["Role"]
    assert starter_headings(config, "event", "legal") == []
    from src.config_loader import get_domain_config

    real = get_domain_config("guantanamo").load_config()
    assert "Capture and CIA custody" in starter_headings(real, "person", "detainee")


def test_update_profile_reorganises_then_splits_when_on(monkeypatch):
    from src.engine import profiles

    monkeypatch.setenv("HINBOX_SECTION_REORGANISE", "on")
    monkeypatch.delenv("HINBOX_SECTION_SPLIT", raising=False)
    monkeypatch.setattr(section_reorganise, "REORGANISE_CHARS", 100)
    seen: List[Any] = []
    monkeypatch.setattr(
        section_reorganise, "make_asker", lambda model, local, step: _asker(GOOD, seen)
    )
    monkeypatch.setitem(
        profiles._UPDATERS, "edit",
        lambda *a, **k: ({"text": TEXT, "sources": ["a0", "a1"]}, []),
    )  # fmt: skip
    out, _, _ = profiles.update_profile(
        "person", "KSM", {"text": "old", "sources": ["a0"]},
        profiles.VersionedProfile(), "article", "a1", approach="edit", kind="detainee",
    )  # fmt: skip
    assert "### Capture and CIA custody" in out["text"]
    assert out["reorganised"] == "done" and len(seen) == 1
    monkeypatch.setenv("HINBOX_SECTION_REORGANISE", "off")
    out, _, _ = profiles.update_profile(
        "person", "KSM", {"text": "old", "sources": ["a0"]},
        profiles.VersionedProfile(), "article", "a1", approach="edit",
    )  # fmt: skip
    assert out["text"] == TEXT
    # By default only people are reorganised
    monkeypatch.delenv("HINBOX_SECTION_REORGANISE")
    out, _, _ = profiles.update_profile(
        "event", "9/11 trial", {"text": "old", "sources": ["a0"]},
        profiles.VersionedProfile(), "article", "a1", approach="edit",
    )  # fmt: skip
    assert out["text"] == TEXT and "reorganised" not in out
    assert section_reorganise.reorganise_enabled("people")


def test_reorganised_claims_keep_their_grounding_verdicts():
    from scripts.section_split_bench import _split_one, regrounded

    articles = {f"a{i}": "Article text." for i in range(5)}
    out, reports, mid = _split_one(
        TEXT,
        "person",
        "KSM",
        _asker(GOOD, []),
        10_000,
        (STARTER, _asker(GOOD, []), 100),
    )
    assert reports[0]["section"] == "*" and reports[0]["status"] == "done"
    assert mid == out  # nothing long enough left to split
    assert regrounded(TEXT, out, articles, "KSM") == 0


def test_replay_entity_spec_takes_a_kind():
    from src.evals import profile_replay as pr

    spec = pr.parse_entity_spec("Khalid Sheikh Mohammed|person|KSM,Mohammed|detainee")
    assert spec.kind == "detainee" and spec.terms == ("KSM", "Mohammed")
    assert pr.parse_entity_spec("Camp X|location|Camp X").kind == ""


def test_a_long_opening_is_numbered_with_the_rest():
    long_opening = " ".join(f"Trump said thing {i}.^[a{i % 5}]" for i in range(80))
    text = long_opening + "\n\n### Background\n\n" + "\n\n".join(PARAS)
    _, _, seen = _run(GOOD, text)
    assert "[1] Trump said thing 0." in seen[0][1]["content"]


def test_the_model_is_told_what_each_heading_is_for():
    seen: List[Any] = []
    notes = {"Background": "origin and life before capture"}
    reorganise_text(
        TEXT, "person", "KSM", STARTER, _asker(GOOD, seen), min_chars=100, notes=notes
    )
    assert "- Background: origin and life before capture" in seen[0][0]["content"]
    from src.config_loader import get_domain_config
    from src.engine.section_reorganise import starter_notes

    real = get_domain_config("guantanamo").load_config()
    assert (
        "rendition"
        in starter_notes(real, "person", "detainee")["Capture and CIA custody"]
    )


def test_moved_long_claims_keep_verdicts_under_a_shorter_heading():
    from scripts.section_split_bench import regrounded

    long_para = " ".join(["The CIA ran sites in Thailand and Poland."] * 18) + "^[a1]"
    before = "### Leadership allegations and secrecy\n\n" + long_para + "\n"
    after = "### Overview\n\n" + long_para + "\n"
    assert len(long_para) > 600
    assert regrounded(before, after, {"a1": "Article."}, "CIA") == 0


def test_a_heading_copied_back_with_its_note_means_the_starter():
    from src.engine.section_reorganise import _starter_for

    starter = ["Overview", "Role", "Role at Guantánamo"]
    assert (
        _starter_for("Overview: what the organization is, its leaders", starter)
        == "Overview"
    )
    assert (
        _starter_for("Role at Guantánamo: its part in detention", starter)
        == "Role at Guantánamo"
    )
    assert _starter_for("role", starter) == "Role"
    assert _starter_for("Roles of lawyers", starter) is None
    reply = SectionSplit(
        subsections=[
            SubSection(
                heading="Background: origin, life and activities before capture",
                paragraphs=[1, 2, 3, 4],
            ),
            SubSection(heading="Health and treatment", paragraphs=[5, 6, 7]),
        ]
    )
    out, report, _ = _run(reply)
    assert report["status"] == "done" and "### Background\n" in out


def test_rewrite_update_keeps_the_reorganised_marker(monkeypatch):
    """The rewrite updater returns a new dict; the marker must survive it."""
    from src.engine import profiles
    from src.engine.profiles import VersionedProfile

    def rewrite(*args, **kwargs):
        return {"text": TEXT, "sources": ["a0", "a1", "a2", "a3", "a4"]}, []

    monkeypatch.setattr(profiles, "_update_profile_internal", rewrite)
    existing = {"text": TEXT, "sources": ["a0"], "reorganised": "refused"}
    updated, _, _ = profiles.update_profile(
        "people", "Khalid Sheik Mohammed", existing, VersionedProfile(),
        "New article.", "a5", approach="rewrite",
    )  # fmt: skip
    assert updated["reorganised"] == "refused"
