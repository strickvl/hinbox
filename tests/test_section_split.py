"""Splitting a long profile section into sub-sections (src/engine/section_split.py).

Scenario: KSM's "### Legal proceedings" holds eight paragraphs, from the 2008
charges to the 2024 plea deal. The model is asked only which paragraphs go
under which sub-heading; code moves them word for word.
"""

import json
import re
from typing import Any, Dict, List

import pytest

from src.engine import section_split
from src.engine.section_split import (
    SectionSplit,
    SubSection,
    long_sections,
    plan_groups,
    split_long_sections,
)

PARAS = [
    "He was charged in 2008 with planning the 9/11 attacks.^[a1]",
    "At his 2008 arraignment he asked to be put to death.^[a1]",
    "Charges were refiled in 2012.^[a2]",
    "Pretrial hearings began in 2012 and dealt with CIA torture.^[a2]",
    "In 2019 the judge heard FBI agents testify.^[a3]",
    "Prosecutors began plea talks in 2017.^[a4]",
    "A plea deal was signed in July 2024.^[a5]",
    "Defense Secretary Austin withdrew the deal in August 2024.^[a5]",
]
TEXT = (
    "### Background\n\nKSM was born in Kuwait.^[a0]\n\n"
    "### Legal proceedings\n\n" + "\n\n".join(PARAS) + "\n\n"
    "### Health\n\nHe has a heart condition.^[a3]\n"
)
GOOD = SectionSplit(
    subsections=[
        SubSection(heading="Charges and arraignment", paragraphs=[1, 2, 3]),
        SubSection(heading="Pretrial hearings", paragraphs=[4, 5]),
        SubSection(heading="Plea negotiations (2017–2024)", paragraphs=[6, 7, 8]),
    ]
)


def _asker(reply: Any, seen: List[List[Dict[str, str]]]):
    def ask(msgs):
        seen.append(msgs)
        if isinstance(reply, Exception):
            raise reply
        return reply

    return ask


def _split(reply: Any, text: str = TEXT):
    seen: List[List[Dict[str, str]]] = []
    out, reports = split_long_sections(
        text, "person", "Khalid Sheik Mohammed", _asker(reply, seen), min_chars=200
    )
    return out, reports, seen


def test_only_the_long_section_is_found():
    found = long_sections(TEXT, min_chars=200)
    assert [s["heading"] for s in found] == ["Legal proceedings"]
    assert found[0]["units"] == PARAS
    assert long_sections(TEXT) == []  # the 6,000-character default


def test_split_moves_paragraphs_word_for_word_under_sub_headings():
    out, reports, seen = _split(GOOD)
    assert len(seen) == 1
    assert reports[0]["status"] == "split"
    assert "#### Charges and arraignment" in out
    assert out.index("#### Plea negotiations") < out.index("### Health")
    for p in PARAS:
        assert out.count(p) == 1
    # the other sections are untouched
    assert out.startswith("### Background\n\nKSM was born in Kuwait.^[a0]\n\n")
    assert out.endswith("### Health\n\nHe has a heart condition.^[a3]\n")


def test_the_model_sees_numbered_paragraphs_and_existing_headings():
    _, _, seen = _split(GOOD)
    system, user = seen[0][0]["content"], seen[0][1]["content"]
    assert '"Background"' in system and '"Health"' in system
    assert "[1] He was charged in 2008" in user and "[8] Defense Secretary" in user


def test_paragraphs_far_apart_can_share_a_sub_section():
    reply = SectionSplit(
        subsections=[
            SubSection(heading="Court case", paragraphs=[1, 2, 3, 4, 5, 8]),
            SubSection(heading="Plea talks", paragraphs=[6, 7]),
        ]
    )
    out, reports, _ = _split(reply)
    assert reports[0]["status"] == "split"
    court = out[out.index("#### Court case") : out.index("#### Plea talks")]
    assert PARAS[7] in court  # kept with its group, after paragraph 5


def test_missing_paragraph_goes_with_its_neighbour_and_repeats_keep_first_place():
    reply = SectionSplit(
        subsections=[
            SubSection(heading="Charges", paragraphs=[1, 2, 3, 4]),
            SubSection(heading="Hearings", paragraphs=[4, 6, 7, 8]),
        ]
    )
    plan, why = plan_groups(reply, 8, [])
    assert why == "" and plan == [("Charges", [1, 2, 3, 4, 5]), ("Hearings", [6, 7, 8])]


@pytest.mark.parametrize(
    "heading",
    ["Legal proceedings", "Health", "", "A heading that is far too long to be a sub-heading here",
     "Charges ^[a1]"],
)  # fmt: skip
def test_bad_headings_refuse_the_split(heading):
    reply = SectionSplit(
        subsections=[
            SubSection(heading=heading, paragraphs=[1, 2, 3, 4]),
            SubSection(heading="Plea talks", paragraphs=[5, 6, 7, 8]),
        ]
    )
    out, reports, _ = _split(reply)
    assert out == TEXT and reports[0]["status"] == "refused"


def test_one_group_or_too_few_placed_refuses():
    one = SectionSplit(
        subsections=[SubSection(heading="All", paragraphs=list(range(1, 9)))]
    )
    few = SectionSplit(
        subsections=[SubSection(heading="A", paragraphs=[1]), SubSection(heading="B", paragraphs=[2])]
    )  # fmt: skip
    for reply in (one, few):
        out, reports, _ = _split(reply)
        assert out == TEXT and reports[0]["status"] == "refused"


def test_a_failed_call_keeps_the_section():
    out, reports, _ = _split(RuntimeError("timeout"))
    assert out == TEXT and reports[0]["status"] == "failed"


def test_markdown_in_headings_is_cleaned():
    reply = SectionSplit(
        subsections=[
            SubSection(heading="#### **Charges**", paragraphs=[1, 2, 3, 4]),
            SubSection(heading="Plea talks:", paragraphs=[5, 6, 7, 8]),
        ]
    )
    out, _, _ = _split(reply)
    assert "#### Charges\n" in out and "#### Plea talks\n" in out


def test_a_long_paragraph_is_cut_into_sentence_runs_that_keep_their_citations():
    long_para = " ".join(f"Hearing {i} was held in 2013.^[a{i}]" for i in range(300))
    text = "### Legal proceedings\n\n" + long_para + "\n"
    units = long_sections(text, min_chars=200)[0]["units"]
    assert len(units) >= 2 and all(len(u) <= 1500 for u in units)
    half = len(units) // 2
    reply = SectionSplit(
        subsections=[
            SubSection(heading="Early hearings", paragraphs=list(range(1, half + 1))),
            SubSection(heading="Later hearings", paragraphs=list(range(half + 1, len(units) + 1))),
        ]
    )  # fmt: skip
    out, reports, _ = _split(reply, text)
    assert reports[0]["status"] == "split"
    assert section_split._sentences(out) == section_split._sentences(text)
    assert "Hearing 299 was held in 2013.^[a299]" in out


def test_list_items_can_be_moved():
    items = [f"- Item {i} filed in 2010^[a{i}]" for i in range(8)]
    text = "### Filings\n\n" + "\n".join(items) + "\n"
    reply = SectionSplit(
        subsections=[
            SubSection(heading="Early", paragraphs=[1, 3, 5, 7]),
            SubSection(heading="Late", paragraphs=[2, 4, 6, 8]),
        ]
    )
    out, reports, _ = _split(reply, text)
    assert reports[0]["status"] == "split"
    assert all(out.count(i) == 1 for i in items)


def test_split_profile_keeps_other_keys(monkeypatch):
    monkeypatch.setattr(section_split, "SPLIT_CHARS", 200)
    profile = {"text": TEXT, "sources": ["a0", "a1"], "tags": ["x"]}
    out, reports = section_split.split_profile(
        profile, "person", "KSM", "m", False, ask=_asker(GOOD, [])
    )
    assert out["sources"] == ["a0", "a1"] and out["tags"] == ["x"]
    assert out["text"] != TEXT and profile["text"] == TEXT


def test_split_is_on_for_people_only_by_default(monkeypatch):
    monkeypatch.delenv("HINBOX_SECTION_SPLIT", raising=False)
    assert section_split.split_enabled("people")
    assert section_split.split_enabled("person")
    assert not section_split.split_enabled("events")
    monkeypatch.setenv("HINBOX_SECTION_SPLIT", "on")
    assert section_split.split_enabled("events")
    monkeypatch.setenv("HINBOX_SECTION_SPLIT", "people, organizations")
    assert section_split.split_enabled("organization")
    assert not section_split.split_enabled("location")
    monkeypatch.setenv("HINBOX_SECTION_SPLIT", "off")
    assert not section_split.split_enabled("people")


def test_sections_are_on_for_local_people_unless_turned_off(monkeypatch):
    # 9 Nov rerun: local Qwen put 93 of 100 checked sentences under the right
    # top-level heading, so local runs section people like cloud runs
    from src.engine.section_reorganise import reorganise_enabled

    for env in ("HINBOX_SECTION_SPLIT", "HINBOX_SECTION_REORGANISE"):
        monkeypatch.delenv(env, raising=False)
    assert section_split.split_enabled("people", local=True)
    assert reorganise_enabled("people", local=True)
    assert not reorganise_enabled("events", local=True)
    monkeypatch.setenv("HINBOX_SECTION_SPLIT", "off")
    monkeypatch.setenv("HINBOX_SECTION_REORGANISE", "off")
    assert not section_split.split_enabled("people", local=True)
    assert not reorganise_enabled("people", local=True)


def test_update_profile_splits_after_the_edit_when_on(monkeypatch):
    from src.engine import profiles

    monkeypatch.setenv("HINBOX_SECTION_SPLIT", "on")
    monkeypatch.setattr(section_split, "SPLIT_CHARS", 200)
    monkeypatch.setenv("HINBOX_SPLIT_REFILE", "off")
    monkeypatch.setattr(
        section_split, "make_asker", lambda model, local: _asker(GOOD, [])
    )
    monkeypatch.setitem(
        profiles._UPDATERS, "edit",
        lambda *a, **k: ({"text": TEXT, "sources": ["a0", "a1"]}, []),
    )  # fmt: skip
    out, versions, _ = profiles.update_profile(
        "people", "KSM", {"text": "old", "sources": ["a0"]},
        profiles.VersionedProfile(), "article", "a1", approach="edit",
    )  # fmt: skip
    assert "#### Plea negotiations" in out["text"]
    if profiles.ENABLE_PROFILE_VERSIONING:
        assert "#### Plea negotiations" in versions.versions[-1].profile_data["text"]


def test_bench_cases_writes_split_copies_of_the_eval_cases(tmp_path, monkeypatch):
    import json

    from scripts import section_split_bench as bench

    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    case = {"case_id": "c1", "entity_type": "people", "entity_name": "KSM",
            "profile": {"text": TEXT, "sources": ["a0"]}, "kind": "new_fact", "split": "dev"}  # fmt: skip
    (eval_dir / "profile_update_cases.jsonl").write_text(json.dumps(case) + "\n")
    (eval_dir / "claim_labels.jsonl").write_text("{}\n")
    monkeypatch.setattr(bench, "_asker", lambda model: _asker(GOOD, []))
    out = tmp_path / "out"
    bench.main(["cases", "--eval-dir", str(eval_dir), "--model", "m", "--out", str(out),
                "--min-chars", "200"])  # fmt: skip
    row = json.loads((out / "eval" / "profile_update_cases.jsonl").read_text())
    assert "#### Pretrial hearings" in row["profile"]["text"]
    assert (out / "eval" / "claim_labels.jsonl").exists()
    summary = json.loads((out / "cases_summary.json").read_text())
    assert (
        summary["cases_changed"] == 1
        and summary["profiles_with_a_changed_sentence"] == 0
    )


def test_bench_regrounding_count_is_zero_after_a_split():
    from scripts.section_split_bench import regrounded

    out, _, _ = _split(GOOD)
    articles = {f"a{i}": PARAS[0] for i in range(6)}
    assert regrounded(TEXT, out, articles, "KSM") == 0
    assert regrounded(TEXT, out.replace("2019", "2018"), articles, "KSM") == 1


def test_uncited_text_moves_with_the_cited_sentence_after_it():
    paras = list(PARAS)
    paras[3] = "He later moved to Karachi."  # no citation of its own
    text = "### Legal proceedings\n\n" + "\n\n".join(paras) + "\n"
    units = long_sections(text, min_chars=200)[0]["units"]
    assert len(units) == 7
    assert units[3] == "He later moved to Karachi.\n\n" + PARAS[4]


def test_uncited_tail_stays_at_the_end_of_the_section():
    # "Hearings resumed." has no citation: its claim is the next citation's
    # (under "### Health"), so it must not move up the section
    text = TEXT.replace("\n\n### Health", "\n\nHearings resumed.\n\n### Health")
    section = long_sections(text, min_chars=200)[0]
    assert section["units"] == PARAS and section["tail"] == "Hearings resumed."
    reply = SectionSplit(
        subsections=[
            SubSection(heading="Charges", paragraphs=[1, 2, 6, 7, 8]),
            SubSection(heading="Hearings", paragraphs=[3, 4, 5]),
        ]
    )
    out, reports, _ = _split(reply, text)
    assert reports[0]["status"] == "split"
    assert out.index("Hearings resumed.") > out.index(PARAS[4])
    assert out.index("Hearings resumed.") < out.index("### Health")
    from scripts.section_split_bench import regrounded

    articles = {f"a{i}": PARAS[0] for i in range(6)}
    assert regrounded(text, out, articles, "KSM") == 0


def test_profiles_with_fact_rows_are_not_split(monkeypatch):
    monkeypatch.setattr(section_split, "SPLIT_CHARS", 200)
    profile = {"text": TEXT, "fact_rows_json": "[{}]"}
    out, reports = section_split.split_profile(
        profile, "person", "KSM", "m", False, ask=_asker(GOOD, [])
    )
    assert out is profile and reports == []


def test_bench_baseline_needs_no_model_whatever_the_judge(monkeypatch):
    from scripts.section_split_bench import regrounded

    monkeypatch.setenv("HINBOX_GROUNDING_MODEL", "ollama/some-local-judge")
    out, _, _ = _split(GOOD)
    articles = {f"a{i}": PARAS[0] for i in range(6)}
    assert regrounded(TEXT, out, articles, "KSM") == 0


def test_a_section_made_by_a_split_is_not_split_again():
    nested = "### Legal proceedings\n\n#### Pretrial hearings\n\n" + "\n\n".join(PARAS)
    assert long_sections(nested, min_chars=200) == []
    top = "### Pretrial hearings\n\n" + "\n\n".join(PARAS)
    assert [s["heading"] for s in long_sections(top, min_chars=200)] == [
        "Pretrial hearings"
    ]


# A reorganised profile: the long sub-section "Legal proceedings" under the
# starter heading grows again and is re-split beside its sibling
NESTED = (
    "### Background\n\nKSM was born in Kuwait.^[a0]\n\n"
    "### Military commission or legal case\n\n"
    "#### Arraignment\n\nHe was arraigned in June 2008.^[a1]\n\n"
    "#### Legal proceedings\n\n" + "\n\n".join(PARAS) + "\n\n"
    "### Health and treatment\n\nHe has a heart condition.^[a3]\n"
)


def test_a_grown_sub_section_is_re_split_beside_its_siblings():
    seen: List[List[Dict[str, str]]] = []
    out, reports = split_long_sections(
        NESTED, "person", "KSM", _asker(GOOD, seen), min_chars=200, resplit=True
    )
    assert reports[0]["status"] == "split"
    assert "#### Legal proceedings" not in out  # replaced, not nested under
    assert "#####" not in out
    for heading in ("Charges and arraignment", "Pretrial hearings"):
        assert f"\n#### {heading}\n" in out
    # Siblings stay under their parent, before the next starter heading
    assert out.index("#### Arraignment") < out.index("#### Charges and arraignment")
    assert out.index("#### Plea negotiations") < out.index("### Health and treatment")
    assert section_split._sentences(out) == section_split._sentences(NESTED)
    system = seen[0][0]["content"]
    assert (
        "beside the other sub-sections of" in system and "may keep the name" in system
    )


def test_a_re_split_may_keep_the_old_name_and_join_a_sibling():
    keep = SectionSplit(subsections=[
        SubSection(heading="Legal proceedings", paragraphs=[1, 2, 3, 4, 5]),
        SubSection(heading="Plea negotiations", paragraphs=[6, 7, 8]),
    ])  # fmt: skip
    out, reports = split_long_sections(
        NESTED, "person", "KSM", _asker(keep, []), min_chars=200, resplit=True
    )
    assert reports[0]["status"] == "split" and out.count("#### Legal proceedings") == 1
    clash = SectionSplit(subsections=[
        SubSection(heading="Arraignment", paragraphs=[1, 2, 3]),
        SubSection(heading="Plea negotiations", paragraphs=[4, 5, 6, 7, 8]),
    ])  # fmt: skip
    seen: List[List[Dict[str, str]]] = []
    out, reports = split_long_sections(
        NESTED, "person", "KSM", _asker(clash, seen), min_chars=200, resplit=True
    )
    # Paragraphs put under a sibling's name join that sibling, after its text
    assert reports[0]["status"] == "split"
    assert reports[0]["joined"] == [["Arraignment", [1, 2, 3]]]
    assert reports[0]["subsections"] == [["Plea negotiations", [4, 5, 6, 7, 8]]]
    assert out.count("#### Arraignment") == 1 and "#### Legal proceedings" not in out
    arraignment = out.split("#### Arraignment")[1].split("####")[0]
    assert arraignment.index("June 2008") < arraignment.index(PARAS[0])
    assert all(p in arraignment for p in PARAS[:3])
    assert section_split._sentences(out) == section_split._sentences(NESTED)
    assert "Arraignment" in seen[0][0]["content"]  # in the don't-repeat list
    # Only new headings may be put under a sibling's name with it
    only = SectionSplit(subsections=[
        SubSection(heading="Arraignments", paragraphs=[1, 2, 3, 4, 5, 6, 7, 8]),
    ])  # fmt: skip
    out, reports = split_long_sections(
        NESTED, "person", "KSM", _asker(only, []), min_chars=200, resplit=True
    )
    assert reports[0]["status"] == "refused" and out == NESTED


def test_a_join_above_a_section_still_to_split_keeps_its_place():
    # Bottom up: "Legal proceedings" joins "Arraignment", above "Hearings",
    # which is split next from line numbers the join moved
    hearings = [f"Hearing {i} was held in 201{i}.^[h{i}]" for i in range(6)]
    text = (
        "### Military commission\n\n#### Arraignment\n\nHe was arraigned.^[a1]\n\n"
        "#### Hearings\n\n" + "\n\n".join(hearings) + "\n\n"
        "#### Legal proceedings\n\n" + "\n\n".join(PARAS) + "\n\n"
        "### Health\n\nHe has a heart condition.^[a3]\n"
    )
    replies = iter([
        SectionSplit(subsections=[
            SubSection(heading="Arraignment", paragraphs=[1, 2, 3]),
            SubSection(heading="Plea talks", paragraphs=[4, 5, 6, 7, 8]),
        ]),
        SectionSplit(subsections=[
            SubSection(heading="Early hearings", paragraphs=[1, 2, 3]),
            SubSection(heading="Later hearings", paragraphs=[4, 5, 6]),
        ]),
    ])  # fmt: skip
    out, reports = split_long_sections(
        text, "person", "KSM", lambda m: next(replies), min_chars=150, resplit=True
    )
    assert [r["status"] for r in reports] == ["split", "split"]
    assert section_split._sentences(out) == section_split._sentences(text)
    early = out.split("#### Early hearings")[1].split("####")[0]
    assert all(h in early for h in hearings[:3]) and PARAS[0] not in early
    arraignment = out.split("#### Arraignment")[1].split("####")[0]
    assert all(p in arraignment for p in PARAS[:3])


def test_refiling_a_re_split_leaves_the_joined_pieces_alone():
    # Qwen replay: the refile was asked about the joined pieces too, one got
    # no usable answer, and looking up its split heading raised KeyError
    from src.engine.section_place import Placement, Placements

    reply = SectionSplit(subsections=[
        SubSection(heading="Arraignment", paragraphs=[1, 2]),
        SubSection(heading="Charges", paragraphs=[3, 4]),
        SubSection(heading="Plea negotiations", paragraphs=[5, 6, 7, 8]),
    ])  # fmt: skip
    asked: List[str] = []

    def place(msgs):
        asked.append(msgs[1]["content"])
        return Placements(placements=[  # sentence 2 of each batch: no answer
            Placement(sentence=1, starts_with="", about="", heading="Charges")
        ])  # fmt: skip

    out, reports = split_long_sections(
        NESTED, "person", "KSM", _asker(reply, []), min_chars=200, resplit=True,
        place=place,
    )  # fmt: skip
    assert reports[0]["status"] == "split"
    assert reports[0]["joined"] == [["Arraignment", [1, 2]]]
    assert not any(PARAS[0] in a or PARAS[1] in a for a in asked)
    assert section_split._sentences(out) == section_split._sentences(NESTED)
    arraignment = out.split("#### Arraignment")[1].split("####")[0]
    assert PARAS[0] in arraignment and PARAS[1] in arraignment


def test_a_re_split_with_a_join_can_still_be_refiled():
    # Codex review: the refiled candidate was compared before the joined
    # pieces were added back, so its claims never matched and it was dropped
    from src.engine.section_place import Placement, Placements

    reply = SectionSplit(subsections=[
        SubSection(heading="Arraignment", paragraphs=[1, 2]),
        SubSection(heading="Charges", paragraphs=[3, 4, 5]),
        SubSection(heading="Plea negotiations", paragraphs=[6, 7, 8]),
    ])  # fmt: skip

    def place(msgs):
        found = re.findall(r"^\[(\d+)\] (.*)$", msgs[1]["content"], re.M)
        return Placements(placements=[
            Placement(sentence=int(n), about="",
                      heading="Plea negotiations" if "FBI" in t else
                      "Charges" if int(n) <= 3 else "Plea negotiations")
            for n, t in found
        ])  # fmt: skip

    out, reports = split_long_sections(
        NESTED, "person", "KSM", _asker(reply, []), min_chars=200, resplit=True,
        place=place,
    )  # fmt: skip
    assert reports[0]["status"] == "split"
    assert reports[0]["refile"]["status"] == "done"
    assert reports[0]["subsections"] == [
        ["Charges", [3, 4]], ["Plea negotiations", [5, 6, 7, 8]],
    ]  # fmt: skip
    assert section_split._sentences(out) == section_split._sentences(NESTED)
    assert PARAS[4] in out.split("#### Plea negotiations")[1]


def test_joins_into_two_siblings_above_keep_a_pending_section_in_place():
    # Codex review: the first join pushed the second one's insertion point
    # past the old start of the section being split, so the second shift
    # was not counted and "Pending" was then split from the wrong lines
    hearings = [f"Hearing {i} was held in 201{i}.^[h{i}]" for i in range(6)]
    source = [f"Filing {i} was made in 200{i % 10}.^[f{i}]" for i in range(12)]
    text = (
        "### Military commission\n\n#### Alpha\n\nAlpha began.^[a1]\n\n"
        "#### Beta\n\nBeta began.^[b1]\n\n"
        "#### Pending\n\n" + "\n\n".join(hearings) + "\n\n"
        "#### Source\n\n" + "\n\n".join(source) + "\n\n"
        "### Health\n\nHe has a heart condition.^[a3]\n"
    )
    replies = iter([
        SectionSplit(subsections=[
            SubSection(heading="Alpha", paragraphs=list(range(1, 10))),
            SubSection(heading="Beta", paragraphs=[10, 11]),
            SubSection(heading="Late filings", paragraphs=[12]),
        ]),
        SectionSplit(subsections=[
            SubSection(heading="Early hearings", paragraphs=[1, 2, 3]),
            SubSection(heading="Later hearings", paragraphs=[4, 5, 6]),
        ]),
    ])  # fmt: skip
    out, reports = split_long_sections(
        text, "person", "KSM", lambda m: next(replies), min_chars=150, resplit=True
    )
    assert [r["status"] for r in reports] == ["split", "split"]
    assert section_split._sentences(out) == section_split._sentences(text)
    early = out.split("#### Early hearings")[1].split("####")[0]
    assert [h in early for h in hearings] == [True] * 3 + [False] * 3
    beta = out.split("#### Beta")[1].split("####")[0]
    assert source[9] in beta and source[10] in beta


def test_only_a_re_split_joins_a_heading_it_repeats():
    clash = SectionSplit(subsections=[
        SubSection(heading="Health", paragraphs=[1, 2, 3, 4]),
        SubSection(heading="Plea talks", paragraphs=[5, 6, 7, 8]),
    ])  # fmt: skip
    assert plan_groups(clash, 8, ["Health"])[0] is None
    assert plan_groups(clash, 8, ["Health"], siblings=["Health"])[0] == [
        ("Health", [1, 2, 3, 4]), ("Plea talks", [5, 6, 7, 8]),
    ]  # fmt: skip


def test_re_split_needs_resplit_and_stops_at_the_sibling_cap(monkeypatch):
    assert long_sections(NESTED, min_chars=200) == []
    found = long_sections(NESTED, min_chars=200, resplit=True)
    assert [(s["heading"], s["parent"]) for s in found] == [
        ("Legal proceedings", "Military commission or legal case")
    ]
    monkeypatch.setattr(section_split, "MAX_SIBLINGS", 2)
    assert long_sections(NESTED, min_chars=200, resplit=True) == []


def test_the_split_call_asks_for_no_more_headings_than_fit(monkeypatch):
    # r21 local replay: every refused Qwen split was a re-split asked for
    # "2 to 5" headings with room for only 2 or 3 ("4 sub-sections")
    seen: List[List[Dict[str, str]]] = []
    monkeypatch.setattr(section_split, "MAX_SIBLINGS", 4)  # Arraignment + 3 more
    split_long_sections(
        NESTED, "person", "KSM", _asker(GOOD, seen), min_chars=200, resplit=True
    )
    assert "under 2 to 3 headings that replace it" in seen[0][0]["content"]
    seen.clear()
    monkeypatch.setattr(section_split, "MAX_SIBLINGS", 8)
    split_long_sections(NESTED, "person", "KSM", _asker(GOOD, seen), min_chars=200,
                        resplit=True)  # fmt: skip
    assert "under 2 to 5 headings" in seen[0][0]["content"]


def test_split_profile_re_splits_only_reorganised_profiles(monkeypatch):
    monkeypatch.setattr(section_split, "SPLIT_CHARS", 200)
    flat, _ = section_split.split_profile(
        {"text": NESTED}, "person", "KSM", "m", False, ask=_asker(GOOD, [])
    )
    assert flat["text"] == NESTED
    done, _ = section_split.split_profile(
        {"text": NESTED, "reorganised": "done"}, "person", "KSM", "m", False,
        ask=_asker(GOOD, []),
    )  # fmt: skip
    assert "#### Pretrial hearings" in done["text"]


def test_re_splits_under_one_parent_share_the_sibling_cap(monkeypatch):
    # Two long sub-sections under one parent, room for one more sub-section:
    # the second split (bottom up) fits, the first must then be refused
    two = NESTED.replace(
        "### Health and treatment",
        "#### Detention hearings\n\n"
        + "\n\n".join(PARAS)
        + "\n\n### Health and treatment",
    )
    replies = iter([
        SectionSplit(subsections=[
            SubSection(heading="Review boards", paragraphs=[1, 2, 3, 4]),
            SubSection(heading="Habeas", paragraphs=[5, 6, 7, 8]),
        ]),
        SectionSplit(subsections=[
            SubSection(heading="Charges", paragraphs=[1, 2, 3, 4]),
            SubSection(heading="Pleas", paragraphs=[5, 6, 7, 8]),
        ]),
    ])  # fmt: skip
    monkeypatch.setattr(section_split, "MAX_SIBLINGS", 4)
    out, reports = split_long_sections(
        two, "person", "KSM", lambda m: next(replies), min_chars=200, resplit=True
    )
    assert [r["status"] for r in reports] == ["refused", "split"]
    assert "over 4 sub-sections" in reports[0]["reason"]
    assert out.count("\n#### ") <= 4


def test_a_parent_split_counts_its_existing_sub_sections(monkeypatch):
    parent = "### Legal case\n\n" + "\n\n".join(PARAS) + "\n\n#### Old\n\nA.^[a1]\n"
    assert long_sections(parent, min_chars=200)[0]["children"] == 1
    monkeypatch.setattr(section_split, "MAX_SIBLINGS", 2)
    out, reports = split_long_sections(
        parent, "person", "KSM", _asker(GOOD, []), min_chars=200
    )
    assert reports[0]["status"] == "refused" and out == parent


def test_a_refused_split_waits_until_its_section_grows(monkeypatch):
    monkeypatch.setattr(section_split, "SPLIT_CHARS", 200)
    clash = SectionSplit(subsections=[
        SubSection(heading="Background", paragraphs=[1, 2, 3, 4]),
        SubSection(heading="Hearings", paragraphs=[5, 6, 7, 8]),
    ])  # fmt: skip
    seen: List[List[Dict[str, str]]] = []
    profile, reports = section_split.split_profile(
        {"text": TEXT}, "person", "KSM", "m", False, ask=_asker(clash, seen)
    )
    assert reports[0]["status"] == "refused" and len(seen) == 1
    stored = json.loads(profile[section_split.REFUSED_KEY])
    assert list(stored.values()) == [reports[0]["chars"]]
    again, reports = section_split.split_profile(
        profile, "person", "KSM", "m", False, ask=_asker(GOOD, seen)
    )
    assert reports[0]["status"] == "waiting" and len(seen) == 1
    grown = {
        **again,
        "text": TEXT.replace(PARAS[7], PARAS[7] + " " + " ".join([PARAS[6]] * 3)),
    }
    done, reports = section_split.split_profile(
        grown, "person", "KSM", "m", False, ask=_asker(GOOD, seen)
    )
    assert reports[0]["status"] == "split" and len(seen) == 2
    assert done[section_split.REFUSED_KEY] == "{}"


def test_a_heading_that_repeats_another_in_other_words_is_refused():
    # Luna rerun: al Nashiri got "Judge Authority and Appeals (2018)" beside
    # "Appeals and Judge Authority"
    reply = SectionSplit(
        subsections=[
            SubSection(heading="Appeals and Judge Authority", paragraphs=[1, 2, 3, 4]),
            SubSection(heading="Judge Authority and Appeals (2018)", paragraphs=[5, 6, 7, 8]),
        ]
    )  # fmt: skip
    plan, why = plan_groups(reply, 8, [])
    assert plan is None and "in other words" in why
    other = SectionSplit(
        subsections=[
            SubSection(heading="Judge authority, appeals", paragraphs=[1, 2, 3, 4]),
            SubSection(heading="Plea talks", paragraphs=[5, 6, 7, 8]),
        ]
    )
    # Against a heading already in the profile too; fewer words is a new one
    assert plan_groups(other, 8, ["Appeal and judge authority"])[0] is None
    assert plan_groups(other, 8, ["Appeals", "Judge authority"])[0] is not None


def test_a_one_heading_answer_is_asked_again_once():
    # 9 Nov local control run: Qwen answered Trump's re-split with one
    # heading for every paragraph three times, and the section grew to 14,440
    one = SectionSplit(subsections=[
        SubSection(heading="Court case", paragraphs=[1, 2, 3, 4, 5, 6, 7, 8]),
        SubSection(heading="Plea talks", paragraphs=[6, 7]),  # all claimed first
    ])  # fmt: skip
    replies = iter([one, GOOD])
    seen: List[List[Dict[str, str]]] = []

    def ask(msgs):
        seen.append(msgs)
        return next(replies)

    out, reports = split_long_sections(TEXT, "person", "KSM", ask, min_chars=200)
    assert reports[0]["status"] == "split" and reports[0]["reasked"] == "1 sub-sections"
    assert len(seen) == 2 and seen[1][:2] == seen[0]
    assert seen[1][2]["role"] == "assistant"
    assert '("Court case")' in seen[1][3]["content"]
    assert "#### Pretrial hearings" in out
    assert section_split._sentences(out) == section_split._sentences(TEXT)
    # A second one-heading answer is refused as before; no third call
    seen.clear()
    replies = iter([one, one, GOOD])
    out, reports = split_long_sections(TEXT, "person", "KSM", ask, min_chars=200)
    assert (out, reports[0]["status"], len(seen)) == (TEXT, "refused", 2)
    assert reports[0]["reason"] == "1 sub-sections"


def test_other_refusals_and_a_failed_re_ask_are_not_asked_again():
    clash = SectionSplit(subsections=[
        SubSection(heading="Background", paragraphs=[1, 2, 3, 4]),
        SubSection(heading="Plea talks", paragraphs=[5, 6, 7, 8]),
    ])  # fmt: skip
    out, reports, seen = _split(clash)
    assert reports[0]["status"] == "refused" and len(seen) == 1
    one = SectionSplit(
        subsections=[SubSection(heading="All", paragraphs=list(range(1, 9)))]
    )
    calls = []

    def ask(msgs):
        calls.append(msgs)
        if len(calls) > 1:
            raise RuntimeError("down")
        return one

    out, reports = split_long_sections(TEXT, "person", "KSM", ask, min_chars=200)
    assert out == TEXT and reports[0]["status"] == "refused"
    assert reports[0]["reason"] == "1 sub-sections; re-ask failed: RuntimeError"
