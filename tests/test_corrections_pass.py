"""Tests for how a pipeline run carries out a historian's corrections
(src/engine/corrections_pass.py): strikes cut from the stored text, the
historian's own sentence put in their place, the edit guard, the run report,
and how the reader shows a strike a run carried out.

Every name, sentence and article here is invented (see tests/test_frontend.py).
"""

import json

from src.engine import corrections_pass as cp
from src.engine.profile_edits import ProfileEdit
from src.frontend import data_access
from src.utils.corrections import (
    append_correction,
    entity_ref,
    last_states,
    load_corrections,
)
from src.utils.file_ops import load_existing_entities
from tests.test_corrections import (  # noqa: F401
    FORCE_FED,
    ORIGIN,
    PAGE,
    _sentence_form,
    corr,
)
from tests.test_frontend import A1, A2, A3, TEXT, client  # noqa: F401

SHIPPING = "He worked for a shipping firm"
STUDIED = "Qadir studied engineering at a local college."


def _strike(
    sentence: str,
    cites,
    replacement: str = "",
    rep_cites: str = A3,
    cid: str = "c-0001",
):
    rec = {
        "id": cid,
        "kind": "strike",
        "entity": PAGE,
        "sentence": sentence,
        "cites": cites,
    }
    if replacement:
        rec.update(replacement=replacement, replacement_cites=rep_cites)
    return rec


def _apply(text, lines, done=(), withdrawn=()):
    return cp.apply_to_text(text, cp._rules(lines), set(done), list(withdrawn))


# ── one page's text ──────────────────────────────────────────


def test_a_strike_is_cut_and_then_held():
    text, states, _ = _apply(TEXT, [_strike(FORCE_FED, [A3])])
    assert FORCE_FED not in text and states["c-0001"] == ("applied", "")
    # The rest of the paragraph and every other sentence stay as they were
    assert "Sources disagree on the year he arrived" in text
    assert "####  " not in text and "\n\n\n" not in text
    assert text.count("^[") == TEXT.count("^[") - 1
    # Next run: nothing to cut, the strike holds
    again, states, _ = _apply(text, [_strike(FORCE_FED, [A3])], done=["c-0001"])
    assert again == text and states["c-0001"] == ("held", "")


def test_your_sentence_goes_in_its_place():
    rec = _strike(FORCE_FED, [A3], "He refused food for several weeks.")
    text, states, _ = _apply(TEXT, [rec])
    assert f"He refused food for several weeks.^[{A3}] Sources disagree" in text
    assert FORCE_FED not in text and states["c-0001"][0] == "applied"
    # Missing later (a run lost it): put back, and said so
    gone = text.replace(f"He refused food for several weeks.^[{A3}] ", "")
    back, states, _ = _apply(gone, [rec], done=["c-0001"])
    assert "He refused food for several weeks." in back
    assert states["c-0001"][0] == "re-applied" and "put back" in states["c-0001"][1]


def test_a_restatement_is_cut_but_not_the_true_half():
    struck = "Qadir was born in Taiz in 1975 and studied engineering in Karachi."
    text = (
        f"The article states that Qadir was born in Taiz in 1975 and studied engineering in Karachi.^[{A1}] "
        f"Qadir was born in Taiz in 1975.^[{A1}]"
    )
    out, states, _ = _apply(text, [_strike(struck, [A1])], done=["c-0001"])
    assert out == f"Qadir was born in Taiz in 1975.^[{A1}]"
    assert states["c-0001"][0] == "re-applied" and "said again" in states["c-0001"][1]


def test_other_words_are_listed_not_cut():
    struck = "Qadir was captured in Mukalla in 2002."
    text = f"Police arrested Qadir at a Mukalla hotel in 2002.^[{A1}]"
    out, states, maybe = _apply(text, [_strike(struck, [A1])])
    assert out == text and states["c-0001"][0] == "needs a look"
    assert maybe == [
        {
            "id": "c-0001",
            "sentence": "Police arrested Qadir at a Mukalla hotel in 2002.",
        }
    ]


def test_a_withdrawn_strike_is_undone():
    rec = _strike(FORCE_FED, [A3], "He refused food for several weeks.")
    text, _, _ = _apply(TEXT, [rec])
    undone, states, _ = _apply(text, [], withdrawn=[rec])
    assert f"{FORCE_FED}^[{A3}]" in undone and "refused food" not in undone
    assert states["c-0001"][0] == "undone"


def test_verdicts_hold_while_the_words_stay():
    rec = {"id": "c-0002", "kind": "verdict", "entity": PAGE, "sentence": FORCE_FED,
           "article_id": A3, "verdict": "supported"}  # fmt: skip
    _, states, _ = _apply(TEXT, [rec])
    assert states["c-0002"] == ("held", "")
    _, states, _ = _apply(TEXT.replace("several", "six"), [rec])
    assert states["c-0002"][0] == "needs a look"
    wait = {"id": "c-0003", "kind": "rename", "entity": PAGE, "name": "Yusuf al-Qadir"}
    assert _apply(TEXT, [wait])[1]["c-0003"][0] == "waiting"


# ── the edit guard ───────────────────────────────────────────


def _cs(tmp_path, *lines):
    path = tmp_path / "c.jsonl"
    for rec in lines:
        append_correction(str(path), {k: v for k, v in rec.items() if k != "id"})
    return load_corrections(str(path))


def test_the_edit_model_cant_bring_a_strike_back_or_reword_yours(tmp_path):
    rec = _strike(FORCE_FED, [A3], "He refused food for several weeks.")
    cs_ = _cs(tmp_path, rec)
    cp.configure_corrections(cs_)
    person = {"name": "Yusuf Qadir", "type": "detainee"}
    text = f"Qadir was held.^[{A1}] He refused food for several weeks.^[{A3}]"
    edits = [
        ProfileEdit(op="add", anchor="END", text=f"{FORCE_FED}^[{A3}] He lost weight.^[{A3}]"),
        ProfileEdit(op="replace", anchor="He refused food for several weeks.",
                    text=f"He was force-fed.^[{A3}]"),
        ProfileEdit(op="add", anchor="END", text=f"The article states that {FORCE_FED[0].lower()}{FORCE_FED[1:]}^[{A3}]"),
    ]  # fmt: skip
    # Outside a run's scope nothing changes
    assert cp.guard_corrections(text, edits)[0] == edits
    with cp.correction_scope("people", person):
        kept, report = cp.guard_corrections(text, edits)
    assert [e.text for e in kept] == [f"He lost weight.^[{A3}]"]
    assert len(report["restated"]) == 2 and len(report["kept_yours"]) == 1
    assert cs_.active[0]["id"] in cp._GUARDED
    # Another page is not guarded
    with cp.correction_scope("people", {"name": "Miriam Hale"}):
        assert cp.guard_corrections(text, edits)[0] == edits
    cp.configure_corrections(None)


# ── the run ──────────────────────────────────────────────────


def _run(tmp_path, monkeypatch):
    """A run's pass over the reader's made-up tables, then a save."""
    from src.process_and_extract import write_entities_to_files

    out = tmp_path / "entities"
    entities = load_existing_entities(str(out))
    path = data_access.build_store("guantanamo").corrections_path
    report = cp.run_corrections_pass(entities, path, str(out))
    write_entities_to_files(entities, str(out))
    data_access._CACHE.clear()
    return entities, report


def test_a_run_carries_out_the_reader_corrections(client, corr, tmp_path, monkeypatch):  # noqa: F811
    from tests.test_corrections import _index

    c = _index(client, FORCE_FED)
    form = {**_sentence_form(client, c), "kind": "strike",
            "replacement": "He refused food for several weeks.", "replacement_cites": A3}  # fmt: skip
    r = client.post("/corrections", data=form, headers=ORIGIN, follow_redirects=False)
    assert r.status_code == 303
    lost = {"kind": "strike", "entity": entity_ref("people", "Nobody", "Nobody", [A1]),
            "sentence": "Nobody was held.", "cites": [A1]}  # fmt: skip
    append_correction(str(corr), lost)
    entities, report = _run(tmp_path, monkeypatch)
    person = entities["people"]["Yusuf Qadir"]
    assert FORCE_FED not in person["profile"]["text"]
    assert "He refused food for several weeks." in person["profile"]["text"]
    latest = person["profile_versions"]["versions"][-1]
    assert latest["trigger_article_id"] == "corrections:c-0001"
    assert report["counts"] == {"applied": 1, "needs a look": 1}
    assert cp.describe_report(report) == "Corrections: 1 applied, 1 needs a look"
    assert last_states(str(corr)) == {"c-0001": "applied", "c-0002": "needs a look"}
    assert len(list(corr.parent.glob("report-*.json"))) == 1
    # The reader marks the stored sentence as theirs, and says a run did it
    page = client.get("/people/Yusuf%20Qadir").text
    # Once in the text (marked as theirs, not added again) and once in the box
    assert page.count("refused food for several weeks") == 2
    assert 'c-mine" id="claim-3">He refused food' in page
    listing = client.get("/corrections").text
    assert "Carried out in the tables by the run of" in listing
    assert "the page can&#39;t be found" in listing
    # The next run finds nothing to do
    _, report = _run(tmp_path, monkeypatch)
    assert (
        report["counts"] == {"held": 1, "needs a look": 1}
        and report["profiles_changed"] == 0
    )


def test_no_file_no_report(tmp_path):
    assert cp.run_corrections_pass({}, str(tmp_path / "none.jsonl")) is None
    assert not list(tmp_path.iterdir())


def test_report_is_written_beside_the_file_only_when_asked(tmp_path):
    path = tmp_path / "c.jsonl"
    path.write_text("")
    report = cp.run_corrections_pass({"people": {}}, str(path), write=False)
    assert report["counts"] == {} and not (tmp_path / cp.REPORT_LATEST).exists()
    cp.write_report(str(path), report)
    assert json.loads((tmp_path / cp.REPORT_LATEST).read_text())["rows"] == []


def test_a_page_whose_key_changed_is_found_by_name():
    from src.utils.corrections import resolve_ref

    ref = entity_ref("organizations", "Old key (abc)", "Harbour Police", [A1])
    pages = [("New key (def)", "Harbour Police", [A1], "p1")]
    assert resolve_ref(ref, pages) == "p1"
    two = pages + [("Third (ghi)", "Harbour Police", [A2], "p2")]
    assert resolve_ref(ref, two) == "p1"  # shares the article
    assert resolve_ref(ref, two + [("Fourth", "Harbour Police", [A1], "p3")]) is None


def test_edit_profile_applies_the_guard_in_a_run(monkeypatch, tmp_path):
    from src.engine import profile_edits as pe
    from src.engine.profile_edits import ProfileEditList

    reply = ProfileEditList(
        edits=[ProfileEdit(op="add", anchor="END", text=f"{FORCE_FED}^[{A3}]")]
    )
    monkeypatch.setattr(pe, "cloud_generation", lambda **_: reply)
    cp.configure_corrections(_cs(tmp_path, _strike(FORCE_FED, [A3])))
    profile = {"text": f"Qadir was held.^[{A1}]", "sources": [A1]}
    try:
        with cp.correction_scope("people", {"name": "Yusuf Qadir"}):
            out = pe.edit_profile(
                "person", "Yusuf Qadir", profile, "x", A3, model="openai/x"
            )
    finally:
        cp.configure_corrections(None)
    assert FORCE_FED not in out["text"]
    assert out["edit_report"]["corrections"]["restated"] == [FORCE_FED]


def test_the_history_names_a_corrections_version(client, corr, tmp_path, monkeypatch):  # noqa: F811
    append_correction(str(corr), _strike(FORCE_FED, [A3]) | {"id": None})
    _run(tmp_path, monkeypatch)
    page = client.get("/people/Yusuf%20Qadir").text
    assert "Your corrections c-0001, carried out" in page


def test_a_claim_over_two_paragraphs_is_cut_whole():
    # The uncited paragraph is part of the claim grounding judges, so a
    # strike of it means both paragraphs, but not the heading between
    text = f"Kept.^[{A1}]\n\nFirst wrong paragraph.\n\n### Later\n\nSecond wrong paragraph.^[{A2}]\n\nStays.^[{A3}]"
    struck = "First wrong paragraph.\n\nSecond wrong paragraph."
    out, states, _ = _apply(text, [_strike(struck, [A2])])
    assert out == f"Kept.^[{A1}]\n\n### Later\n\nStays.^[{A3}]"
    assert states["c-0001"][0] == "applied"


def test_cuts_never_shift_the_text_before_them():
    text = (
        f"Kept.^[{A1}]\n\n\nFirst wrong.^[{A1}] Second wrong.^[{A1}] Final kept.^[{A1}]"
    )
    lines = [
        _strike("First wrong.", [A1]),
        _strike("Second wrong.", [A1], cid="c-0002"),
    ]
    out, _, _ = _apply(text, lines)
    assert out == f"Kept.^[{A1}]\n\n\nFinal kept.^[{A1}]"
    # A list item left with no words goes, and one blank line stays between paragraphs
    listed = f"Intro.^[{A1}]\n\n- Wrong item.^[{A1}]\n- Kept item.^[{A1}]\n\nWrong paragraph.^[{A2}]\n\nEnd.^[{A3}]"
    lines = [
        _strike("- Wrong item.", [A1]),  # grounding keeps the list mark
        _strike("Wrong paragraph.", [A2], cid="c-0002"),
    ]
    out, _, _ = _apply(listed, lines)
    assert out == f"Intro.^[{A1}]\n\n- Kept item.^[{A1}]\n\nEnd.^[{A3}]"


def test_a_replace_on_part_of_your_sentence_is_dropped(tmp_path):
    rec = _strike(FORCE_FED, [A3], "He refused food. Guards logged it daily.")
    cp.configure_corrections(_cs(tmp_path, rec))
    text = f"Qadir was held.^[{A1}] He refused food. Guards logged it daily.^[{A3}]"
    edit = ProfileEdit(
        op="replace",
        anchor="Guards logged it daily.",
        text=f"Guards logged it weekly.^[{A3}]",
    )
    try:
        with cp.correction_scope("people", {"name": "Yusuf Qadir"}):
            kept, report = cp.guard_corrections(text, [edit])
    finally:
        cp.configure_corrections(None)
    assert kept == [] and report["kept_yours"] == ["Guards logged it daily."]


def test_an_add_inside_your_sentence_goes_after_it(tmp_path):
    from src.engine.profile_edits import apply_edits

    rec = _strike(FORCE_FED, [A3], "He refused food. Guards logged it daily.")
    cp.configure_corrections(_cs(tmp_path, rec))
    text = f"Qadir was held.^[{A1}] He refused food. Guards logged it daily.^[{A3}]"
    edit = ProfileEdit(
        op="add", anchor="He refused food.", text=f"He lost weight.^[{A3}]"
    )
    try:
        with cp.correction_scope("people", {"name": "Yusuf Qadir"}):
            kept, report = cp.guard_corrections(text, [edit])
    finally:
        cp.configure_corrections(None)
    out, _ = apply_edits(text, kept)
    assert "He refused food. Guards logged it daily.^[" in out
    assert out.index("He lost weight.") > out.index("Guards logged")
    assert report["moved_after_yours"] == [f"He lost weight.^[{A3}]"]


def test_a_sentence_put_back_after_an_empty_heading_is_its_own_paragraph():
    rec = _strike(FORCE_FED, [A3], "He refused food for several weeks.")
    text = f"Qadir was held.^[{A1}]\n\n## Later\n"
    out, states, _ = _apply(text, [rec], done=["c-0001"])
    assert out.endswith(f"## Later\n\nHe refused food for several weeks.^[{A3}]\n")
    again, states, _ = _apply(out, [rec], done=["c-0001"])
    assert again == out and states["c-0001"] == ("held", "")


def test_your_words_stay_literal_in_the_tables():
    from src.frontend.profile_render import Overlay, render_profile

    rec = _strike(FORCE_FED, [A3], "# He was *not* force-fed.")
    text, _, _ = _apply(TEXT, [rec])
    assert "\\# He was \\*not\\* force-fed." in text
    again, states, _ = _apply(text, [rec], done=["c-0001"])
    assert again == text and states["c-0001"] == ("held", "")
    rp = render_profile(text, overlay=Overlay(struck={"x": rec}))
    assert "# He was *not* force-fed." in rp.html and "<em>" not in rp.html
    mine = [c for c in rp.claims if c.correction == "mine"]
    assert [c.text for c in mine] == ["# He was *not* force-fed."]


def test_a_newer_strike_of_the_same_sentence_replaces_the_older_one():
    old = _strike(FORCE_FED, [A3], "He refused food for several weeks.")
    new = _strike(FORCE_FED, [A3], "He refused food for six weeks.", cid="c-0002")
    text, _, _ = _apply(TEXT, [old])
    # Run 2: c-0002 replaced c-0001, whose sentence a run carried out
    again, states, _ = cp.apply_to_text(text, cp._rules([new]), {"c-0001"}, [], [old])
    assert f"He refused food for six weeks.^[{A3}] Sources disagree" in again
    assert "several weeks" not in again and FORCE_FED not in again
    assert states == {"c-0002": ("applied", "")}
    # The same words written in both: nothing to swap, the newer one holds
    same = _strike(FORCE_FED, [A3], "He refused food for several weeks.", cid="c-0002")
    kept, states, _ = cp.apply_to_text(text, cp._rules([same]), {"c-0001"}, [], [old])
    assert kept == text and states == {"c-0002": ("held", "")}


def test_a_run_carries_out_the_newer_strike(tmp_path):
    path = tmp_path / "c.jsonl"
    person = {"name": "Yusuf Qadir", "type": "detainee",
              "articles": [{"article_id": a} for a in (A1, A2, A3)],
              "profile": {"text": TEXT}}  # fmt: skip
    entities = {"people": {"Yusuf Qadir": person}}
    for rep in ("He refused food for several weeks.", "He refused food for six weeks."):
        append_correction(str(path), {k: v for k, v in _strike(FORCE_FED, [A3], rep).items() if k != "id"})  # fmt: skip
        report = cp.run_corrections_pass(entities, str(path))
        assert report["counts"] == {"applied": 1}, report["rows"]
    text = person["profile"]["text"]
    assert "six weeks" in text and "several weeks" not in text
    report = cp.run_corrections_pass(entities, str(path))  # a third run: held
    assert report["counts"] == {"held": 1} and person["profile"]["text"] == text


def test_your_sentence_in_a_list_item_is_found_again():
    from src.frontend.profile_render import Overlay, render_profile

    listed = f"## Life\n\n- He was born in Taiz.^[{A1}]\n- He arrived in 2000.^[{A1}]\n- He left.^[{A2}]\n"  # fmt: skip
    # The reader copies a list item's claim with its marker
    rec = _strike("- He arrived in 2000.", [A1], "He arrived in 2001.")
    text, states, _ = _apply(listed, [rec])
    assert f"- He arrived in 2001.^[{A3}]\n- He left." in text
    assert states["c-0001"] == ("applied", "")
    again, states, _ = _apply(text, [rec], done=["c-0001"])
    assert again == text and states["c-0001"] == ("held", "")
    # The edit guard knows it is theirs
    edits = [ProfileEdit(op="replace", anchor="He arrived in 2001.", text="He came in 2003.")]  # fmt: skip
    token = cp._CURRENT.set(cp._rules([rec]))
    try:
        kept, _ = cp.guard_corrections(text, edits)
    finally:
        cp._CURRENT.reset(token)
    assert kept == []
    # The reader marks it as theirs and doesn't show it twice
    rp = render_profile(text, overlay=Overlay(struck={"x": rec}))
    assert [c.text for c in rp.claims if c.correction == "mine"] == [
        "He arrived in 2001."
    ]
    assert rp.html.count("He arrived in 2001.") == 1
