"""Filing an update's new sentences under their section's sub-headings
(src/engine/section_subplace.py) and scoring it (src/evals/subplacement.py).

Scenario: Obama's profile has "## Cases and decisions" with the sub-headings
"Guantanamo detention policy" and "Military commissions". An update anchors
"Trump revoked the closure order in 2018" on the parent heading, so it lands
above both sub-headings; the sub-heading call files it under the detention
policy, and code moves it there word for word.
"""

import json
import re
from typing import Any, Callable, Dict, List

from src.engine import section_place
from src.engine.opening_sort import _claims
from src.engine.section_place import STAY, Placement, Placements
from src.engine.section_split import _sentences
from src.engine.section_subplace import (
    heading_at,
    outline,
    place_new_subs,
    sub_place_enabled,
    subplace_profile,
)
from src.evals import subplacement

RULES = [
    ("closure", "Guantanamo detention policy"),
    ("transfer", "Guantanamo detention policy"),
    ("commission", "Military commissions"),
    ("court", STAY),
]

OLD = (
    "Barack Obama was the 44th U.S. president.^[a0]\n\n"
    "## Role\n\nHe was president from 2009 to 2017.^[a0]\n\n"
    "## Cases and decisions\n\n"
    "He fought many legal fights.^[a1]\n\n"
    "### Guantanamo detention policy\n\n"
    "He signed an order to close the prison in 2009.^[a1]\n\n"
    "### Military commissions\n\n"
    "He revived the military commissions in 2009.^[a2]\n"
)


def _asker(seen: List[Any], fail: bool = False) -> Callable:
    """Files each numbered sentence by a keyword; copies its first words."""

    def ask(msgs: List[Dict[str, str]]) -> Placements:
        seen.append(msgs)
        if fail:
            raise RuntimeError("down")
        out = []
        for n, sentence in re.findall(r"^\[(\d+)\] (.*)$", msgs[1]["content"], re.M):
            heading = next(
                (h for k, h in RULES if k in sentence), "Military commissions"
            )
            starts = " ".join(sentence.split()[:3])
            out.append(
                Placement(
                    sentence=int(n), starts_with=starts, about="", heading=heading
                )
            )
        return Placements(placements=out)

    return ask


def _add(text: str, after: str, piece: str) -> str:
    assert text.count(after) == 1
    return text.replace(after, after + "\n\n" + piece)


def test_a_new_piece_above_the_sub_headings_moves_into_the_one_it_fits():
    new = _add(OLD, "fights.^[a1]", "Trump revoked the closure order in 2018.^[a3]")
    seen: List[Any] = []
    out, report = place_new_subs(OLD, new, "person", "Obama", _asker(seen))
    assert report["status"] == "done"
    assert report["moved"] == {"Cases and decisions -> Guantanamo detention policy": 1}
    policy = out.split("### Guantanamo detention policy")[1].split("###")[0]
    assert "revoked the closure order" in policy
    assert _sentences(out) == _sentences(new)
    # The prompt offers this section's sub-headings and the way to stay above
    prompt = seen[0][0]["content"]
    assert "Guantanamo detention policy" in prompt and STAY in prompt
    assert "Role" not in prompt


def test_a_general_piece_in_a_sub_section_moves_above_the_sub_headings():
    new = _add(OLD, "commissions in 2009.^[a2]", "The court heard many cases.^[a3]")
    out, report = place_new_subs(OLD, new, "person", "Obama", _asker([]))
    assert report["moved"] == {"Military commissions -> Cases and decisions": 1}
    top = out.split("## Cases and decisions")[1].split("###")[0]
    assert "The court heard" in top
    assert _sentences(out) == _sentences(new)


def test_pieces_already_in_place_or_leaning_on_the_one_before_stay():
    new = _add(
        OLD, "close the prison in 2009.^[a1]",
        "A transfer ban followed in 2011.^[a3] However, a commission met.^[a3]",
    )  # fmt: skip
    seen: List[Any] = []
    out, report = place_new_subs(OLD, new, "person", "Obama", _asker(seen))
    assert out == new and report["status"] == "kept"
    assert "However" not in seen[0][1]["content"]


def test_nothing_new_under_sub_headings_or_a_failed_call_keeps_the_update():
    new = _add(OLD, "2009 to 2017.^[a0]", "He was a senator before.^[a3]")
    seen: List[Any] = []
    out, report = place_new_subs(OLD, new, "person", "Obama", _asker(seen))
    assert (out, report["status"], seen) == (new, "skipped", [])
    new = _add(OLD, "fights.^[a1]", "Trump revoked the closure order.^[a3]")
    out, report = place_new_subs(OLD, new, "person", "Obama", _asker([], fail=True))
    assert (out, report["status"]) == (new, "failed")


def test_a_moved_piece_goes_after_the_last_citation_so_claims_stay():
    # Uncited text at the end of the detention sub-section belongs to the
    # next citation's claim: the piece goes before it, not after it
    old = OLD.replace(
        "close the prison in 2009.^[a1]",
        "close the prison in 2009.^[a1] It never closed.",
    )
    new = _add(old, "fights.^[a1]", "Trump revoked the closure order.^[a3]")
    out, report = place_new_subs(old, new, "person", "Obama", _asker([]))
    assert report["status"] == "done" and report["refused"] == 0
    assert "2009.^[a1]\n\nTrump revoked the closure order.^[a3]\n\nIt never" in out
    assert _claims(out) == _claims(new)


def test_the_outline_and_where_a_position_sits():
    assert outline(OLD) == {
        "Role": [],
        "Cases and decisions": ["Guantanamo detention policy", "Military commissions"],
    }
    deeper = OLD + "\n#### Detail\n\nMore.^[a2]\n"
    assert "Cases and decisions" not in outline(deeper)
    at = OLD.index("revived")
    assert heading_at(OLD, at) == ("Cases and decisions", "Military commissions")
    assert heading_at(OLD, OLD.index("fights")) == ("Cases and decisions", None)


def test_sub_headings_are_offered_by_name_alone():
    new = _add(OLD, "fights.^[a1]", "Trump revoked the closure order.^[a3]")
    seen: List[Any] = []
    place_new_subs(OLD, new, "person", "Obama", _asker(seen))
    listed = seen[0][0]["content"]
    assert "- Guantanamo detention policy\n" in listed
    assert "He signed an order" not in listed


def test_only_reorganised_profiles_and_the_setting(monkeypatch):
    new = _add(OLD, "fights.^[a1]", "Trump revoked the closure order.^[a3]")
    seen: List[Any] = []
    monkeypatch.setattr(section_place, "make_asker", lambda *a, **k: _asker(seen))
    flat = {"text": new}
    assert subplace_profile(flat, OLD, "person", "Obama", "m", True) == (flat, None)
    done = {"text": new, "reorganised": "done"}
    out, report = subplace_profile(done, OLD, "person", "Obama", "m", True)
    assert report["status"] == "done" and out["text"] != new
    monkeypatch.setenv("HINBOX_SUB_PLACE", "local")
    assert sub_place_enabled(True) and not sub_place_enabled(False)
    monkeypatch.setenv("HINBOX_SUB_PLACE", "off")
    assert subplace_profile(done, OLD, "person", "Obama", "m", True)[1] is None


def _label(**kw: Any) -> Dict[str, Any]:
    row = {
        "entity": "Barack Obama", "entity_type": "people", "kind": "government",
        "sentence": "Trump revoked the closure order in 2018.^[a3]",
        "top_heading": "Cases and decisions",
        "subheadings": ["Guantanamo detention policy", "Military commissions"],
        "placed": "Cases and decisions", "gold": "Guantanamo detention policy",
        "confidence": "sure",
    }  # fmt: skip
    return {**row, **kw}


def test_the_eval_reads_gold_and_placed_in_the_same_terms(tmp_path):
    assert subplacement.gold_answer(_label()) == "Guantanamo detention policy"
    assert subplacement.gold_answer(_label(gold="(above)")) == STAY
    assert subplacement.gold_answer(_label(gold="(none fits)")) == STAY
    assert subplacement.gold_answer(_label(gold="Elsewhere")) is None
    assert subplacement.placed_answer(_label()) == STAY
    placed = "Cases and decisions > Military commissions"
    assert subplacement.placed_answer(_label(placed=placed)) == "Military commissions"
    assert subplacement.placed_answer(_label(placed="Role")) == "(other section)"
    path = tmp_path / "labels.jsonl"
    rows = [_label(), _label(confidence="unsure"), _label(gold="Elsewhere")]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    loaded, skipped = subplacement.load_labels(str(path))
    assert len(loaded) == 1 and skipped == {"unsure": 1, "bad_gold": 1}


def test_the_eval_scores_answers_against_gold_and_the_baseline():
    rows = [
        {**_label(), "id": "1", "gold": "Guantanamo detention policy"},
        {**_label(sentence="The court heard it.^[a3]"), "id": "2", "gold": STAY},
        {**_label(sentence="A commission met.^[a3]"), "id": "3",
         "gold": "Guantanamo detention policy"},
    ]  # fmt: skip
    groups = subplacement.batches(rows, 8)
    assert len(groups) == 1
    records = subplacement.score_batches(groups, _asker([]), 1)
    summary = subplacement.summarize(records)
    assert summary["right"] == 2 and summary["right_stay_gold"] == "1/1"
    assert summary["baseline"] == "1/3" and summary["after_step"] == "2/3"
    assert summary["mistakes"] == {"sub -> other sub": 1}
