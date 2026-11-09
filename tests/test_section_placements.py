"""The replay placement report (scripts/section_placements.py)."""

import json

from scripts import section_placements

FLAT = "### Legal\n\nHe was charged in 2008.^[a1]\n"
DONE = (
    "### Background\n\nHe was born in Mecca.^[a0]\n\n"
    "### Military commission or legal case\n\nHe was charged in 2008.^[a1]\n"
)
NEXT = (
    "### Background\n\nHe was born in Mecca.^[a0]\n\n"
    "### Detention at Guantánamo\n\nHe was moved to Camp 7.^[a3]\n\n"
    "### Military commission or legal case\n\nHe was charged in 2008.^[a1]\n\n"
    "#### Plea talks\n\nTalks began in 2022.^[a3]\n"
)


def _step(n, text, mark=None):
    profile = {"text": text} | ({"reorganised": mark} if mark else {})
    return {"step": n, "article": {"title": f"T{n}"}, "profile": profile}


def test_counts_where_sentences_added_after_the_reorganise_land(tmp_path):
    entity = tmp_path / "replay" / "al-nashiri"
    entity.mkdir(parents=True)
    steps = [_step(1, FLAT), _step(2, DONE, "done"), _step(3, NEXT, "done")]
    for s in steps:
        (entity / f"step_{s['step']:03d}.json").write_text(json.dumps(s))
    section_placements.main([str(tmp_path / "replay"), "--out", str(tmp_path / "o")])
    summary = json.loads((tmp_path / "o" / "summary.json").read_text())["al-nashiri"]
    assert summary["reorganised_at_step"] == 2
    assert summary["added_sentences"] == 2
    assert summary["added_by_section"] == {
        "Detention at Guantánamo": 1,
        "Military commission or legal case": 1,
    }
    assert summary["added_to_last_top_section"] == 1
    assert summary["headings_created_by_updates"] == [
        "Detention at Guantánamo",
        "Plea talks",
    ]
    assert summary["deepest_level_below_top"] == 1
    review = (tmp_path / "o" / "review.md").read_text()
    assert "**Military commission or legal case › Plea talks**: Talks began" in review


def test_abbreviations_and_initials_do_not_cut_a_sentence():
    old = "### Legal\n\nHe was charged.^[a1]\n"
    new = (
        "### Legal\n\nHe was charged.^[a1] Rear Adm. John F. Smith ruled for the"
        " U.S. government.^[a2] Gen. Mark Martins agreed.^[a3]\n"
    )
    assert section_placements.added_sentences(old, new) == [
        "Rear Adm. John F. Smith ruled for the U.S. government.^[a2]",
        "Gen. Mark Martins agreed.^[a3]",
    ]
