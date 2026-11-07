"""Sub-sections in profiles: where a heading-anchored edit lands, and that a
claim keeps its grounding verdict when a sub-heading is put in front of it.

Scenario: KSM's "### Legal proceedings" is split into "#### Charges" and
"#### Plea negotiations". An edit anchored on "### Legal proceedings" must
still land at the end of the whole section (after "Plea negotiations"), and
the next grounding pass must not send the moved sentences to the judge again.
"""

from typing import Dict
from unittest.mock import patch

from src.engine.profile_edits import ProfileEdit, apply_edits
from src.utils.grounding import (
    ClaimVerification,
    SupportLevel,
    _claim_key,
    _earlier_verdict,
    verify_profile_grounding,
)

FLAT = (
    "### Background\n\nKSM was born in Kuwait.^[a1]\n\n"
    "### Legal proceedings\n\nHe was charged in 2008.^[a1]\n\n"
    "Plea talks began in 2017.^[a2]\n\n"
    "### Health\n\nHe has a heart condition.^[a2]\n"
)
SPLIT = (
    "### Background\n\nKSM was born in Kuwait.^[a1]\n\n"
    "### Legal proceedings\n\n#### Charges\n\nHe was charged in 2008.^[a1]\n\n"
    "#### Plea negotiations\n\nPlea talks began in 2017.^[a2]\n\n"
    "### Health\n\nHe has a heart condition.^[a2]\n"
)
NEW = "A plea deal was approved in July 2024.^[a3]"


def _add(text: str, anchor: str) -> str:
    out, report = apply_edits(text, [ProfileEdit(op="add", anchor=anchor, text=NEW)])
    assert report["applied"] == 1 and report["appended"] == 0
    return out


def test_add_to_a_section_goes_after_its_sub_sections():
    out = _add(SPLIT, "### Legal proceedings")
    assert out.index(NEW) > out.index("Plea talks began in 2017")
    assert out.index(NEW) < out.index("### Health")


def test_add_to_a_sub_section_stays_inside_it():
    out = _add(SPLIT, "#### Charges")
    assert out.index("He was charged") < out.index(NEW) < out.index("#### Plea")


def test_heading_anchor_matches_whatever_its_level():
    # The model wrote "##" for the profile's "###": the add used to fall
    # through to the end of the profile, under "### Health"
    out = _add(FLAT, "## Legal proceedings")
    assert out.index("Plea talks began") < out.index(NEW) < out.index("### Health")


def test_unknown_heading_still_appends_at_the_end():
    edit = ProfileEdit(op="add", anchor="### Release", text=NEW)
    out, report = apply_edits(FLAT, [edit])
    assert report["appended"] == 1 and out.rstrip().endswith(NEW)


def test_claim_key_ignores_heading_lines_and_spacing():
    assert _claim_key("#### Plea negotiations\n\nPlea talks  began.") == (
        _claim_key("Plea talks began.")
    )
    assert _claim_key("Plea talks began.") != _claim_key("Plea talks ended.")


def _verdict(claim: str) -> ClaimVerification:
    return ClaimVerification(
        article_id="a1",
        citation="^[a1]",
        claim=claim,
        support_level=SupportLevel.SUPPORTED,
    )


def test_a_change_late_in_a_long_claim_is_not_reused():
    # Two 500-character claims that differ only in the year near the end
    old = "x " * 230 + "He was released in 2012."
    new = "x " * 230 + "He was released in 2016."
    known = {_claim_key(old): _verdict(old)}
    assert _earlier_verdict(new, known, 600) is None
    assert _earlier_verdict("#### Release\n\n" + old, known, 600) is not None


def test_a_stored_claim_cut_at_the_cap_matches_by_its_start():
    full = "y " * 400 + "End."
    stored = full[:600]
    known = {_claim_key(stored): _verdict(stored)}
    assert _earlier_verdict(full, known, 600) is not None
    assert _earlier_verdict("#### Part 1\n\n" + full, known, 600) is not None
    assert _earlier_verdict("z" + full, known, 600) is None


def test_removing_a_heading_that_brings_changed_words_into_view_is_rechecked():
    # The old claim's heading pushed "released in 2012" past the 600-character
    # cut, so the judge never saw the year. Without the heading the whole new
    # claim fits, year changed: the old verdict says nothing about it.
    body = "The detention continued. " * 23
    old = "### Legal proceedings\n\n" + body + "He was released in 2012."
    new = body + "He was released in 2016."
    stored = old[:600]
    known = {_claim_key(stored): _verdict(stored)}
    assert _earlier_verdict(new, known, 600) is None
    assert _earlier_verdict(new + " " + "w " * 100, known, 600) is None


ARTICLES: Dict[str, str] = {
    "a1": "KSM was born in Kuwait. He was charged in 2008.",
    "a2": "Plea talks began in 2017. He has a heart condition.",
}


def _judge(*, messages, response_model, temperature, **kw):
    sent = messages[1]["content"].split("## CLAIMS TO VERIFY")[1]
    n = len([ln for ln in sent.splitlines() if ln[:1].isdigit()])
    return [
        ClaimVerification(
            article_id="x", citation="?", claim="?", support_level=SupportLevel.PARTIAL
        )
        for _ in range(n)
    ]


def _ground(text: str, previous=None):
    with patch("src.utils.llm.cloud_generation", side_effect=_judge) as mock:
        report = verify_profile_grounding(
            profile_text=text, article_texts=ARTICLES, previous=previous
        )
    return report, mock.call_count


def test_split_section_keeps_every_grounding_verdict():
    first, calls = _ground(FLAT)
    assert calls == 2  # one call per article
    second, calls = _ground(SPLIT, previous=first.model_dump())
    assert calls == 0
    assert [v.support_level for v in second.verifications] == [
        v.support_level for v in first.verifications
    ]


def test_reworded_sentence_after_split_is_judged_again():
    first, _ = _ground(FLAT)
    reworded = SPLIT.replace("Plea talks began in 2017.", "Plea talks began in 2016.")
    _, calls = _ground(reworded, previous=first.model_dump())
    assert calls == 1  # only a2, which has the changed claim
