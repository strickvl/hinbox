"""Place and organization stubs: sentences that use the name only as a label.

On the 7 Nov 150-article run a "USS Cole" location stub held only sentences
about "the USS Cole case"; the match check read it as a court case and
rejected the ship. Such sentences now come last, and a stub made only of them
says so in an uncited line.
"""

from src.engine.profile_stubs import (
    build_stub_profile,
    mention_sentences,
    only_as_label,
)
from src.utils.quality_controls import CITATION_RE

ARTICLE = (
    "Lawyers filed an appeal of the now stalled USS Cole death-penalty case. "
    "The judge said the USS Cole case judge lacked authority. "
    "The USS Cole was bombed in Aden harbor in October 2000. "
    "Pentagon officials declined to comment. "
    "The Pentagon said it would review the ruling."
)


def test_only_as_label():
    assert only_as_label("the stalled USS Cole death-penalty case", ["USS Cole"])
    assert only_as_label("Guantánamo detainees were moved.", ["Guantánamo"])
    assert not only_as_label("The USS Cole was bombed in Aden.", ["USS Cole"])
    # One label use and one real use: the sentence names the entity
    assert not only_as_label(
        "The USS Cole case concerns the attack on the USS Cole in Aden.", ["USS Cole"]
    )
    assert not only_as_label("No mention here.", ["USS Cole"])


def test_label_sentences_come_after_real_mentions():
    picked = mention_sentences(ARTICLE, "locations", ["USS Cole"], limit=2)
    assert picked[0].startswith("Lawyers filed")  # article order kept
    assert any("bombed in Aden" in s for s in picked)
    assert not any("case judge" in s for s in picked)


def test_people_and_events_are_unchanged():
    text = "The Trump administration said nothing. Trump officials agreed."
    assert mention_sentences(text, "people", ["Trump"]) == [
        "The Trump administration said nothing.",
        "Trump officials agreed.",
    ]


def test_organization_label_sentence_demoted():
    picked = mention_sentences(ARTICLE, "organizations", ["Pentagon"], limit=1)
    assert picked == ["The Pentagon said it would review the ruling."]


def test_all_label_stub_gets_an_uncited_note_after_the_mentions():
    article = (
        "An appeal of the now stalled USS Cole death-penalty case is pending. "
        "Judge Lamberth is deciding the contempt authority of the USS Cole case judge."
    )
    stub = build_stub_profile(
        "locations", ("USS Cole", "other"), {"name": "USS Cole"}, article, "a1"
    )
    text = stub["text"]
    assert text.endswith(
        "(The article mentions USS Cole only to describe something else, as in "
        '"USS Cole death-penalty case".)'
    )
    last_block = text.split("\n\n")[-1]
    assert not CITATION_RE.search(last_block)


def test_no_note_when_a_sentence_names_the_place():
    stub = build_stub_profile(
        "locations", ("USS Cole", "other"), {"name": "USS Cole"}, ARTICLE, "a1"
    )
    assert "only to describe" not in stub["text"]


def test_note_is_not_a_grounding_claim():
    from src.utils.grounding import _extract_cited_claims

    article = "An appeal of the now stalled USS Cole death-penalty case is pending."
    stub = build_stub_profile(
        "locations", ("USS Cole", "other"), {"name": "USS Cole"}, article, "a1"
    )
    claims = _extract_cited_claims(stub["text"])
    assert len(claims) == 1
    assert "only to describe" not in str(claims)
