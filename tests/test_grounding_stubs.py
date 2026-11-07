"""Stub profiles and the grounding check's copied-sentence shortcut.

A stub profile is a few article sentences copied word for word, so the
grounding judge has nothing to check in them. Before, every stub still sent
its first sentence to the model: the claim was cut from the profile with the
"### Mentions" heading on it, and "### mentions khalid…" is nowhere in the
article. Sentences opening with a quote mark, or that the article wraps over
two lines, went to the model too.
"""

from unittest.mock import patch

from src.engine.profile_stubs import build_stub_profile
from src.utils.grounding import SupportLevel, verify_profile_grounding
from tests.test_grounding_recheck import _judge

ARTICLE = """Guantanamo war court news.

Khalid Shaikh Mohammed appeared in court on Monday at Camp Justice for a hearing.
"We will appeal this ruling," Mohammed's lawyer, David Nevin, told reporters afterwards.
The judge said Mohammed would be tried
in 2027 at the earliest. Lawyers objected."""


def _ground(text: str, name: str, article: str = ARTICLE):
    with patch("src.utils.llm.cloud_generation", side_effect=_judge()) as mock:
        report = verify_profile_grounding(
            profile_text=text, article_texts={"a1": article}, entity_name=name
        )
    return report, mock


def test_person_stub_needs_no_model_call():
    stub = build_stub_profile("people", "Khalid Shaikh Mohammed", {}, ARTICLE, "a1")
    assert stub["text"].startswith("### Mentions")
    report, mock = _ground(stub["text"], "Khalid Shaikh Mohammed")
    assert mock.call_count == 0
    assert report.total_citations == report.verified == 3
    assert {c.support_level for c in report.verifications} == {SupportLevel.SUPPORTED}


def test_heading_before_a_copied_sentence_is_ignored():
    text = "### Court\n\nThe judge said Mohammed would be tried in 2027 at the earliest.^[a1]"
    _, mock = _ground(text, "Khalid Shaikh Mohammed")
    assert mock.call_count == 0


def test_sentence_opening_with_a_quote_mark_is_a_quote():
    text = '"We will appeal this ruling," Mohammed\'s lawyer, David Nevin, told reporters afterwards.^[a1]'
    _, mock = _ground(text, "David Nevin")
    assert mock.call_count == 0


def test_quote_mark_does_not_make_part_of_a_sentence_a_quote():
    article = "Officials denied 'Omar Khadr was released from custody in 2012.'"
    text = "Omar Khadr was released from custody in 2012.^[a1]"
    _, mock = _ground(text, "Omar Khadr", article)
    assert mock.call_count == 1


def test_event_stub_summary_still_goes_to_the_model():
    """An event stub starts with the extracted description, which a model
    wrote: that one claim is still checked, the copied sentences are not."""
    event = {
        "description": "Mohammed was arraigned at Camp Justice in 2027.",
        "aliases": ["Camp Justice"],
    }
    stub = build_stub_profile(
        "events", "Mohammed hearing", event, ARTICLE, "a1", "alternative_titles"
    )
    report, mock = _ground(stub["text"], "Mohammed hearing")
    assert mock.call_count == 1
    sent = mock.call_args.kwargs["messages"][1]["content"]
    claims = sent.split("## CLAIMS TO VERIFY")[1]
    assert "arraigned" in claims and "appeared in court" not in claims
    assert report.total_citations == 2


def test_edited_stub_sentence_goes_to_the_model():
    """Once the updater changes a stub sentence it is no longer the article's
    words, so it is checked as before."""
    text = "### Mentions\n\nKhalid Shaikh Mohammed appeared in court on Tuesday at Camp Justice for a hearing.^[a1]"
    _, mock = _ground(text, "Khalid Shaikh Mohammed")
    assert mock.call_count == 1


def test_quoted_claim_the_article_rejects_goes_to_the_model():
    """The quote check must end where the article's sentence ends."""
    article = '"Omar Khadr was released from custody in 2012" is a false claim, officials said.'
    text = "Omar Khadr was released from custody in 2012.^[a1]"
    _, mock = _ground(text, "Omar Khadr", article)
    assert mock.call_count == 1


def test_claim_cut_at_a_comma_goes_to_the_model():
    article = (
        "Omar Khadr was released from custody in 2012, his critics claimed wrongly."
    )
    text = "Omar Khadr was released from custody in 2012.^[a1]"
    _, mock = _ground(text, "Omar Khadr", article)
    assert mock.call_count == 1
