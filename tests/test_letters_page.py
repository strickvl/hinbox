"""Tests for checking letters-to-the-editor pages letter by letter.

These tests verify:
- split_letters() finds each signed letter, including a signature spread
  over several lines (name, group, place) and bullet separators
- News text, bylines with "and" and pages with fewer than 3 letters are not
  treated as letters pages
- A later run rebuilds the kept letters from the stored relevance outcome
- On a letters page only the relevant letters reach extraction, and the
  trimmed text is what profiles and stubs see; no relevant letter means
  "not relevant"; a failed check on one letter fails the page
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.engine import ArticleProcessor
from src.process_and_extract import extract_single_article_only
from src.utils.letters_page import (
    signature_name,
    split_letters,
    stored_letters_text,
)
from src.utils.outcomes import PhaseOutcome

# Made-up letters in the layout of a Miami Herald letters page
PAGE = """Re the May 29 story about the hot pink islands in Biscayne Bay.

The islands were one of the best things that ever happened to the city.

Harold Example, Coconut Grove

I served my time in the military and wonder about the wreath ceremony.

Raymond Sample, Kendall

We congratulate the reporter for her coverage of the military commissions at Guantanamo Bay.

Phyllida Testcase,

Families for Peaceful Tomorrows,

White Plains, New York

▪

After reading the May 27 article about the hurricane chief, I would like to apply to run NASA.

Monique P. Demo, Miami Shores

This story was originally published May 30, 2018."""

ARGS = SimpleNamespace(force_reprocess=False, relevance_check=True, verbose=False)


class TestSplitLetters:
    def test_finds_each_letter_and_signer(self):
        letters = split_letters(PAGE)
        assert [x.signer for x in letters] == [
            "Harold Example",
            "Raymond Sample",
            "Phyllida Testcase",
            "Monique P. Demo",
        ]
        assert letters[0].body.startswith("Re the May 29 story")
        assert "Guantanamo" in letters[2].body

    def test_multi_line_signature_stays_with_its_letter(self):
        letter = split_letters(PAGE)[2]
        assert letter.signature == (
            "Phyllida Testcase,\nFamilies for Peaceful Tomorrows,\n"
            "White Plains, New York"
        )
        assert "▪" not in split_letters(PAGE)[3].text

    def test_short_trailing_text_is_dropped(self):
        assert all("originally published" not in x.text for x in split_letters(PAGE))

    def test_news_article_is_not_a_letters_page(self):
        news = (
            "Defense lawyer Jane Doe said the hearing was delayed, Miami.\n\n"
            "Patricia Mazzei, Maria Perez and Melissa Sanchez\n\n"
            "The judge, John Smith, ruled on Tuesday."
        )
        assert split_letters(news) == []

    def test_two_letters_are_not_enough(self):
        two = PAGE.split("Phyllida Testcase,")[0]
        assert split_letters(two) == []

    def test_signature_shapes(self):
        assert signature_name("Ray Turner, Kendall") == "Ray Turner"
        assert signature_name("Charles E. Hannemann, Palmetto Bay") == (
            "Charles E. Hannemann"
        )
        assert signature_name("Phyllis Rodriguez,") == "Phyllis Rodriguez"
        assert signature_name("Rolando Carol, Montgomery, Alabama") == "Rolando Carol"
        assert signature_name("Kendall") is None
        assert (
            signature_name("Mimi Whitefield, Daniel Chang and Andres Viglucci") is None
        )
        assert signature_name("However, the judge ruled against him.") is None
        assert signature_name("The judge, who was appointed in 2010, ruled.") is None


def _run(page: str, verdict):
    """Extract ``page`` with a relevance check that answers ``verdict(text)``."""
    processor = ArticleProcessor(domain="guantanamo", model_type="gemini")
    check = MagicMock(side_effect=lambda text, _id: verdict(text))
    extract = MagicMock(return_value={})
    with (
        patch.object(processor, "check_relevance", check),
        patch.object(processor, "extract_all_entities", extract),
    ):
        row = {"id": "letters-1", "title": "Pink islands", "content": page}
        result = extract_single_article_only(
            row, 1, processor, ARGS, status_snapshot={}
        )
    return result, processor, check, extract


def _about_guantanamo(text: str) -> PhaseOutcome:
    return PhaseOutcome.ok("relevance", value="Guantanamo" in text)


class TestLettersPageRelevance:
    def test_only_relevant_letters_are_extracted(self):
        result, processor, check, extract = _run(PAGE, _about_guantanamo)

        assert check.call_count == 4  # once per letter, not for the page
        assert result.was_extracted
        text = extract.call_args.args[0]
        assert "Phyllida Testcase" in text and "Guantanamo" in text
        assert "Raymond Sample" not in text and "NASA" not in text
        # Stubs and profiles read the same trimmed text
        assert result.article_info["content"] == text
        assert processor.article_texts["letters-1"] == text
        meta = result.processing_metadata["phase_outcomes"]["relevance"]["meta"]
        assert meta["kept_signers"] == ["Phyllida Testcase"]
        assert "Raymond Sample" in meta["dropped_signers"]

    def test_no_relevant_letter_means_not_relevant(self):
        no = lambda _text: PhaseOutcome.ok("relevance", value=False)  # noqa: E731
        result, _, _, extract = _run(PAGE, no)
        assert result.skip_reason == "not_relevant"
        extract.assert_not_called()

    def test_failed_letter_check_fails_the_page(self):
        def flaky(text: str) -> PhaseOutcome:
            if "NASA" in text:
                return PhaseOutcome.fail(
                    "relevance", error=RuntimeError("down"), fallback=False
                )
            return _about_guantanamo(text)

        result, _, _, extract = _run(PAGE, flaky)
        assert result.skip_reason == "relevance_error"
        extract.assert_not_called()

    def test_ordinary_article_is_checked_whole(self):
        article = "A hearing at Guantanamo was delayed on Tuesday."
        result, _, check, extract = _run(article, _about_guantanamo)
        assert check.call_count == 1
        assert extract.call_args.args[0] == article


class TestStoredLettersText:
    def test_later_run_rebuilds_the_kept_letters(self):
        result, processor, _, _ = _run(PAGE, _about_guantanamo)
        metadata = result.processing_metadata
        assert (
            stored_letters_text(PAGE, metadata)
            == (processor.article_texts["letters-1"])
        )

    def test_changed_or_ordinary_text_is_left_alone(self):
        result, _, _, _ = _run(PAGE, _about_guantanamo)
        changed = PAGE.replace("Kendall", "Doral")
        assert stored_letters_text(changed, result.processing_metadata) == changed
        assert stored_letters_text(PAGE, None) == PAGE
        assert stored_letters_text(PAGE, {"content_hash": "x"}) == PAGE
