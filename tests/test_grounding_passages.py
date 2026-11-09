"""Saved grounding passages: only the article's own words, found where the
judge quoted them, ever get stored or highlighted.

Every article, name and claim here is invented.
"""

from typing import List
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from src.engine.grounding_pass import backfill_passages
from src.process_and_extract import load_existing_entities
from src.utils import grounding
from src.utils.file_ops import write_entities_table
from src.utils.grounding import (
    ClaimVerdict,
    ClaimVerdictWithPassage,
    SupportLevel,
    grounding_is_current,
    verify_profile_grounding,
)
from src.utils.grounding_passages import (
    find_passage,
    passages_for,
    quotes_in,
    verified_passages,
)

ARTICLE = (
    "Officials said on Tuesday that fourteen men were flown to the base.\n"
    "Among them was Yusuf Qadir, a made-up detainee. He had studied\n"
    "engineering at a local college. “We will appeal,” his lawyer said."
)
PROFILE = (
    "Yusuf Qadir was flown to the base with thirteen other men.^[a1] "
    "He studied law.^[a1]"
)


# ── finding a quote in the article ──────────────────────────
def test_find_passage_forgives_case_spaces_and_quote_marks_only():
    span = find_passage(ARTICLE, "he had studied engineering at a local college")
    assert span and ARTICLE[span[0] : span[1]] == (
        "He had studied\nengineering at a local college"
    )
    # Straight quotes and a different case still find the curly-quoted words
    # (the quote marks at either end are trimmed off before the lookup)
    span = find_passage(ARTICLE, '"We will appeal," His Lawyer said')
    assert ARTICLE[span[0] : span[1]] == "We will appeal,” his lawyer said"
    # One word different is not the same passage
    assert find_passage(ARTICLE, "fifteen men were flown to the base") is None
    # A match must start and end on word edges
    assert find_passage("Iran today said so", "ran to") is None


def test_quotes_in_reasoning():
    why = (
        "The article says “fourteen men were flown to the base” and "
        "'Qadir's lawyer' is not named; it adds \"Among them was ... a made-up "
        'detainee".'
    )
    assert quotes_in(why) == [
        "fourteen men were flown to the base",
        "Qadir's lawyer",
        "Among them was",
        "a made-up detainee",
    ]


def test_verified_passages_keep_only_real_long_quotes():
    kept = verified_passages(
        ARTICLE,
        [
            "Fourteen men were flown to the base",
            "men were flown to the base",  # inside the first: not repeated
            "Qadir was flown to Cuba in 2006",  # not in the article
            "the base",  # too short to place for certain
        ],
    )
    assert kept == ["fourteen men were flown to the base"]
    # The article's own characters are stored, not the model's spelling
    assert passages_for(ARTICLE, None, "FOURTEEN MEN WERE FLOWN TO THE BASE.") == [
        "fourteen men were flown to the base"
    ]
    assert passages_for(ARTICLE, "The article says nothing about law.") is None


# ── the grounding step stores them ──────────────────────────
def _reply(passage: str = "") -> List[ClaimVerdict]:
    why = 'The article says "fourteen men were flown to the base".'
    two = "Article says he studied engineering, not law."
    if passage:
        return [
            ClaimVerdictWithPassage(
                claim_number=1,
                reasoning=why,
                support_level=SupportLevel.SUPPORTED,
                passage=passage,
            ),
            ClaimVerdictWithPassage(
                claim_number=2,
                reasoning=two,
                support_level=SupportLevel.NOT_SUPPORTED,
                passage="He studied law at Harvard.",  # invented
            ),
        ]
    return [
        ClaimVerdict(
            claim_number=1, reasoning=why, support_level=SupportLevel.SUPPORTED
        ),
        ClaimVerdict(
            claim_number=2, reasoning=two, support_level=SupportLevel.NOT_SUPPORTED
        ),
    ]


def test_quotes_in_reasoning_are_stored_and_prompt_is_unchanged_by_default():
    with patch("src.utils.llm.cloud_generation", return_value=_reply()) as gen:
        report = verify_profile_grounding(
            profile_text=PROFILE,
            article_texts={"a1": ARTICLE},
            entity_name="Yusuf Qadir",
        )
    kwargs = gen.call_args.kwargs
    assert kwargs["messages"][0]["content"] == grounding._GROUNDING_SYSTEM_PROMPT
    assert kwargs["response_model"] == List[ClaimVerdict]
    first, second = report.verifications
    assert first.passages == ["fourteen men were flown to the base"]
    assert second.passages is None


def test_copied_passage_is_kept_only_when_it_is_in_the_article(monkeypatch):
    monkeypatch.setattr(grounding, "GROUNDING_PASSAGE", True)
    copied = "Among them was Yusuf Qadir, a made-up detainee"
    with patch("src.utils.llm.cloud_generation", return_value=_reply(copied)) as gen:
        report = verify_profile_grounding(
            profile_text=PROFILE,
            article_texts={"a1": ARTICLE},
            entity_name="Yusuf Qadir",
        )
    kwargs = gen.call_args.kwargs
    assert "copy into passage" in kwargs["messages"][0]["content"]
    assert kwargs["response_model"] == List[ClaimVerdictWithPassage]
    first, second = report.verifications
    assert first.passages == [copied, "fourteen men were flown to the base"]
    assert second.passages is None  # the invented passage is dropped
    assert second.support_level == SupportLevel.NOT_SUPPORTED


def test_quoted_stub_sentence_gets_its_passage_without_a_call():
    text = "Among them was Yusuf Qadir, a made-up detainee.^[a1]"
    with patch("src.utils.llm.cloud_generation") as gen:
        report = verify_profile_grounding(
            profile_text=text, article_texts={"a1": ARTICLE}, entity_name="Qadir"
        )
    gen.assert_not_called()
    assert report.verifications[0].passages == [
        "Among them was Yusuf Qadir, a made-up detainee"
    ]


def test_reused_verdicts_from_before_passages_gain_them_without_a_call():
    with patch("src.utils.llm.cloud_generation", return_value=_reply()):
        report = verify_profile_grounding(
            profile_text=PROFILE,
            article_texts={"a1": ARTICLE},
            entity_name="Yusuf Qadir",
        )
    old = report.model_dump(exclude_none=True)
    for v in old["verifications"]:
        v.pop("passages", None)
    with patch("src.utils.llm.cloud_generation") as gen:
        again = verify_profile_grounding(
            profile_text=PROFILE + " He was held.^[a1]",
            article_texts={"a1": ARTICLE},
            entity_name="Yusuf Qadir",
            previous=old,
        )
    assert gen.call_count == 1  # only the new sentence
    assert again.verifications[0].passages == ["fourteen men were flown to the base"]


# ── filling in a finished run ───────────────────────────────
def _stored(article_texts, tmp_path):
    with patch("src.utils.llm.cloud_generation", return_value=_reply()):
        report = verify_profile_grounding(
            profile_text=PROFILE,
            article_texts=article_texts,
            entity_name="Yusuf Qadir",
        ).model_dump(exclude_none=True)
    for v in report["verifications"]:
        v.pop("passages", None)
    return {
        "people": {
            "Yusuf Qadir": {
                "name": "Yusuf Qadir",
                "profile": {"text": PROFILE},
                "profile_grounding": report,
            }
        }
    }


def test_backfill_adds_passages_and_keeps_reports_current(tmp_path):
    texts = {"a1": ARTICLE}
    entities = _stored(texts, tmp_path)
    before = [
        {k: v for k, v in d.items()}
        for d in entities["people"]["Yusuf Qadir"]["profile_grounding"]["verifications"]
    ]
    counts = backfill_passages(entities, texts)
    assert counts["added"] == 1 and counts["no_quote_found"] == 1
    report = entities["people"]["Yusuf Qadir"]["profile_grounding"]
    after = report["verifications"]
    assert after[0]["passages"] == ["fourteen men were flown to the base"]
    for old, new in zip(before, after):
        new = {k: v for k, v in new.items() if k != "passages"}
        assert new == old  # verdicts and reasoning untouched
    assert grounding_is_current(report, PROFILE, texts)

    # Saved with a table where other rows have no passages, and read back
    out = tmp_path / "out"
    out.mkdir()
    rows = list(entities["people"].values()) + [{"name": "Nobody"}]
    write_entities_table("people", rows, str(out))
    for t in ("events", "locations", "organizations"):
        write_entities_table(t, [], str(out))
    loaded = load_existing_entities(str(out))["people"]["Yusuf Qadir"]
    verifs = loaded["profile_grounding"]["verifications"]
    assert list(verifs[0]["passages"]) == ["fourteen men were flown to the base"]
    assert not verifs[1].get("passages")


def test_backfill_skips_an_article_whose_text_changed(tmp_path):
    entities = _stored({"a1": ARTICLE}, tmp_path)
    counts = backfill_passages(entities, {"a1": ARTICLE.replace("Tuesday", "Monday")})
    assert counts["added"] == 0 and counts["article_changed"] == 2


def test_backfill_script_writes_a_copy(tmp_path, monkeypatch):
    import sys

    from scripts import grounding_passages as script

    run = tmp_path / "run"
    run.mkdir()
    entities = _stored({"a1": ARTICLE}, tmp_path)
    write_entities_table("people", list(entities["people"].values()), str(run))
    for t in ("events", "locations", "organizations"):
        write_entities_table(t, [], str(run))
    arts = tmp_path / "articles.parquet"
    pq.write_table(pa.Table.from_pylist([{"id": "a1", "content": ARTICLE}]), arts)
    out = tmp_path / "copy"
    original = (run / "people.parquet").read_bytes()
    monkeypatch.setattr(
        sys,
        "argv",
        ["x", "--entities", str(run), "--articles", str(arts), "--out", str(out)],
    )
    script.main()
    assert (run / "people.parquet").read_bytes() == original
    loaded = load_existing_entities(str(out))["people"]["Yusuf Qadir"]
    verifs = loaded["profile_grounding"]["verifications"]
    assert list(verifs[0]["passages"]) == ["fourteen men were flown to the base"]


def test_backfill_gives_a_quoted_stub_sentence_its_place(tmp_path):
    text = "Among them was Yusuf Qadir, a made-up detainee.^[a1]"
    with patch("src.utils.llm.cloud_generation"):
        report = verify_profile_grounding(
            profile_text=text, article_texts={"a1": ARTICLE}, entity_name="Qadir"
        ).model_dump(exclude_none=True)
    report["verifications"][0].pop("passages")
    entities = {"people": {"Q": {"profile_grounding": report}}}
    assert backfill_passages(entities, {"a1": ARTICLE})["added"] == 1
    assert report["verifications"][0]["passages"] == [
        "Among them was Yusuf Qadir, a made-up detainee"
    ]


def test_find_passage_after_a_letter_that_lowercases_to_two():
    text = "İstanbul: fourteen men were flown to the base."
    span = find_passage(text, "fourteen men were flown to the base.")
    assert text[span[0] : span[1]] == "fourteen men were flown to the base"


def test_single_quoted_sentence_keeps_its_apostrophes():
    why = (
        "The article says 'Hamdan's next hearing won't be held until May' and "
        "‘the judge’s order isn’t final’; it says nothing else."
    )
    assert quotes_in(why) == [
        "Hamdan's next hearing won't be held until May",
        "the judge’s order isn’t final",
    ]
    # A possessive after a closing quote doesn't swallow the rest
    assert quotes_in("It quotes 'the base' and Qadir's lawyer.") == ["the base"]


def test_a_phrase_the_article_has_twice_is_not_kept():
    article = (
        "He was moved to the maximum-security Camp 7 in 2006.\n"
        "Lawyers have never seen the maximum-security Camp 7."
    )
    assert verified_passages(article, ["the maximum-security Camp 7"]) == []
    assert verified_passages(article, ["moved to the maximum-security Camp 7"]) == [
        "moved to the maximum-security Camp 7"
    ]
