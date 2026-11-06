"""Grouped citations and which stored grounding results a later run keeps.

Scenario behind these tests: run 1 checks a profile, the provider times out
on one article, and the report is saved. Run 2 sees the same profile text.
Before, it skipped the profile because the text hash matched, so the timeout
stayed in the report forever. Now it re-asks the model for that one article
and keeps the other articles' results.
"""

from typing import Dict, List
from unittest.mock import patch

from src.utils.grounding import (
    GROUNDING_MAX_TOKENS,
    ClaimVerdict,
    ClaimVerification,
    SupportLevel,
    _extract_cited_claims,
    finish_report,
    grounding_is_current,
    judge_pending,
    plan_grounding,
    verify_profile_grounding,
)
from src.utils.quality_controls import run_profile_qc


def _judge(level: SupportLevel = SupportLevel.SUPPORTED):
    """A fake model: one verdict per claim sent, with a wrong article ID
    (the code must not trust the model's copy of it)."""

    def fake(*, messages, response_model, temperature, **kw):
        sent = messages[1]["content"].split("## CLAIMS TO VERIFY")[1]
        claims = [ln for ln in sent.splitlines() if ln[:1].isdigit()]
        return [
            ClaimVerification(
                article_id="model-typo", citation="?", claim="?", support_level=level
            )
            for _ in claims
        ]

    return fake


def _run(text: str, articles: Dict[str, str], previous=None, side_effect=None):
    with patch(
        "src.utils.llm.cloud_generation", side_effect=side_effect or _judge()
    ) as mock:
        report = verify_profile_grounding(
            profile_text=text, article_texts=articles, previous=previous
        )
    return report, mock.call_count


class TestGroupedCitations:
    def test_grouped_marker_gives_one_claim_per_id(self):
        claims = _extract_cited_claims("He was moved in 2016.^[a1, a2]")
        assert [c["article_id"] for c in claims] == ["a1", "a2"]
        assert claims[0]["claim"] == claims[1]["claim"] == "He was moved in 2016."
        assert claims[0]["citation"] == "^[a1, a2]"

    def test_grouped_marker_without_space(self):
        claims = _extract_cited_claims("Claim.^[a1,a2]")
        assert [c["article_id"] for c in claims] == ["a1", "a2"]

    def test_grouped_citation_is_checked_against_both_articles(self):
        report, calls = _run("Claim.^[a1, a2]", {"a1": "Text one", "a2": "Text two"})
        assert calls == 2
        assert report.total_citations == 2
        assert report.verified == 2
        assert report.missing_source == 0

    def test_profile_qc_counts_grouped_ids(self):
        _, qc = run_profile_qc(
            profile={"text": "x" * 200 + "^[a1, a2]", "tags": ["a"], "confidence": 0.5},
            min_text_len=10,
            min_tags=1,
        )
        assert qc.citation_count == 2
        assert "no_citations" not in qc.flags

    def test_punctuation_after_marker_is_not_a_claim(self):
        claims = _extract_cited_claims("He was held.^[a1].^[a2] Then freed^[a3].")
        assert [c["claim"] for c in claims] == [
            "He was held.",
            "He was held.",
            "Then freed",
        ]

    def test_leading_punctuation_dropped_from_next_claim(self):
        claims = _extract_cited_claims("Held in 2002^[a1]. Freed in 2016^[a2].")
        assert claims[1]["claim"] == "Freed in 2016"


class TestReuseEarlierResults:
    ARTICLES = {"a1": "Article one text", "a2": "Article two text"}
    TEXT = "First claim.^[a1] Second claim.^[a2]"

    def test_rerun_with_nothing_changed_makes_no_calls(self):
        first, _ = _run(self.TEXT, self.ARTICLES)
        second, calls = _run(self.TEXT, self.ARTICLES, previous=first.model_dump())
        assert calls == 0
        assert second.reused_articles == 2
        assert second.verified == 2

    def test_verdicts_are_stored_under_the_cited_id(self):
        report, _ = _run(self.TEXT, self.ARTICLES)
        assert [v.article_id for v in report.verifications] == ["a1", "a2"]
        assert report.verifications[0].claim == "First claim."

    def test_failed_article_is_retried_alone(self):
        def a2_fails(*, messages, response_model, temperature, **kw):
            if "ID: a2" in messages[1]["content"]:
                raise RuntimeError("timeout")
            return _judge()(
                messages=messages, response_model=response_model, temperature=0
            )

        first, _ = _run(self.TEXT, self.ARTICLES, side_effect=a2_fails)
        assert "verification_error" in first.flags
        assert [s.status for s in first.sources] == ["checked", "error"]

        second, calls = _run(self.TEXT, self.ARTICLES, previous=first.model_dump())
        assert calls == 1  # only a2
        assert second.reused_articles == 1
        assert second.verified == 2
        assert "verification_error" not in second.flags

    def test_short_reply_is_retried(self):
        def one_short(*, messages, response_model, temperature, **kw):
            return []

        first, _ = _run(self.TEXT, self.ARTICLES, side_effect=one_short)
        assert all(s.status == "error" for s in first.sources)
        _, calls = _run(self.TEXT, self.ARTICLES, previous=first.model_dump())
        assert calls == 2

    def test_missing_source_checked_once_available(self):
        first, calls = _run(self.TEXT, {"a1": "Article one text"})
        assert calls == 1
        assert first.missing_source == 1
        second, calls = _run(self.TEXT, self.ARTICLES, previous=first.model_dump())
        assert calls == 1  # a2 only; a1 kept
        assert second.missing_source == 0

    def test_still_missing_source_costs_no_call(self):
        first, _ = _run(self.TEXT, {"a1": "Article one text"})
        _, calls = _run(
            self.TEXT, {"a1": "Article one text"}, previous=first.model_dump()
        )
        assert calls == 0

    def test_changed_claims_for_one_article_rechecks_only_it(self):
        first, _ = _run(self.TEXT, self.ARTICLES)
        edited = "First claim.^[a1] Second claim, now longer.^[a2]"
        second, calls = _run(edited, self.ARTICLES, previous=first.model_dump())
        assert calls == 1
        assert second.reused_articles == 1

    def test_changed_article_text_is_rechecked(self):
        first, _ = _run(self.TEXT, self.ARTICLES)
        changed = {"a1": "Article one text, corrected", "a2": "Article two text"}
        _, calls = _run(self.TEXT, changed, previous=first.model_dump())
        assert calls == 1


class TestGroundingIsCurrent:
    ARTICLES = {"a1": "Article one text", "a2": "Article two text"}
    TEXT = "First claim.^[a1] Second claim.^[a2]"

    def test_clean_report_is_current(self):
        report, _ = _run(self.TEXT, self.ARTICLES)
        assert grounding_is_current(report.model_dump(), self.TEXT, self.ARTICLES)

    def test_changed_text_is_not_current(self):
        report, _ = _run(self.TEXT, self.ARTICLES)
        assert not grounding_is_current(
            report.model_dump(), self.TEXT + " More.", self.ARTICLES
        )

    def test_error_is_not_current(self):
        report, _ = _run(self.TEXT, self.ARTICLES, side_effect=RuntimeError("down"))
        assert not grounding_is_current(report.model_dump(), self.TEXT, self.ARTICLES)

    def test_missing_source_now_available_is_not_current(self):
        report, _ = _run(self.TEXT, {"a1": "Article one text"})
        stored = report.model_dump()
        assert grounding_is_current(stored, self.TEXT, {"a1": "Article one text"})
        assert not grounding_is_current(stored, self.TEXT, self.ARTICLES)

    def test_old_report_with_error_flag_is_not_current(self):
        import hashlib

        old: Dict = {
            "profile_text_hash": hashlib.sha256(self.TEXT.encode()).hexdigest(),
            "total_citations": 2,
            "flags": ["verification_error"],
            "verifications": [],
        }
        assert not grounding_is_current(old, self.TEXT, self.ARTICLES)
        old["flags"] = []
        assert grounding_is_current(old, self.TEXT, self.ARTICLES)

    def test_old_report_that_misread_grouped_citation_is_not_current(self):
        import hashlib

        text = "Claim.^[a1, a2]"
        old: Dict = {
            "profile_text_hash": hashlib.sha256(text.encode()).hexdigest(),
            "total_citations": 0,  # the old parser skipped "a1, a2"
            "flags": ["no_citations"],
            "verifications": [],
        }
        assert not grounding_is_current(old, text, self.ARTICLES)

    def test_report_with_old_claim_text_is_not_current(self):
        """A report checked when the parser cut a different claim (". Freed")
        goes back for a recheck of that article only."""
        text = "Held in 2002^[a1]. Freed in 2016^[a2]."
        report, _ = _run(text, self.ARTICLES)
        stored = report.model_dump()
        assert grounding_is_current(stored, text, self.ARTICLES)
        stored["sources"][1]["claims_hash"] = "hash-of-old-claims"
        stored["verifications"][1]["claim"] = ". Freed in 2016"
        assert not grounding_is_current(stored, text, self.ARTICLES)
        _, calls = _run(text, self.ARTICLES, previous=stored)
        assert calls == 1

    def test_old_report_with_punctuation_claim_is_not_current(self):
        import hashlib

        text = "Held.^[a1].^[a2]"
        old: Dict = {
            "profile_text_hash": hashlib.sha256(text.encode()).hexdigest(),
            "total_citations": 2,
            "flags": [],
            "verifications": [
                {"article_id": "a1", "claim": "Held.", "support_level": "supported"},
                {"article_id": "a2", "claim": ".", "support_level": "not_supported"},
            ],
        }
        assert not grounding_is_current(old, text, self.ARTICLES)

    def test_parquet_round_trip_keeps_report_usable(self, tmp_path):
        import pyarrow as pa
        import pyarrow.parquet as pq

        report, _ = _run(self.TEXT, self.ARTICLES)
        rows: List[Dict] = [{"profile_grounding": report.model_dump(exclude_none=True)}]
        pq.write_table(pa.Table.from_pylist(rows), tmp_path / "t.parquet")
        stored = pq.read_table(tmp_path / "t.parquet").to_pylist()[0][
            "profile_grounding"
        ]
        assert grounding_is_current(stored, self.TEXT, self.ARTICLES)
        _, calls = _run(self.TEXT, self.ARTICLES, previous=stored)
        assert calls == 0


class TestOnlyNewClaimsGoToTheModel:
    """An update adds one sentence citing a1. Before, every a1 claim went back
    to the model; now only the new one does."""

    ARTICLES = {"a1": "Article one text", "a2": "Article two text"}
    TEXT = "First claim.^[a1] Second claim.^[a2]"

    def test_added_claim_is_sent_alone(self):
        first, _ = _run(self.TEXT, self.ARTICLES)
        edited = self.TEXT + " Third claim.^[a1]"
        with patch("src.utils.llm.cloud_generation", side_effect=_judge()) as mock:
            second = verify_profile_grounding(
                profile_text=edited,
                article_texts=self.ARTICLES,
                previous=first.model_dump(),
            )
        assert mock.call_count == 1
        sent = mock.call_args.kwargs["messages"][1]["content"]
        assert "Third claim." in sent and "First claim." not in sent
        assert second.verified == 3
        assert [s.status for s in second.sources] == ["checked", "checked"]
        assert grounding_is_current(second.model_dump(), edited, self.ARTICLES)

    def test_claims_from_a_failed_check_are_not_reused(self):
        first, _ = _run(self.TEXT, self.ARTICLES, side_effect=RuntimeError("down"))
        _, calls = _run(self.TEXT + " More.^[a1]", self.ARTICLES, first.model_dump())
        assert calls == 2


class TestBatchedCalls:
    TEXT = "One.^[a1] Two.^[a1] Three.^[a1] Four.^[a1] Five.^[a1]"

    def test_many_claims_are_split_over_calls(self):
        plan = plan_grounding(profile_text=self.TEXT, article_texts={"a1": "Text"})
        with patch("src.utils.llm.cloud_generation", side_effect=_judge()) as mock:
            calls = judge_pending([plan], {"a1": "Text"}, max_claims_per_call=2)
        assert calls == mock.call_count == 3
        assert finish_report(plan).verified == 5

    def test_verdicts_matched_by_claim_number(self):
        def reversed_reply(*, messages, **kw):
            return [
                ClaimVerdict(claim_number=2, support_level=SupportLevel.NOT_SUPPORTED),
                ClaimVerdict(claim_number=1, support_level=SupportLevel.SUPPORTED),
            ]

        report, _ = _run("Held.^[a1] Freed.^[a1]", {"a1": "Text"}, None, reversed_reply)
        levels = [v.support_level for v in report.verifications]
        assert levels == [SupportLevel.SUPPORTED, SupportLevel.NOT_SUPPORTED]

    def test_skipped_claim_marks_article_for_retry(self):
        def one_verdict(**kw):
            return [ClaimVerdict(claim_number=2, support_level=SupportLevel.SUPPORTED)]

        report, _ = _run("Held.^[a1] Freed.^[a1]", {"a1": "Text"}, None, one_verdict)
        assert report.verifications[0].support_level == SupportLevel.UNCLEAR
        assert report.verifications[1].support_level == SupportLevel.SUPPORTED
        assert report.sources[0].status == "error"
        assert "llm_count_mismatch" in report.flags

    def test_entity_name_is_shown_with_each_claim(self):
        plan = plan_grounding(
            profile_text="He was freed.^[a1]",
            article_texts={"a1": "Text"},
            entity_name="Omar Khadr",
        )
        with patch("src.utils.llm.cloud_generation", side_effect=_judge()) as mock:
            judge_pending([plan], {"a1": "Text"})
        sent = mock.call_args.kwargs["messages"][1]["content"]
        assert "1. (Omar Khadr) He was freed." in sent
        assert mock.call_args.kwargs["max_tokens"] == GROUNDING_MAX_TOKENS


class TestQuotedClaims:
    ARTICLE = (
        "The judge, Col. James Pohl, set the hearing for May 2013. Lawyers objected."
    )

    def test_sentence_copied_from_article_needs_no_call(self):
        text = "The judge, Col. James Pohl, set the hearing for May  2013.^[a1]"
        report, calls = _run(text, {"a1": self.ARTICLE})
        assert calls == 0
        assert report.verified == 1
        assert report.sources[0].status == "checked"

    def test_short_or_changed_claims_still_go_to_the_model(self):
        text = "Lawyers objected.^[a1] Pohl set the hearing for June 2013.^[a1]"
        _, calls = _run(text, {"a1": self.ARTICLE})
        assert calls == 1
