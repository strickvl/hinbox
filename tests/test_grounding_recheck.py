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
    ClaimVerification,
    SupportLevel,
    _extract_cited_claims,
    grounding_is_current,
    verify_profile_grounding,
)
from src.utils.quality_controls import run_profile_qc


def _judge(level: SupportLevel = SupportLevel.SUPPORTED):
    """A fake model: one verdict per claim sent, with a wrong article ID
    (the code must not trust the model's copy of it)."""

    def fake(*, messages, response_model, temperature):
        claims = [
            ln for ln in messages[1]["content"].splitlines() if ". Citation: " in ln
        ]
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
        def a2_fails(*, messages, response_model, temperature):
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
        def one_short(*, messages, response_model, temperature):
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
