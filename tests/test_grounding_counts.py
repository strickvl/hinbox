"""Claim counts per verdict, shown beside the grounding score.

The score counts "partial" as a pass, so a profile with one wrong claim and
nine partly wrong ones still scores 0.9. These tests check that the counts
behind it are stored and reported.
"""

from unittest.mock import patch

from src.engine.grounding_pass import stored_claim_totals
from src.utils.grounding import (
    ClaimVerdict,
    SupportLevel,
    claim_counts,
    describe_claim_counts,
    verify_profile_grounding,
)

LEVELS = [
    SupportLevel.SUPPORTED,
    SupportLevel.PARTIAL,
    SupportLevel.NOT_SUPPORTED,
    SupportLevel.UNCLEAR,
]


def _one_of_each(*, messages, response_model, temperature, **kw):
    lines = messages[1]["content"].split("## CLAIMS TO VERIFY")[1].splitlines()
    numbers = [int(ln.split(".")[0]) for ln in lines if ln[:1].isdigit()]
    return [ClaimVerdict(claim_number=n, support_level=LEVELS[n - 1]) for n in numbers]


def test_report_stores_counts_per_verdict():
    text = "Alice was held.^[a1] She left in 2010.^[a1] She met Bob.^[a1] Odd.^[a1]"
    with patch("src.utils.llm.cloud_generation", side_effect=_one_of_each):
        report = verify_profile_grounding(
            profile_text=text, article_texts={"a1": "An unrelated article."}
        )
    assert (report.supported, report.partial) == (1, 1)
    assert (report.not_supported, report.unclear) == (1, 1)
    assert report.grounding_score == 0.5  # partial passes, unclear fails
    assert claim_counts(report)["not_supported"] == 1


def test_counts_come_from_verifications_for_older_stored_reports():
    stored = {  # a report saved before the count fields existed
        "grounding_score": 0.75,
        "verifications": [
            {"support_level": "supported"},
            {"support_level": "partial"},
            {"support_level": "partial"},
            {"support_level": "not_supported"},
        ],
    }
    assert claim_counts(stored) == {
        "supported": 1,
        "partial": 2,
        "not_supported": 1,
        "unclear": 0,
        "missing_source": 0,
    }
    assert claim_counts(None)["supported"] == 0


def test_describe_lists_what_the_score_hides():
    counts = {"supported": 30, "partial": 5, "not_supported": 2, "unclear": 1}
    assert describe_claim_counts(counts) == (
        "2 not supported, 5 partial, 1 unclear, 0 missing source of 38 claims"
    )


def test_totals_over_all_stored_reports():
    def report(*levels):
        return {
            "profile_grounding": {
                "verifications": [{"support_level": x} for x in levels]
            }
        }

    entities = {
        "people": {"a": report("supported", "partial"), "b": report("supported")},
        "events": {"c": report("not_supported"), "d": {}},
    }
    totals = stored_claim_totals(entities)
    assert totals["claims"] == 4
    assert totals["not_supported"] == 1 and totals["partial"] == 1
    assert totals["profiles_with_problems"] == 2
