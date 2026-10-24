"""Tests for the sectioned match-check prompt and its reply parser."""

from unittest.mock import patch

import pytest

from src.engine import match_checker
from src.engine.match_checker import (
    MatchCheckResult,
    _build_sectioned_messages,
    local_model_check_match,
    parse_sectioned_reply,
)

REPLY = """[[ ## reasoning ## ]]
Both describe the same county; one adds "County".

[[ ## is_match ## ]]
True

[[ ## confidence ## ]]
0.92

[[ ## completed ## ]]"""


def test_parse_reads_all_three_fields():
    result = parse_sectioned_reply(REPLY)
    assert result.is_match is True
    assert result.confidence == pytest.approx(0.92)
    assert result.reason.startswith("Both describe the same county")


@pytest.mark.parametrize(
    "answer,expected", [("False", False), ("`false`", False), ("yes", True)]
)
def test_parse_accepts_answer_variants(answer, expected):
    text = REPLY.replace("True", answer)
    assert parse_sectioned_reply(text).is_match is expected


def test_parse_clamps_or_defaults_confidence():
    assert parse_sectioned_reply(REPLY.replace("0.92", "1.7")).confidence == 1.0
    assert parse_sectioned_reply(REPLY.replace("0.92", "high")).confidence == 0.5


def test_parse_rejects_missing_answer():
    with pytest.raises(ValueError):
        parse_sectioned_reply("[[ ## reasoning ## ]]\nThey look alike.")


def test_messages_put_each_input_and_the_rules_in_their_own_section():
    system, user = _build_sectioned_messages(
        "Miami-Dade", "Miami-Dade County", "new text", "old text", "events"
    )
    assert "same real-world event" in system["content"]
    for field in ("entity_type", "rules", "new_name", "existing_profile_text"):
        assert f"[[ ## {field} ## ]]" in user["content"]
    assert "Event-Specific Rules" in user["content"]
    assert "Event-Specific Rules" not in system["content"]


def test_check_uses_text_reply():
    with patch.object(match_checker, "text_generation", return_value=REPLY) as gen:
        result = local_model_check_match("a", "b", "x", "y", entity_type="locations")
    assert result.is_match is True
    assert gen.call_args.kwargs["local"] is True


def test_check_falls_back_to_structured_prompt_on_bad_reply():
    fallback = MatchCheckResult(is_match=False, confidence=0.8, reason="structured")
    with (
        patch.object(match_checker, "text_generation", return_value="no sections"),
        patch.object(match_checker, "local_generation", return_value=fallback),
    ):
        result = local_model_check_match("a", "b", "x", "y")
    assert result is fallback
