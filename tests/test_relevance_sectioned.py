"""Tests for the sectioned relevance prompt."""

from unittest.mock import patch

from src.engine import relevance

REPLY = """[[ ## reasoning ## ]]
The article is about a detainee's military commission hearing.

[[ ## is_relevant ## ]]
True

[[ ## completed ## ]]"""


def test_messages_drop_the_json_output_format_and_hold_the_article():
    system, user = relevance._sectioned_messages("ARTICLE BODY", "guantanamo")
    assert "Relevance Criteria" in system["content"]
    assert "Output Format" not in system["content"]
    assert '"is_relevant": true' not in system["content"]
    assert "[[ ## article ## ]]\nARTICLE BODY" in user["content"]
    assert "`[[ ## is_relevant ## ]]` (True or False)" in user["content"]


def test_check_reads_the_text_reply():
    with patch.object(relevance, "text_generation", return_value=REPLY) as gen:
        result = relevance.ollama_check_relevance("text")
    assert result.is_relevant is True
    assert result.reason.startswith("The article is about")
    assert gen.call_args.kwargs["local"] is True


def test_check_falls_back_to_structured_prompt_on_bad_reply():
    with (
        patch.object(relevance, "text_generation", return_value="no sections"),
        patch.object(relevance, "extract_entities_cloud", return_value="fallback"),
    ):
        assert relevance.gemini_check_relevance("text") == "fallback"
