"""Check article relevance to configured domain topics."""

import re

from src.config_loader import get_system_prompt
from src.constants import CLOUD_MODEL, OLLAMA_MODEL
from src.dynamic_models import get_relevance_model
from src.logging_config import get_logger
from src.utils.extraction import (
    extract_entities_cloud,
    extract_entities_local,
)
from src.utils.llm import text_generation
from src.utils.sectioned_prompt import (
    build_sectioned_messages,
    parse_sections,
    read_bool,
)

logger = get_logger("engine.relevance")

# Part of the relevance fingerprint: bump it when the way the prompt is asked
# changes, so stored "not relevant" verdicts are checked again.
RELEVANCE_PROMPT_LAYOUT = "sectioned-v1"

# The domain prompt's own "## Output Format" section asks for JSON; the
# sectioned layout replaces it.
_OUTPUT_FORMAT_SECTION = re.compile(r"\n#+\s*Output Format\b.*", re.S | re.I)

_INPUTS = [("article", "the article text")]
_OUTPUTS = [
    ("reasoning", "step-by-step reasoning about the article against the criteria"),
    ("is_relevant", "True if the article is relevant, otherwise False"),
]


def _sectioned_messages(text: str, domain: str):
    instructions = _OUTPUT_FORMAT_SECTION.sub(
        "", get_system_prompt("relevance", domain)
    ).strip()
    return build_sectioned_messages(
        instructions,
        _INPUTS,
        {"article": text},
        _OUTPUTS,
        hints={"is_relevant": "True or False"},
    )


def _check_relevance(text: str, model: str, domain: str, *, local: bool):
    """Ask in labelled sections with written reasoning; fall back to the
    structured prompt when the reply has no readable yes/no."""
    ArticleRelevance = get_relevance_model(domain)
    try:
        reply = text_generation(
            _sectioned_messages(text, domain), model=model, local=local, temperature=0
        )
        fields = parse_sections(reply)
        return ArticleRelevance(
            is_relevant=read_bool(fields, "is_relevant"),
            reason=fields.get("reasoning", ""),
        )
    except Exception as e:
        logger.warning(
            "Sectioned relevance check failed (%s); using the structured prompt", e
        )
    extract = extract_entities_local if local else extract_entities_cloud
    return extract(
        text=text,
        system_prompt=get_system_prompt("relevance", domain),
        response_model=ArticleRelevance,
        model=model,
        temperature=0,
    )


def gemini_check_relevance(
    text: str,
    model: str = CLOUD_MODEL,
    domain: str = "guantanamo",
):
    """
    Check if an article is relevant to the domain using a cloud model.

    Args:
        text: The article text to check
        model: The cloud model to use
        domain: The domain configuration to use

    Returns:
        ArticleRelevance object with is_relevant flag and reason
    """
    return _check_relevance(text, model, domain, local=False)


def ollama_check_relevance(
    text: str,
    model: str = OLLAMA_MODEL,
    domain: str = "guantanamo",
):
    """
    Check if an article is relevant to the domain using Ollama.

    Args:
        text: The article text to check
        model: The Ollama model to use
        domain: The domain configuration to use

    Returns:
        ArticleRelevance object with is_relevant flag and reason
    """
    return _check_relevance(text, model, domain, local=True)
