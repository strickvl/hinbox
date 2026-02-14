"""Entity match checking functionality."""

from typing import Dict, Optional

from pydantic import BaseModel, Field

from src.constants import CLOUD_MODEL, OLLAMA_MODEL
from src.logging_config import get_logger, log
from src.utils.cache_utils import LRUCache, sha256_text
from src.utils.llm import cloud_generation, local_generation

logger = get_logger("engine.match_checker")

# ── Entity-type-specific matching rules ──
# These supplement the generic prompt with hard rules that prevent
# common false merges and false skips per entity type.

_TYPE_SPECIFIC_RULES: Dict[str, str] = {
    "organizations": """
## Organization-Specific Rules (MUST follow)

- **Different government departments are DISTINCT entities**, even if both
  relate to security or defense.  Examples of entities that must NEVER match:
    - "Department of Homeland Security" ≠ "Department of Defense"
    - "FBI" ≠ "CIA"
    - "Coast Guard" ≠ "Navy"
    - "State Department" ≠ "Department of Justice"
- **Generic category phrases are NOT real entities.** If one name is a
  vague plural category like "defense departments", "security agencies", or
  "military officials", it is NOT the same as a specific named organization.
  Set is_match=false when one name is a generic category.
- Acronyms and full names CAN match (e.g., "ICE" = "Immigration and Customs
  Enforcement") — but only when one is the recognized abbreviation of the other.
""",
    "locations": """
## Location-Specific Rules (MUST follow)

- **Sub-location rule (refined):** Named camps or facilities INSIDE a larger
  location are NOT the same entity as the larger location.
    - Camp Delta ≠ Guantánamo Bay / Naval Station Guantanamo Bay
    - Camp X-Ray ≠ Guantánamo Bay
    - Camp Echo ≠ Guantánamo Bay
- **Guantánamo metonymy:** In this domain, "Guantánamo Bay", "Guantanamo",
  "GTMO", and "Naval Station Guantanamo Bay" typically refer to the SAME
  entity (the U.S. naval base / detention complex). Descriptive phrases like
  "U.S. military base in Guantánamo Bay" or "the detention facility at
  Guantanamo" also refer to this same entity — set is_match=true.
- **Descriptive phrase = canonical name:** If one name is a descriptive phrase
  (e.g., "military base in Cuba") and the other is the canonical proper noun
  it describes (e.g., "Naval Station Guantanamo Bay"), they CAN be the same
  entity. Focus on whether they refer to the same real-world place, not
  whether the names are formatted identically.
- **U.S. / United States metonymy:** "U.S. soil", "American territory", and
  similar phrases refer to the "United States" — they are the same entity.
""",
    "people": """
## Person-Specific Rules (MUST follow)

- **Name variations are common:** Arabic names have many transliteration
  variants (e.g., "al-Qahtani" / "al-Kahtani" / "Al Qahtani"). These CAN
  be the same person — check biographical details and context carefully.
- **Different people with similar names:** Two people can share a family
  name or partial name. Always check roles, nationalities, and context
  before confirming a match.
""",
    "events": """
## Event-Specific Rules (MUST follow)

- **Different hearings/proceedings on different dates are DISTINCT events,**
  even if they involve the same detainee or the same legal case.
- **Same event reported differently:** The same hearing/transfer/event
  described from different angles or at different levels of detail CAN
  be the same event — check dates and participants.
""",
}

_BASE_SYSTEM_PROMPT = """You are an expert analyst specializing in entity resolution \
for news articles about Guantánamo Bay.

Your task is to determine if two profiles refer to the same real-world {entity_label}.

Consider the following when making your determination:
1. Name variations: Different spellings, nicknames, titles, abbreviations, or partial names
2. Contextual information: Role, affiliations, actions, and biographical details
3. Temporal consistency: Whether the information in both profiles could apply to the same entity at different times
4. Generic vs. specific: If one name is a vague generic phrase (not a proper noun), it is almost certainly NOT the same entity as a specific named {entity_label}

You MUST provide:
- is_match: true or false
- confidence: a float from 0.0 to 1.0 indicating how certain you are \
(0.9+ = very confident, 0.5-0.7 = uncertain, below 0.5 = guessing)
- reason: a detailed explanation citing specific evidence

{type_rules}"""

_USER_PROMPT = """I need to determine if these two {entity_label}s refer to the \
same real-world entity:

## EVIDENCE FROM NEW ARTICLE:
Name: {new_name}
{new_profile_text}

## EXISTING PROFILE IN DATABASE:
Name: {existing_name}
Profile: {existing_profile_text}

Do these refer to the same {entity_label}? Provide your analysis with a confidence score."""


# ---------------------------------------------------------------------------
# Per-run match-check memoization
# ---------------------------------------------------------------------------
_MATCH_MEMO: Optional[LRUCache] = None
_MATCH_MEMO_ENABLED: bool = False


def configure_match_check_memo(*, enabled: bool, max_items: int = 8192) -> None:
    """Configure per-run memoization for deterministic match checks.

    Call once from pipeline startup after loading domain config.
    """
    global _MATCH_MEMO, _MATCH_MEMO_ENABLED
    _MATCH_MEMO_ENABLED = enabled
    if enabled:
        _MATCH_MEMO = LRUCache(max_items=max_items)
        logger.info(f"Match-check memoization enabled (max_items={max_items})")
    else:
        _MATCH_MEMO = None


def reset_match_check_memo() -> None:
    """Clear memoization state (useful between test runs)."""
    global _MATCH_MEMO, _MATCH_MEMO_ENABLED
    if _MATCH_MEMO is not None:
        _MATCH_MEMO.clear()
    _MATCH_MEMO_ENABLED = False
    _MATCH_MEMO = None


def _match_memo_key(
    *,
    backend: str,
    model: str,
    entity_type: str,
    new_name: str,
    existing_name: str,
    new_profile_text: str,
    existing_profile_text: str,
) -> str:
    """Build a stable hash key from all inputs that affect the match result."""
    parts = "|".join(
        [
            backend,
            model,
            entity_type,
            sha256_text(new_name),
            sha256_text(existing_name),
            sha256_text(new_profile_text),
            sha256_text(existing_profile_text),
        ]
    )
    return sha256_text(parts)


class MatchCheckResult(BaseModel):
    is_match: bool
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="How confident the model is in the match decision (0=uncertain, 1=certain)",
    )
    reason: str


def _build_prompts(
    new_name: str,
    existing_name: str,
    new_profile_text: str,
    existing_profile_text: str,
    entity_type: str,
) -> tuple:
    """Build system and user prompts with entity-type-specific rules."""
    entity_label = entity_type.rstrip("s")  # "organizations" → "organization"
    type_rules = _TYPE_SPECIFIC_RULES.get(entity_type, "")

    system_content = _BASE_SYSTEM_PROMPT.format(
        entity_label=entity_label,
        type_rules=type_rules,
    )
    user_content = _USER_PROMPT.format(
        entity_label=entity_label,
        new_name=new_name,
        new_profile_text=new_profile_text,
        existing_name=existing_name,
        existing_profile_text=existing_profile_text,
    )
    return system_content, user_content


def local_model_check_match(
    new_name: str,
    existing_name: str,
    new_profile_text: str,
    existing_profile_text: str,
    *,
    entity_type: str = "organizations",
    model: str = OLLAMA_MODEL,
) -> MatchCheckResult:
    """Check if new article evidence refers to the same entity as an existing profile.

    Args:
        new_name: The name extracted from the new article
        existing_name: The name from an existing profile in our database
        new_profile_text: Evidence text (or profile) from the new article
        existing_profile_text: The existing profile text we're comparing against
        entity_type: The entity type being compared (people, organizations, locations, events)
        model: The LLM model to use for comparison
    """
    # Memo lookup
    if _MATCH_MEMO_ENABLED and _MATCH_MEMO is not None:
        memo_key = _match_memo_key(
            backend="local",
            model=model,
            entity_type=entity_type,
            new_name=new_name,
            existing_name=existing_name,
            new_profile_text=new_profile_text,
            existing_profile_text=existing_profile_text,
        )
        cached = _MATCH_MEMO.get(memo_key)
        if cached is not None:
            logger.debug(f"Match-check memo hit for {new_name} vs {existing_name}")
            return cached

    system_content, user_content = _build_prompts(
        new_name, existing_name, new_profile_text, existing_profile_text, entity_type
    )

    try:
        result = local_generation(
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            response_model=MatchCheckResult,
            model=model,
            temperature=0,
        )
    except Exception as e:
        log("Error with Ollama API", level="error", exception=e)
        result = MatchCheckResult(
            is_match=False, confidence=0.0, reason=f"API error: {str(e)}"
        )

    # Memo store
    if _MATCH_MEMO_ENABLED and _MATCH_MEMO is not None:
        _MATCH_MEMO.set(memo_key, result)

    return result


def cloud_model_check_match(
    new_name: str,
    existing_name: str,
    new_profile_text: str,
    existing_profile_text: str,
    *,
    entity_type: str = "organizations",
    model: str = CLOUD_MODEL,
) -> MatchCheckResult:
    """Check if new article evidence refers to the same entity as an existing profile.

    Args:
        new_name: The name extracted from the new article
        existing_name: The name from an existing profile in our database
        new_profile_text: Evidence text (or profile) from the new article
        existing_profile_text: The existing profile text we're comparing against
        entity_type: The entity type being compared (people, organizations, locations, events)
        model: The LLM model to use for comparison
    """
    # Memo lookup
    if _MATCH_MEMO_ENABLED and _MATCH_MEMO is not None:
        memo_key = _match_memo_key(
            backend="cloud",
            model=model,
            entity_type=entity_type,
            new_name=new_name,
            existing_name=existing_name,
            new_profile_text=new_profile_text,
            existing_profile_text=existing_profile_text,
        )
        cached = _MATCH_MEMO.get(memo_key)
        if cached is not None:
            logger.debug(f"Match-check memo hit for {new_name} vs {existing_name}")
            return cached

    system_content, user_content = _build_prompts(
        new_name, existing_name, new_profile_text, existing_profile_text, entity_type
    )

    try:
        result = cloud_generation(
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            response_model=MatchCheckResult,
            model=model,
            temperature=0,
        )
    except Exception as e:
        log("Error with Gemini API", level="error", exception=e)
        result = MatchCheckResult(
            is_match=False, confidence=0.0, reason=f"API error: {str(e)}"
        )

    # Memo store
    if _MATCH_MEMO_ENABLED and _MATCH_MEMO is not None:
        _MATCH_MEMO.set(memo_key, result)

    return result
