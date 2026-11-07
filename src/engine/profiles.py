import copy
import json
import traceback
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from src.config_loader import get_domain_config
from src.constants import (
    CLOUD_MODEL,
    DEFAULT_TEMPERATURE,
    ENABLE_PROFILE_VERSIONING,
    MAX_ITERATIONS,
    OLLAMA_MODEL,
    PROFILE_CREATE_APPROACH,
    PROFILE_MAX_TOKENS,
    PROFILE_UPDATE_APPROACH,
)
from src.engine.profile_create import CREATE_APPROACHES, create_with_checks
from src.exceptions import (
    ProfileGenerationError,
    ProfileUpdateError,
)
from src.logging_config import get_logger, log
from src.utils.citation_ids import repair_profile_citations
from src.utils.error_handler import (
    handle_profile_error,
    retry_on_error,
)
from src.utils.llm import (
    GenerationMode,
    cloud_generation,
    iterative_improve,
    local_generation,
    text_generation,
)
from src.utils.outcomes import PhaseOutcome
from src.utils.profiles import extract_profile_text
from src.utils.quality_controls import run_profile_qc
from src.utils.step_timing import timed
from src.utils.tracing import annotate

logger = get_logger("profiles")


def _resolve_mode(model_type: str, model: Optional[str]) -> GenerationMode:
    """An explicit ``model`` decides cloud vs local by its prefix; else ``model_type``."""
    if model:
        return (
            GenerationMode.LOCAL
            if model.startswith("ollama/")
            else GenerationMode.CLOUD
        )
    return GenerationMode.CLOUD if model_type == "gemini" else GenerationMode.LOCAL


def _prompt_model(model_type: str, model: Optional[str]) -> str:
    """The model that will answer, which picks the prompt variant to load."""
    if model:
        return model
    return CLOUD_MODEL if model_type == "gemini" else OLLAMA_MODEL


@timed("draft")
def _generate_initial_profile_text(
    messages: List[Dict], mode: GenerationMode, model: Optional[str] = None
) -> str:
    """Generate a first-draft profile and return it as JSON text for reflection."""
    if mode == GenerationMode.CLOUD:
        response = cloud_generation(
            messages=messages,
            response_model=EntityProfile,
            model=model or CLOUD_MODEL,
            max_tokens=PROFILE_MAX_TOKENS,
            temperature=DEFAULT_TEMPERATURE,
        )
    else:
        response = local_generation(
            messages=messages,
            response_model=EntityProfile,
            model=model or OLLAMA_MODEL,
            max_tokens=PROFILE_MAX_TOKENS,
            temperature=DEFAULT_TEMPERATURE,
        )
    return response.model_dump_json() if response else "{}"


def _apply_profile_qc(profile: Optional[Dict], entity_name: str) -> Optional[Dict]:
    """Run deterministic profile QC and record the result on the profile.

    Applies QC's safe fixes (default tags, clamped confidence) and stores
    ``{"passed", "flags"}`` under ``profile["qc"]``. A cited article ID one or
    two typos away from one of the profile's sources is snapped back to it first
    (``repair_profile_citations``). A failing profile is kept
    as-is; QC only flags it, it never triggers regeneration.
    """
    if not isinstance(profile, dict):
        return profile
    profile, repaired = repair_profile_citations(profile)
    profile, report = run_profile_qc(profile=profile)
    flags = list(report.flags) + (["citation_ids_repaired"] if repaired else [])
    profile["qc"] = {"passed": report.passed, "flags": flags}
    if repaired:
        log(f"Fixed {repaired} garbled citation ID(s) in '{entity_name}'", level="info")
    if not report.passed:
        log(
            f"Profile QC failed for '{entity_name}': {', '.join(report.flags)}",
            level="warning",
        )
    return profile


class ProfileVersion(BaseModel):
    """A single version of an entity profile."""

    version_number: int
    profile_data: Dict  # Complete profile snapshot
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    trigger_article_id: Optional[str] = None

    @field_validator("profile_data")
    @classmethod
    def deep_copy_profile_data(cls, v):
        """Ensure profile_data is deep copied to prevent mutation."""
        return copy.deepcopy(v)


class VersionedProfile(BaseModel):
    """Container for versioned profile history."""

    current_version: int = 1
    versions: List[ProfileVersion] = Field(default_factory=list)

    def add_version(
        self, profile_data: Dict, trigger_article_id: Optional[str] = None
    ) -> ProfileVersion:
        """Add a new version to the history."""
        new_version = ProfileVersion(
            version_number=len(self.versions) + 1,
            profile_data=copy.deepcopy(profile_data),
            trigger_article_id=trigger_article_id,
        )
        self.versions.append(new_version)
        self.current_version = new_version.version_number
        return new_version

    def get_version(self, version_number: int) -> Optional[ProfileVersion]:
        """Get a specific version by number."""
        for version in self.versions:
            if version.version_number == version_number:
                return version
        return None

    def get_latest(self) -> Optional[ProfileVersion]:
        """Get the latest version."""
        return self.versions[-1] if self.versions else None


class EntityProfile(BaseModel):
    """Structured profile model for an entity."""

    text: str = Field(..., description="Comprehensive profile text about the entity")
    tags: List[str] = Field(
        default_factory=list,
        description="List of relevant tags/keywords for this entity",
    )
    confidence: float = Field(
        ...,
        description="Confidence score (0-1) in the accuracy of this profile",
        ge=0,
        le=1,
    )
    sources: List[str] = Field(
        default_factory=list,
        description="List of article IDs used as sources for this profile",
    )


@retry_on_error(max_retries=2, initial_delay=1.0)
def generate_profile_with_reflection(
    entity_type: str,
    entity_name: str,
    article_text: str,
    article_id: str,
    model_type: str = "gemini",
    max_iterations: int = MAX_ITERATIONS,
    domain: str = "guantanamo",
    model: Optional[str] = None,
    approach: Optional[str] = None,
) -> (Dict, list):
    """
    Generate a profile with reflection and improvement.
    Uses iterative_improve from utils.py.
    Returns a tuple: (final_profile_dict, improvement_history).
    """

    # Load prompt template from config
    config = get_domain_config(domain)
    prompt_model = _prompt_model(model_type, model)
    prompt_template = config.load_profile_prompt("generation", prompt_model)

    # Format the prompt with the specific values
    custom_system_prompt = prompt_template.format(
        entity_type=entity_type, entity_name=entity_name, article_id=article_id
    )

    # Build messages for the new iterative_improve signature
    generation_messages = [
        {"role": "system", "content": custom_system_prompt},
        {
            "role": "user",
            "content": f"Article text:\n\n{article_text}\n\nCreate a profile for {entity_type} '{entity_name}'. Article ID: {article_id}",
        },
    ]

    mode = _resolve_mode(model_type, model)
    approach = approach or PROFILE_CREATE_APPROACH
    if approach not in CREATE_APPROACHES:
        raise ValueError(f"Unknown profile create approach {approach!r}")
    annotate(create_approach=approach)
    if approach == "check":
        profile, history = create_with_checks(
            draft=lambda msgs: _generate_initial_profile_text(msgs, mode, model),
            write_text=lambda msgs: text_generation(
                msgs,
                model=model
                or (CLOUD_MODEL if mode == GenerationMode.CLOUD else OLLAMA_MODEL),
                local=mode == GenerationMode.LOCAL,
                max_tokens=PROFILE_MAX_TOKENS,
            ),
            generation_messages=generation_messages,
            article_id=article_id,
            entity_type=entity_type,
        )
        final_text = json.dumps(profile) if profile else ""
        return _finish_generated_profile(
            final_text, history, entity_name, entity_type, article_id
        )

    # Load reflection prompt from config
    reflection_template = config.load_profile_prompt("reflection", prompt_model)
    reflection_prompt = reflection_template.format(
        entity_type=entity_type, entity_name=entity_name, article_id=article_id
    )

    # Generate initial profile text for iterative improvement
    initial_text = _generate_initial_profile_text(generation_messages, mode, model)

    # Use iterative_improve to refine the profile
    result = iterative_improve(
        initial_text=initial_text,
        generation_messages=generation_messages,
        reflection_prompt=reflection_prompt,
        response_model=EntityProfile,
        max_iterations=max_iterations,
        mode=mode,
        model=model,
        max_tokens=PROFILE_MAX_TOKENS,
    )

    return _finish_generated_profile(
        result.get("text", "{}"),
        result.get("reflection_history", []),
        entity_name,
        entity_type,
        article_id,
    )


def _finish_generated_profile(
    final_text: str,
    history: list,
    entity_name: str,
    entity_type: str,
    article_id: str,
) -> Tuple[Dict, list]:
    """Parse the final profile JSON, or return the fallback profile on failure."""
    try:
        if final_text.strip():
            if final_text.strip().startswith("{"):
                data = json.loads(final_text)
                final_result = EntityProfile(**data)
            else:
                final_result = EntityProfile(
                    text=final_text, tags=[], confidence=0.8, sources=[article_id]
                )
        else:
            final_result = None
    except Exception as e:
        logger.error(f"Error parsing profile generation result for {entity_name}: {e}")
        final_result = None

    if final_result is None:
        # Use standardized error handling for profile generation failures
        error = ProfileGenerationError(
            f"Profile generation failed after {len(history)} iterations",
            entity_name,
            entity_type,
            article_id,
            {"iterations": len(history), "history": history},
        )
        fallback_profile = handle_profile_error(
            entity_name, entity_type, article_id, error, "generation"
        )
        return fallback_profile, history

    result_dict = final_result.model_dump()
    result_dict["sources"] = [article_id]

    return result_dict, history


def generate_profile(
    entity_type: str,
    entity_name: str,
    article_text: str,
    article_id: str,
    model_type: str = "gemini",
    domain: str = "guantanamo",
    model: Optional[str] = None,
    approach: Optional[str] = None,
) -> (Dict, list):
    """
    Generate a profile for an entity based on article text using structured extraction.
    Now uses reflection pattern for validation and improvement.
    Returns (profile_dict, improvement_history).
    """
    profile_dict, improvement_history = generate_profile_with_reflection(
        entity_type=entity_type,
        entity_name=entity_name,
        article_text=article_text,
        article_id=article_id,
        model_type=model_type,
        domain=domain,
        model=model,
        approach=approach,
    )
    return profile_dict, improvement_history


def _update_profile_internal(
    entity_type: str,
    entity_name: str,
    existing_profile: Dict,
    new_article_text: str,
    new_article_id: str,
    model_type: str = "gemini",
    domain: str = "guantanamo",
    model: Optional[str] = None,
) -> Tuple[Dict, list]:
    """
    Internal function to update an existing profile with new information from an article.

    We treat the existing profile text plus the new article text as input for improvement.
    Returns (updated_profile, improvement_history).
    """
    log(
        f"Updating profile for {entity_type} '{entity_name}' "
        f"(model={model or model_type}, article={new_article_id}, "
        f"existing={len(existing_profile.get('text', ''))}ch, new={len(new_article_text)}ch)",
        level="debug",
    )

    try:
        # Load update prompt template from config
        config = get_domain_config(domain)
        prompt_model = _prompt_model(model_type, model)
        update_template = config.load_profile_prompt("update", prompt_model)

        # Format the base prompt
        base_prompt = update_template.format(
            entity_type=entity_type, entity_name=entity_name
        )

        # Build the complete system prompt with existing profile and new article
        system_prompt = f"""{base_prompt}

Existing Profile (already validated):
----------------
{existing_profile.get("text", "")}

Existing Tags: {existing_profile.get("tags", [])}
Existing Confidence: {existing_profile.get("confidence", 0.0)}
Existing Sources: {existing_profile.get("sources", [])}

New Article (ID: {new_article_id}):
----------------
{new_article_text}
"""

        log("Starting iterative improvement process", level="debug")

        # For the iterative process, we treat the entire text above as the 'prompt' so the LLM merges them.
        # Import already at module level

        generation_mode = _resolve_mode(model_type, model)

        # Build messages for iterative improvement
        generation_messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": "Please create an updated profile by merging the existing profile with new information from the article.",
            },
        ]

        # Load reflection prompt for updates
        reflection_template = config.load_profile_prompt("reflection", prompt_model)
        reflection_prompt = reflection_template.format(
            entity_type=entity_type, entity_name=entity_name, article_id=new_article_id
        )

        # Draft the merged profile first, so the first reflection grades a
        # real profile instead of an empty "{}" (which always fails).
        initial_text = _generate_initial_profile_text(
            generation_messages, generation_mode, model
        )

        # We'll use the same EntityProfile structure for the updated profile.
        result = iterative_improve(
            initial_text=initial_text,
            generation_messages=generation_messages,
            reflection_prompt=reflection_prompt,
            response_model=EntityProfile,
            max_iterations=3,
            mode=generation_mode,
            model=model,
            max_tokens=PROFILE_MAX_TOKENS,
        )

        final_result = None
        history = result.get("reflection_history", [])

        # Parse the final result
        try:
            import json

            final_text = result.get("text", "{}")
            if final_text.strip():
                if final_text.strip().startswith("{"):
                    data = json.loads(final_text)
                    final_result = EntityProfile(**data)
                else:
                    # If it's not JSON, wrap it
                    final_result = EntityProfile(
                        text=final_text,
                        tags=existing_profile.get("tags", []),
                        confidence=0.8,
                        sources=existing_profile.get("sources", []) + [new_article_id],
                    )
        except Exception as e:
            log(f"Error parsing iterative improvement result: {e}", level="error")
            final_result = None

        if final_result is None:
            error = ProfileUpdateError(
                f"Failed to update profile for {entity_name}",
                entity_name,
                entity_type,
                new_article_id,
                {"existing_profile": existing_profile, "history": history},
            )
            log(
                f"Profile update failed for {entity_name}, falling back to existing profile",
                level="warning",
            )
            logger.error(f"Profile update failed: {error}")
            return existing_profile, history

        # Log reflection pass/fail results
        for idx, entry in enumerate(history):
            passed_str = "PASSED" if entry.get("valid", False) else "FAILED"
            log(
                f"Reflection {idx + 1} for {entity_name}: {passed_str} — {entry.get('reasoning', '')}",
                level="debug",
            )

        # Convert final_result to a dict
        updated_profile_dict = final_result.model_dump()

        # Ensure there's a non-empty "text" field to avoid KeyError later
        if "text" not in updated_profile_dict or not isinstance(
            updated_profile_dict["text"], str
        ):
            log(
                f"Profile updated but no valid 'text' found. Using fallback for {entity_name}.",
                level="warning",
            )
            updated_profile_dict["text"] = (
                f"Profile update for {entity_name} is incomplete^[{new_article_id}]"
            )

        # Merge old and new sources
        old_sources = existing_profile.get("sources", [])
        all_sources = list(set(old_sources + [new_article_id]))
        updated_profile_dict["sources"] = all_sources

        final_text_len = len(updated_profile_dict.get("text", ""))
        log(
            f"Updated profile for {entity_name}: "
            f"{final_text_len}ch, conf={updated_profile_dict.get('confidence', 0.0):.2f}, "
            f"tags={len(updated_profile_dict.get('tags', []))}, iters={len(history)}",
            level="debug",
        )
        if final_text_len < 50:
            log(
                f"Profile text for {entity_name} under 50 chars — may be incomplete",
                level="warning",
            )

        return updated_profile_dict, history

    except Exception as e:
        log(f"Error updating profile for {entity_name}: {e}", level="error")
        logger.debug(f"Traceback:\n{traceback.format_exc()}")
        raise


def _edit_profile_update(
    entity_type: str,
    entity_name: str,
    existing_profile: Dict,
    new_article_text: str,
    new_article_id: str,
    model_type: str = "gemini",
    domain: str = "guantanamo",
    model: Optional[str] = None,
    article_dates: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict, list]:
    """Update by applying the model's add/replace edits to the existing text.

    Keeps the profile's keys as they are (the edit list and its report are
    logged, not stored). Falls back to the existing profile if the call fails.
    ``article_dates`` (published date by article id) lets the edit guards
    tell a later update from a dispute.
    """
    from src.engine.profile_edits import edit_profile
    from src.utils.ollama_context import PromptTooLongError

    local = _resolve_mode(model_type, model) == GenerationMode.LOCAL
    try:
        edited = edit_profile(
            entity_type,
            entity_name,
            existing_profile,
            new_article_text,
            new_article_id,
            model=model,
            domain=domain,
            local=local,
            article_dates=article_dates,
        )
    except PromptTooLongError:
        # The edit-list prompt holds the whole profile; the fact-list updater
        # never sends it, so a profile too long for the local model still
        # gets updated
        annotate(edit_fallback="facts")
        log(f"Profile of {entity_name} too long to edit; using the fact list")
        return _fact_profile_update(
            entity_type,
            entity_name,
            existing_profile,
            new_article_text,
            new_article_id,
            model_type,
            domain,
            model,
        )
    except Exception as e:
        log(
            f"Profile edit failed for {entity_name}, keeping existing profile: {e}",
            level="warning",
        )
        return existing_profile, []
    report = edited["edit_report"]
    annotate(
        profile_chars_shown=report.get("profile_chars_shown"),
        later_updates=len((report.get("guard") or {}).get("later_updates") or []),
        edits_applied=report["applied"],
        edits_appended=report["appended"],
        edits_skipped=len(report["skipped"]),
        disagreements=len(edited["disagreements"]),
    )
    log(
        f"Edited profile for {entity_name}: {report['applied']} applied, "
        f"{report['appended']} appended, {len(report['skipped'])} skipped, "
        f"{len(edited['disagreements'])} disagreements",
        level="debug",
    )
    return {
        **existing_profile,
        "text": edited["text"],
        "sources": edited["sources"],
    }, []


def _fact_profile_update(
    entity_type: str,
    entity_name: str,
    existing_profile: Dict,
    new_article_text: str,
    new_article_id: str,
    model_type: str = "gemini",
    domain: str = "guantanamo",
    model: Optional[str] = None,
) -> Tuple[Dict, list]:
    """Update from the profile's fact list (``src/engine/fact_list.py``).

    Saves the rows under ``fact_rows_json`` for the next update; the step
    report is logged, not stored. Falls back to the existing profile on failure.
    """
    from src.engine.fact_list import FACT_ROWS_KEY, update_from_facts

    if _resolve_mode(model_type, model) == GenerationMode.LOCAL and not model:
        model = OLLAMA_MODEL
    try:
        updated = update_from_facts(
            entity_type,
            entity_name,
            existing_profile,
            new_article_text,
            new_article_id,
            model=model,
            domain=domain,
        )
    except Exception as e:
        log(
            f"Fact-list update failed for {entity_name}, keeping existing profile: {e}",
            level="warning",
        )
        return existing_profile, []
    report = updated["fact_report"]
    log(
        f"Fact-list update for {entity_name}: {report.get('claims', 0)} claims, "
        f"{report.get('new', 0)} new, {report.get('same', 0)} same, "
        f"{report.get('update', 0)} timeline, {report.get('conflict', 0)} conflict",
        level="debug",
    )
    return {
        **existing_profile,
        "text": updated["text"],
        "sources": updated["sources"],
        FACT_ROWS_KEY: updated[FACT_ROWS_KEY],
    }, []


_UPDATERS = {
    "edit": _edit_profile_update,
    "facts": _fact_profile_update,
}


def _split_sections(
    profile: Dict,
    entity_type: str,
    entity_name: str,
    model_type: str,
    model: Optional[str],
    kind: Optional[str] = None,
    domain: str = "guantanamo",
) -> Dict:
    """Give a long profile its kind's starter headings once when
    ``HINBOX_SECTION_REORGANISE`` is on (``section_reorganise.py``), then split
    over-long sections into sub-sections when ``HINBOX_SECTION_SPLIT`` is on
    (``section_split.py``); the update's model answers both."""
    from src.engine.section_reorganise import reorganise_enabled, reorganise_profile
    from src.engine.section_split import split_enabled, split_profile

    if not profile:
        return profile
    local = _resolve_mode(model_type, model) == GenerationMode.LOCAL
    if reorganise_enabled():
        profile, report = reorganise_profile(
            profile, entity_type, entity_name, kind, domain,
            _prompt_model(model_type, model), local,
        )  # fmt: skip
        annotate(section_reorganise=report and report["status"])
    if not split_enabled():
        return profile
    split, reports = split_profile(
        profile, entity_type, entity_name, _prompt_model(model_type, model), local
    )
    annotate(section_splits=[f"{r['section']}: {r['status']}" for r in reports] or None)
    return split


@timed("profile.update")
def update_profile(
    entity_type: str,
    entity_name: str,
    existing_profile: Dict,
    versioned_profile: VersionedProfile,
    new_article_text: str,
    new_article_id: str,
    model_type: str = "gemini",
    domain: str = "guantanamo",
    model: Optional[str] = None,
    approach: Optional[str] = None,
    article_dates: Optional[Dict[str, Any]] = None,
    kind: Optional[str] = None,
) -> Tuple[Dict, VersionedProfile, list]:
    """
    Update profile and add version if versioning is enabled.

    ``approach`` is "edit", "facts" (the fact-list updater) or "rewrite"
    (default: ``PROFILE_UPDATE_APPROACH``). ``article_dates`` maps article
    ids (this one and the profile's sources) to published dates; the edit
    approach uses them to write a changed count or trial date as a later
    update rather than a dispute. ``kind`` is the entity's ``type``
    ("detainee"), which picks its starter headings.
    Returns (updated_profile, versioned_profile, improvement_history); the
    edit approach has no reflection rounds, so its history is empty.
    """
    approach = approach or PROFILE_UPDATE_APPROACH
    annotate(
        entity_type=entity_type,
        entity_name=entity_name,
        article_id=new_article_id,
        approach=approach,
        chars_before=len(str(existing_profile.get("text") or "")),
    )
    if approach not in ("edit", "rewrite", "facts"):
        raise ValueError(f"Unknown profile update approach: {approach!r}")
    update = _UPDATERS.get(approach, _update_profile_internal)
    extra = {"article_dates": article_dates} if approach == "edit" else {}
    updated_profile, history = update(
        entity_type,
        entity_name,
        existing_profile,
        new_article_text,
        new_article_id,
        model_type,
        domain,
        model,
        **extra,
    )
    # The rewrite and fact-list updaters build a new dict: keep the marker
    # that stops a second reorganise
    if existing_profile.get("reorganised") and updated_profile:
        updated_profile.setdefault("reorganised", existing_profile["reorganised"])
    updated_profile = _split_sections(
        updated_profile, entity_type, entity_name, model_type, model, kind, domain
    )

    updated_profile = _apply_profile_qc(updated_profile, entity_name)
    _annotate_profile(updated_profile)

    # Add new version if versioning enabled
    if ENABLE_PROFILE_VERSIONING:
        versioned_profile.add_version(
            profile_data=updated_profile, trigger_article_id=new_article_id
        )

    return updated_profile, versioned_profile, history


def _annotate_profile(profile: Dict) -> None:
    qc = profile.get("qc") or {}
    annotate(
        chars_after=len(str(profile.get("text") or "")),
        qc_passed=qc.get("passed"),
        qc_flags=[str(f) for f in qc.get("flags") or []],
    )


@timed("profile.create")
def create_profile(
    entity_type: str,
    entity_name: str,
    article_text: str,
    article_id: str,
    model_type: str = "gemini",
    domain: str = "guantanamo",
    model: Optional[str] = None,
    approach: Optional[str] = None,
) -> Tuple[Dict, VersionedProfile, list]:
    """
    Create an initial profile for an entity based on article text using structured extraction.
    Returns (profile_dict, versioned_profile, improvement_history).
    """
    annotate(entity_type=entity_type, entity_name=entity_name, article_id=article_id)
    log(
        f"Creating profile for {entity_type} '{entity_name}' "
        f"(model={model or model_type}, article={article_id}, {len(article_text)}ch)",
        level="debug",
    )

    try:
        profile_dict, improvement_history = generate_profile(
            entity_type,
            entity_name,
            article_text,
            article_id,
            model_type,
            domain,
            model,
            approach,
        )

        # Extract the actual text from nested/parsed fields before logging
        profile_dict = extract_profile_text(profile_dict)
        profile_dict = _apply_profile_qc(profile_dict, entity_name)
        _annotate_profile(profile_dict)

        log(
            f"Created profile for {entity_name}: "
            f"{len(profile_dict.get('text', ''))}ch, "
            f"conf={profile_dict.get('confidence', 0.0):.2f}, "
            f"tags={len(profile_dict.get('tags', []))}, iters={len(improvement_history)}",
            level="debug",
        )

        # Create versioned profile
        versioned_profile = VersionedProfile()
        if ENABLE_PROFILE_VERSIONING:
            versioned_profile.add_version(
                profile_data=profile_dict, trigger_article_id=article_id
            )

        return profile_dict, versioned_profile, improvement_history
    except Exception as e:
        log(f"Error creating profile for {entity_name}: {e}", level="error")
        logger.debug(f"Traceback:\n{traceback.format_exc()}")
        raise


def create_profile_with_outcome(
    entity_type: str,
    entity_name: str,
    article_text: str,
    article_id: str,
    model_type: str = "gemini",
    domain: str = "guantanamo",
) -> Tuple[Dict, VersionedProfile, list, PhaseOutcome]:
    """Create a profile and return a PhaseOutcome alongside the normal return value.

    Never raises — returns a fallback profile + failure outcome on error.
    Runs deterministic profile QC on the result.
    """
    try:
        profile_dict, versioned_profile, history = create_profile(
            entity_type, entity_name, article_text, article_id, model_type, domain
        )

        # Run profile QC
        profile_dict, qc_report = run_profile_qc(profile=profile_dict)

        outcome = PhaseOutcome.ok(
            f"profile.create.{entity_type}",
            value=profile_dict,
            counts={
                "text_length": qc_report.text_length,
                "citations": qc_report.citation_count,
                "tags": qc_report.tag_count,
            },
            flags=qc_report.flags,
            meta={"qc_fixes": qc_report.fixes, "qc_passed": qc_report.passed},
        )
        return profile_dict, versioned_profile, history, outcome

    except Exception as e:
        logger.error(f"create_profile_with_outcome failed for {entity_name}: {e}")
        fallback = handle_profile_error(
            entity_name, entity_type, article_id, e, "creation"
        )
        versioned_profile = VersionedProfile()
        outcome = PhaseOutcome.fail(
            f"profile.create.{entity_type}",
            error=e,
            fallback=fallback,
            context={
                "entity_name": entity_name,
                "entity_type": entity_type,
                "article_id": article_id,
            },
            flags=["profile_creation_exception"],
        )
        return fallback, versioned_profile, [], outcome


def update_profile_with_outcome(
    entity_type: str,
    entity_name: str,
    existing_profile: Dict,
    versioned_profile: VersionedProfile,
    new_article_text: str,
    new_article_id: str,
    model_type: str = "gemini",
    domain: str = "guantanamo",
) -> Tuple[Dict, VersionedProfile, list, PhaseOutcome]:
    """Update a profile and return a PhaseOutcome alongside the normal return value.

    Never raises — returns existing profile as fallback on error.
    Runs deterministic profile QC on the result.
    """
    try:
        updated_profile, versioned_profile, history = update_profile(
            entity_type,
            entity_name,
            existing_profile,
            versioned_profile,
            new_article_text,
            new_article_id,
            model_type,
            domain,
        )

        # Run profile QC
        updated_profile, qc_report = run_profile_qc(profile=updated_profile)

        outcome = PhaseOutcome.ok(
            f"profile.update.{entity_type}",
            value=updated_profile,
            counts={
                "text_length": qc_report.text_length,
                "citations": qc_report.citation_count,
                "tags": qc_report.tag_count,
            },
            flags=qc_report.flags,
            meta={"qc_fixes": qc_report.fixes, "qc_passed": qc_report.passed},
        )
        return updated_profile, versioned_profile, history, outcome

    except Exception as e:
        logger.error(f"update_profile_with_outcome failed for {entity_name}: {e}")
        outcome = PhaseOutcome.fail(
            f"profile.update.{entity_type}",
            error=e,
            fallback=existing_profile,
            context={
                "entity_name": entity_name,
                "entity_type": entity_type,
                "article_id": new_article_id,
            },
            flags=["profile_update_exception", "fallback_to_existing"],
        )
        return existing_profile, versioned_profile, [], outcome
