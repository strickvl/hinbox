"""Entity merging and deduplication functionality."""

import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rich.markdown import Markdown
from rich.panel import Panel

from src.constants import (
    ENABLE_PROFILE_VERSIONING,
    SIMILARITY_THRESHOLD,
)
from src.exceptions import (
    EmbeddingError,
    EntityMergeError,
    ProfileGenerationError,
    SimilarityCalculationError,
)
from src.logging_config import console, display_markdown, get_logger, log
from src.match_checker import cloud_model_check_match, local_model_check_match
from src.profiles import VersionedProfile, create_profile, update_profile
from src.utils.embeddings import EmbeddingManager
from src.utils.error_handler import handle_merge_error
from src.utils.file_ops import write_entity_to_file
from src.utils.profiles import extract_profile_text

# Get module-specific logger
logger = get_logger("merge")

# Global embedding manager for merge operations
_embedding_manager = None


def get_embedding_manager(
    model_type: str = "default", domain: str = "guantanamo"
) -> EmbeddingManager:
    """Get or create embedding manager for merge operations."""
    global _embedding_manager
    if _embedding_manager is None:
        _embedding_manager = EmbeddingManager(model_type=model_type, domain=domain)
    return _embedding_manager


# Match check functions moved to match_checker.py to avoid circular imports


def normalize_vector(vector: List[float]) -> np.ndarray:
    """
    Normalize a vector to unit length.

    Args:
        vector: The vector to normalize

    Returns:
        Normalized vector as numpy array
    """
    if not vector:
        return np.array([])

    vector_np = np.array(vector, dtype=np.float32)
    norm = np.linalg.norm(vector_np)

    if norm > 0:
        return vector_np / norm
    return vector_np


def compute_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Compute cosine similarity between two vectors using centralized embedding manager.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity score (0-1)
    """
    embedding_manager = get_embedding_manager()
    return embedding_manager.compute_similarity(vec1, vec2)


def find_similar_person(
    person_name: str,
    person_embedding: List[float],
    entities: Dict[str, Dict],
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> Tuple[Optional[str], Optional[float]]:
    """
    Find the most similar person in the entities database using embedding similarity.

    Args:
        person_name: Name of the person to find
        person_embedding: Embedding vector of the person
        entities: Dictionary of entities
        similarity_threshold: Minimum similarity score to consider a match

    Returns:
        Tuple of (matched_person_name, similarity_score) or (None, None) if no match
    """
    if not person_embedding or not entities["people"]:
        return None, None

    best_match = None
    best_score = 0.0

    # First check for exact name match
    if person_name in entities["people"]:
        existing_person = entities["people"][person_name]
        if "profile_embedding" in existing_person:
            similarity = compute_similarity(
                person_embedding, existing_person["profile_embedding"]
            )
            log(
                f"Exact name match for '{person_name}' with similarity: {similarity:.4f}",
                level="info",
            )
            if similarity >= similarity_threshold:
                return person_name, similarity

    # If no exact match or similarity below threshold, search all entities
    for existing_name, existing_person in entities["people"].items():
        # Skip exact name match as we already checked it
        if existing_name == person_name:
            continue

        if "profile_embedding" in existing_person:
            similarity = compute_similarity(
                person_embedding, existing_person["profile_embedding"]
            )

            if similarity > best_score:
                best_score = similarity
                best_match = existing_name

    if best_match and best_score >= similarity_threshold:
        log(
            f"Found similar person: '{best_match}' for '{person_name}' with similarity: {best_score:.4f}",
            level="info",
        )
        return best_match, best_score

    return None, None


def find_similar_location(
    loc_name: str,
    loc_type: str,
    loc_embedding: List[float],
    entities: Dict[str, Dict],
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> Tuple[Optional[Tuple[str, str]], Optional[float]]:
    """
    Find the most similar location in the entities database using embedding similarity.

    Args:
        loc_name: Name of the location to find
        loc_type: Type of the location
        loc_embedding: Embedding vector of the location
        entities: Dictionary of entities
        similarity_threshold: Minimum similarity score to consider a match

    Returns:
        Tuple of (matched_location_key, similarity_score) or (None, None) if no match
    """
    if not loc_embedding or not entities["locations"]:
        return None, None

    best_match = None
    best_score = 0.0

    # First check for exact match
    location_key = (loc_name, loc_type)
    if location_key in entities["locations"]:
        existing_loc = entities["locations"][location_key]
        if "profile_embedding" in existing_loc:
            similarity = compute_similarity(
                loc_embedding, existing_loc["profile_embedding"]
            )
            log(
                f"Exact match for location '{loc_name}' with similarity: {similarity:.4f}",
                level="info",
            )
            if similarity >= similarity_threshold:
                return location_key, similarity

    # If no exact match or similarity below threshold, search all entities
    for existing_key, existing_loc in entities["locations"].items():
        # Skip exact key match as we already checked it
        if existing_key == location_key:
            continue

        if "profile_embedding" in existing_loc:
            similarity = compute_similarity(
                loc_embedding, existing_loc["profile_embedding"]
            )

            if similarity > best_score:
                best_score = similarity
                best_match = existing_key

    if best_match and best_score >= similarity_threshold:
        log(
            f"Found similar location: '{best_match[0]}' for '{loc_name}' with similarity: {best_score:.4f}",
            level="warning",
        )
        return best_match, best_score

    return None, None


def find_similar_organization(
    org_name: str,
    org_type: str,
    org_embedding: List[float],
    entities: Dict[str, Dict],
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> Tuple[Optional[Tuple[str, str]], Optional[float]]:
    """
    Find the most similar organization in the entities database using embedding similarity.

    Args:
        org_name: Name of the organization to find
        org_type: Type of the organization
        org_embedding: Embedding vector of the organization
        entities: Dictionary of entities
        similarity_threshold: Minimum similarity score to consider a match

    Returns:
        Tuple of (matched_organization_key, similarity_score) or (None, None) if no match
    """
    if not org_embedding or not entities["organizations"]:
        return None, None

    best_match = None
    best_score = 0.0

    # First check for exact match
    org_key = (org_name, org_type)
    if org_key in entities["organizations"]:
        existing_org = entities["organizations"][org_key]
        if "profile_embedding" in existing_org:
            similarity = compute_similarity(
                org_embedding, existing_org["profile_embedding"]
            )
            log(
                f"Exact match for organization '{org_name}' with similarity: {similarity:.4f}",
                level="info",
            )
            if similarity >= similarity_threshold:
                return org_key, similarity

    # If no exact match or similarity below threshold, search all entities
    for existing_key, existing_org in entities["organizations"].items():
        # Skip exact key match as we already checked it
        if existing_key == org_key:
            continue

        if "profile_embedding" in existing_org:
            similarity = compute_similarity(
                org_embedding, existing_org["profile_embedding"]
            )

            if similarity > best_score:
                best_score = similarity
                best_match = existing_key

    if best_match and best_score >= similarity_threshold:
        log(
            f"Found similar organization: '{best_match[0]}' for '{org_name}' with similarity: {best_score:.4f}",
            level="warning",
        )
        return best_match, best_score

    return None, None


def find_similar_event(
    event_title: str,
    event_type: str,
    event_start_date: str,
    event_embedding: List[float],
    entities: Dict[str, Dict],
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> Tuple[Optional[Tuple[str, str]], Optional[float]]:
    """
    Find the most similar event in the entities database using embedding similarity.

    Args:
        event_title: Title of the event to find
        event_type: Type of the event
        event_start_date: Start date of the event
        event_embedding: Embedding vector of the event
        entities: Dictionary of entities
        similarity_threshold: Minimum similarity score to consider a match

    Returns:
        Tuple of (matched_event_key, similarity_score) or (None, None) if no match
    """
    if not event_embedding or not entities["events"]:
        return None, None

    best_match = None
    best_score = 0.0

    # First check for exact match
    event_key = (event_title, event_start_date)
    if event_key in entities["events"]:
        existing_event = entities["events"][event_key]
        if "profile_embedding" in existing_event:
            similarity = compute_similarity(
                event_embedding, existing_event["profile_embedding"]
            )
            log(
                f"Exact match for event '{event_title}' with similarity: {similarity:.4f}",
                level="info",
            )
            if similarity >= similarity_threshold:
                return event_key, similarity

    # If no exact match or similarity below threshold, search all entities
    for existing_key, existing_event in entities["events"].items():
        # Skip exact key match as we already checked it
        if existing_key == event_key:
            continue

        if "profile_embedding" in existing_event:
            similarity = compute_similarity(
                event_embedding, existing_event["profile_embedding"]
            )

            if similarity > best_score:
                best_score = similarity
                best_match = existing_key

    if best_match and best_score >= similarity_threshold:
        log(
            f"Found similar event: '{best_match[0]}' for '{event_title}' with similarity: {best_score:.4f}",
            level="warning",
        )
        return best_match, best_score

    return None, None


# (Removed)


def merge_people(
    extracted_people: List[Dict[str, Any]],
    entities: Dict[str, Dict],
    article_id: str,
    article_title: str,
    article_url: str,
    article_published_date: Any,
    article_content: str,
    extraction_timestamp: str,
    model_type: str = "gemini",
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    langfuse_session_id: str = None,
    langfuse_trace_id: str = None,
):
    # Determine embedding model type based on main model type
    embedding_model_type = "local" if model_type == "ollama" else "cloud"
    embedding_manager = get_embedding_manager(model_type=embedding_model_type)

    log(
        f"Starting merge_people with {len(extracted_people)} people to process",
        level="processing",
    )
    log(
        f"Using model: {model_type}, embedding model: {embedding_model_type}",
        level="info",
    )

    for p in extracted_people:
        person_name = p.get("name", "")
        if not person_name:
            log("Skipping person with empty name", level="error")
            continue

        log(f"Processing person: {person_name}", level="processing")
        entity_updated = False

        try:
            log(f"Attempting to create profile for {person_name}...", level="info")
            proposed_profile, proposed_versioned_profile, reflection_history = (
                create_profile(
                    "person",
                    person_name,
                    article_content,
                    article_id,
                    model_type,
                    "guantanamo",  # domain parameter was missing
                    langfuse_session_id,
                    langfuse_trace_id,
                )
            )

            # Extract profile text from response
            log("Extracting profile text from response...", level="info")
            proposed_profile = extract_profile_text(proposed_profile)
            if not proposed_profile or not proposed_profile.get("text"):
                log(
                    f"Failed to generate profile for {person_name}. Profile data: {proposed_profile}",
                    level="error",
                )
                continue

            # Generate embedding for the person name and type
            proposed_profile_text = proposed_profile.get("text", "")
            log(
                f"Generating embedding for profile text (length: {len(proposed_profile_text)})",
                level="info",
            )
            proposed_person_embedding = embedding_manager.embed_text(
                proposed_profile_text
            )
            log(
                f"Generated embedding of size: {len(proposed_person_embedding)}",
                level="info",
            )

            # Find similar person using embeddings
            log(
                f"Searching for similar person with similarity threshold: {similarity_threshold}",
                level="info",
            )
            similar_name, similarity_score = find_similar_person(
                person_name, proposed_person_embedding, entities, similarity_threshold
            )

            if similar_name:
                log(
                    f"Doing a final check to see if '{person_name}' is the same as '{similar_name}'...",
                    level="info",
                )

                existing_profile = entities["people"][similar_name].get("profile", {})
                existing_profile_text = existing_profile.get("text", "")
                if not existing_profile_text:
                    log(f"No existing profile text for {similar_name}", level="error")
                    continue

                log("Performing model-based match check...", level="info")
                if model_type == "ollama":
                    result = local_model_check_match(
                        person_name,
                        similar_name,
                        proposed_profile_text,
                        existing_profile_text,
                        langfuse_session_id=langfuse_session_id,
                        langfuse_trace_id=langfuse_trace_id,
                    )
                else:
                    result = cloud_model_check_match(
                        person_name,
                        similar_name,
                        proposed_profile_text,
                        existing_profile_text,
                        langfuse_session_id=langfuse_session_id,
                        langfuse_trace_id=langfuse_trace_id,
                    )

                log(
                    f"Match check result: {result.is_match} - {result.reason}",
                    level="info",
                )

                if result.is_match:
                    log(
                        f"The profiles match! Merging '{person_name}' with '{similar_name}'",
                        level="success",
                    )
                else:
                    log(
                        f"The profiles do not match. Skipping merge for '{person_name}'",
                        level="error",
                    )
                    continue

                # We found a similar person - use that instead of creating a new one
                log(
                    f"Merging '{person_name}' with existing person '[bold]{similar_name}[/bold]' (similarity: {similarity_score:.4f})",
                    level="processing",
                )

                # Use the existing person's name as the key
                existing_person = entities["people"][similar_name]

                # Check if this article is already associated with the person
                article_exists = any(
                    a.get("article_id") == article_id
                    for a in existing_person["articles"]
                )

                if not article_exists:
                    log(
                        f"Adding new article reference for {similar_name}", level="info"
                    )
                    existing_person["articles"].append(
                        {
                            "article_id": article_id,
                            "article_title": article_title,
                            "article_url": article_url,
                            "article_published_date": article_published_date,
                        }
                    )
                    entity_updated = True

                    # Update profile with new information
                    if "profile" in existing_person:
                        log(
                            f"\n[yellow]Updating profile for person:[/] {similar_name}",
                            level="info",
                        )
                        # Load existing versioned profile or create new one
                        if (
                            ENABLE_PROFILE_VERSIONING
                            and "profile_versions" in existing_person
                        ):
                            versioned_profile = VersionedProfile(
                                **existing_person["profile_versions"]
                            )
                        else:
                            versioned_profile = VersionedProfile()

                        updated_profile, versioned_profile, reflection_history = (
                            update_profile(
                                "person",
                                similar_name,
                                existing_person["profile"],
                                versioned_profile,
                                article_content,
                                article_id,
                                model_type,
                                "guantanamo",
                                langfuse_session_id=langfuse_session_id,
                                langfuse_trace_id=langfuse_trace_id,
                            )
                        )
                        existing_person["profile"] = updated_profile
                        existing_person["profile_versions"] = (
                            versioned_profile.model_dump()
                            if ENABLE_PROFILE_VERSIONING
                            else None
                        )
                        log(
                            "Generating new embedding for updated profile...",
                            level="info",
                        )
                        existing_person["profile_embedding"] = (
                            embedding_manager.embed_text(updated_profile["text"])
                        )
                        # Store reflection iteration history for debugging
                        existing_person.setdefault("reflection_history", [])
                        existing_person["reflection_history"].extend(reflection_history)

                        display_markdown(
                            updated_profile["text"],
                            title=f"Updated Profile: {similar_name}",
                            style="yellow",
                        )
                    else:
                        log(
                            f"\n[green]Creating initial profile for person:[/] {similar_name}",
                            level="info",
                        )
                        new_profile, new_versioned_profile, reflection_history = (
                            create_profile(
                                "person",
                                similar_name,
                                article_content,
                                article_id,
                                model_type,
                                "guantanamo",
                                langfuse_session_id=langfuse_session_id,
                                langfuse_trace_id=langfuse_trace_id,
                            )
                        )
                        existing_person["profile"] = new_profile
                        existing_person["profile_versions"] = (
                            new_versioned_profile.model_dump()
                            if ENABLE_PROFILE_VERSIONING
                            else None
                        )
                        log("Generating embedding for new profile...", level="info")
                        existing_person["profile_embedding"] = (
                            embedding_manager.embed_text(new_profile["text"])
                        )
                        # Store reflection iteration history
                        existing_person.setdefault("reflection_history", [])
                        existing_person["reflection_history"].extend(reflection_history)

                        display_markdown(
                            new_profile["text"],
                            title=f"New Profile: {similar_name}",
                            style="green",
                        )

                    # Store alternative names if they differ
                    if person_name != similar_name:
                        if "alternative_names" not in existing_person:
                            existing_person["alternative_names"] = []

                        if person_name not in existing_person["alternative_names"]:
                            existing_person["alternative_names"].append(person_name)
                            log(
                                f"Added alternative name: '{person_name}' for '{similar_name}'",
                                level="processing",
                            )
                            entity_updated = True

                # Update extraction timestamp to earliest
                existing_timestamp = existing_person.get(
                    "extraction_timestamp", extraction_timestamp
                )
                if existing_timestamp != min(existing_timestamp, extraction_timestamp):
                    existing_person["extraction_timestamp"] = min(
                        existing_timestamp, extraction_timestamp
                    )
                    entity_updated = True

                if entity_updated:
                    log(
                        f"Writing updated entity to file for {similar_name}...",
                        level="info",
                    )
                    write_entity_to_file("people", similar_name, existing_person)
                    entities["people"][similar_name] = existing_person
                    log(
                        f"Updated person entity saved to file: {similar_name}",
                        level="success",
                    )
            else:
                # No similar person found - create new entry
                log(f"Creating new person entry for: {person_name}", level="success")

                # We already have proposed_profile, reflection_history, and proposed_person_embedding
                # so we simply reuse them instead of calling create_profile() again.
                profile_embedding = proposed_person_embedding
                reflection_history = reflection_history or []

                new_person = {
                    "name": person_name,
                    "profile": proposed_profile,
                    "profile_versions": proposed_versioned_profile.model_dump()
                    if ENABLE_PROFILE_VERSIONING
                    else None,
                    "articles": [
                        {
                            "article_id": article_id,
                            "article_title": article_title,
                            "article_url": article_url,
                            "article_published_date": article_published_date,
                        }
                    ],
                    "profile_embedding": profile_embedding,
                    "extraction_timestamp": extraction_timestamp,
                    "alternative_names": [],
                    "reflection_history": reflection_history,
                }

                entities["people"][person_name] = new_person
                write_entity_to_file("people", person_name, new_person)
                log(
                    f"[green]New person entity saved to file:[/] {person_name}",
                    level="info",
                )

        except Exception as e:
            log(f"Error processing person {person_name}:", level="error")
            log(f"Error details: {str(e)}", level="error")
            log(f"Traceback:\n{traceback.format_exc()}", level="error")
            continue

    log("Completed merge_people function", level="success")


def merge_locations(
    extracted_locations: List[Dict[str, Any]],
    entities: Dict[str, Dict],
    article_id: str,
    article_title: str,
    article_url: str,
    article_published_date: Any,
    article_content: str,
    extraction_timestamp: str,
    model_type: str = "gemini",
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    langfuse_session_id: str = None,
    langfuse_trace_id: str = None,
):
    # Determine embedding model type based on main model type
    embedding_model_type = "local" if model_type == "ollama" else "cloud"
    embedding_manager = get_embedding_manager(model_type=embedding_model_type)

    for loc in extracted_locations:
        loc_name = loc.get("name", "")
        loc_type = loc.get("type", "")
        if not loc_name:
            continue
        location_key = (loc_name, loc_type)

        entity_updated = False

        # Generate embedding for the location name and type
        proposed_profile, proposed_versioned_profile, reflection_history = (
            create_profile(
                "location",
                loc_name,
                article_content,
                article_id,
                model_type,
                "guantanamo",
                langfuse_session_id=langfuse_session_id,
                langfuse_trace_id=langfuse_trace_id,
            )
        )
        # Extract profile text from response
        proposed_profile = extract_profile_text(proposed_profile)
        proposed_profile_text = (
            proposed_profile.get("text") if proposed_profile else None
        )
        if not proposed_profile_text:
            log(f"Failed to generate profile for location {loc_name}", level="error")
            continue

        proposed_location_embedding = embedding_manager.embed_text(
            proposed_profile_text
        )

        # Find similar location using embeddings
        similar_key, similarity_score = find_similar_location(
            loc_name,
            loc_type,
            proposed_location_embedding,
            entities,
            similarity_threshold,
        )

        if similar_key:
            log(
                f"[purple]Doing a final check to see if '{loc_name}' is the same as '{similar_key[0]}'[/purple]...",
                level="info",
            )
            if model_type == "ollama":
                result = local_model_check_match(
                    loc_name,
                    similar_key,
                    proposed_profile_text,
                    entities["locations"][similar_key]["profile"]["text"],
                    langfuse_session_id=langfuse_session_id,
                    langfuse_trace_id=langfuse_trace_id,
                )
            else:
                result = cloud_model_check_match(
                    loc_name,
                    similar_key,
                    proposed_profile_text,
                    entities["locations"][similar_key]["profile"]["text"],
                    langfuse_session_id=langfuse_session_id,
                    langfuse_trace_id=langfuse_trace_id,
                )
            if result.is_match:
                log(
                    f"The profiles match! Merging '{loc_name}' with '{similar_key[0]}'",
                    level="success",
                )
            else:
                log(
                    f"The profiles do not match. Skipping merge for '{loc_name}'",
                    level="error",
                )
                continue

            # We found a similar location - use that instead of creating a new one
            similar_name, _ = similar_key
            log(
                f"Merging location '{loc_name}' with existing location '[bold]{similar_name}[/bold]' (similarity: {similarity_score:.4f})",
                level="processing",
            )

            # Use the existing location's key
            existing_loc = entities["locations"][similar_key]

            # Check if this article is already associated with the location
            article_exists = any(
                a.get("article_id") == article_id for a in existing_loc["articles"]
            )

            if not article_exists:
                existing_loc["articles"].append(
                    {
                        "article_id": article_id,
                        "article_title": article_title,
                        "article_url": article_url,
                        "article_published_date": article_published_date,
                    }
                )
                entity_updated = True

                # Update profile
                if "profile" in existing_loc:
                    log(
                        f"\n[yellow]Updating profile for location:[/] {similar_name}",
                        level="info",
                    )
                    # Load existing versioned profile or create new one
                    if ENABLE_PROFILE_VERSIONING and "profile_versions" in existing_loc:
                        versioned_profile = VersionedProfile(
                            **existing_loc["profile_versions"]
                        )
                    else:
                        versioned_profile = VersionedProfile()

                    updated_profile, versioned_profile, reflection_history = (
                        update_profile(
                            "location",
                            similar_name,
                            existing_loc["profile"],
                            versioned_profile,
                            article_content,
                            article_id,
                            model_type,
                            "guantanamo",
                            langfuse_session_id=langfuse_session_id,
                            langfuse_trace_id=langfuse_trace_id,
                        )
                    )
                    existing_loc["profile"] = updated_profile
                    existing_loc["profile_versions"] = (
                        versioned_profile.model_dump()
                        if ENABLE_PROFILE_VERSIONING
                        else None
                    )
                    existing_loc["profile_embedding"] = embedding_manager.embed_text(
                        updated_profile["text"]
                    )
                    existing_loc.setdefault("reflection_history", [])
                    existing_loc["reflection_history"].extend(reflection_history)

                    console.print(
                        Panel(
                            Markdown(updated_profile["text"]),
                            title=f"Updated Profile: {similar_name}",
                            border_style="yellow",
                        )
                    )
                else:
                    log(
                        f"\n[green]Creating initial profile for location:[/] {similar_name}",
                        level="info",
                    )
                    new_profile, new_versioned_profile, reflection_history = (
                        create_profile(
                            "location",
                            similar_name,
                            article_content,
                            article_id,
                            model_type,
                            "guantanamo",
                            langfuse_session_id=langfuse_session_id,
                            langfuse_trace_id=langfuse_trace_id,
                        )
                    )
                    existing_loc["profile"] = new_profile
                    existing_loc["profile_versions"] = (
                        new_versioned_profile.model_dump()
                        if ENABLE_PROFILE_VERSIONING
                        else None
                    )
                    existing_loc["profile_embedding"] = embedding_manager.embed_text(
                        new_profile["text"]
                    )
                    existing_loc.setdefault("reflection_history", [])
                    existing_loc["reflection_history"].extend(reflection_history)

                    console.print(
                        Panel(
                            Markdown(new_profile["text"]),
                            title=f"New Profile: {similar_name}",
                            border_style="green",
                        )
                    )

                # Store alternative names if they differ
                if location_key != similar_key:
                    if "alternative_names" not in existing_loc:
                        existing_loc["alternative_names"] = []

                    alt_name_entry = {"name": loc_name, "type": loc_type}
                    if alt_name_entry not in existing_loc["alternative_names"]:
                        existing_loc["alternative_names"].append(alt_name_entry)
                        log(
                            f"Added alternative name: '{loc_name}' (type: {loc_type}) for '{similar_name}'",
                            level="processing",
                        )
                        entity_updated = True

            existing_timestamp = existing_loc.get(
                "extraction_timestamp", extraction_timestamp
            )
            if existing_timestamp != min(existing_timestamp, extraction_timestamp):
                existing_loc["extraction_timestamp"] = min(
                    existing_timestamp, extraction_timestamp
                )
                entity_updated = True

            if entity_updated:
                write_entity_to_file("locations", similar_key, existing_loc)
                entities["locations"][similar_key] = existing_loc
                log(
                    f"[blue]Updated location entity saved to file:[/] {similar_name}",
                    level="info",
                )
        else:
            # No similar location found - create new entry
            log(
                f"\n[green]Creating profile for new location:[/] {loc_name}",
                level="info",
            )

            # Reuse the proposed_profile, reflection_history, and proposed_location_embedding
            profile = proposed_profile
            profile_embedding = proposed_location_embedding
            reflection_history = reflection_history or []

            # Optionally display the newly generated profile text
            console.print(
                Panel(
                    Markdown(profile["text"]),
                    title=f"New Profile: {loc_name}",
                    border_style="green",
                )
            )

            new_location = {
                "name": loc_name,
                "type": loc_type,
                "profile": profile,
                "profile_versions": proposed_versioned_profile.model_dump()
                if ENABLE_PROFILE_VERSIONING
                else None,
                "articles": [
                    {
                        "article_id": article_id,
                        "article_title": article_title,
                        "article_url": article_url,
                        "article_published_date": article_published_date,
                    }
                ],
                "profile_embedding": profile_embedding,
                "extraction_timestamp": extraction_timestamp,
                "alternative_names": [],
                "reflection_history": reflection_history,
            }

            entities["locations"][location_key] = new_location
            write_entity_to_file("locations", location_key, new_location)
            log(
                f"[green]New location entity saved to file:[/] {loc_name}", level="info"
            )


def merge_organizations(
    extracted_orgs: List[Dict[str, Any]],
    entities: Dict[str, Dict],
    article_id: str,
    article_title: str,
    article_url: str,
    article_published_date: Any,
    article_content: str,
    extraction_timestamp: str,
    model_type: str = "gemini",
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    langfuse_session_id: str = None,
    langfuse_trace_id: str = None,
):
    # Determine embedding model type based on main model type
    embedding_model_type = "local" if model_type == "ollama" else "cloud"
    embedding_manager = get_embedding_manager(model_type=embedding_model_type)

    for org in extracted_orgs:
        org_name = org.get("name", "")
        org_type = org.get("type", "")
        if not org_name:
            continue
        org_key = (org_name, org_type)

        try:
            entity_updated = False

            # Generate embedding for the organization name and type
            try:
                proposed_profile, proposed_versioned_profile, reflection_history = (
                    create_profile(
                        "organization",
                        org_name,
                        article_content,
                        article_id,
                        model_type,
                        "guantanamo",
                        langfuse_session_id=langfuse_session_id,
                        langfuse_trace_id=langfuse_trace_id,
                    )
                )
                # Extract profile text from response
                proposed_profile = extract_profile_text(proposed_profile)
                proposed_profile_text = (
                    proposed_profile.get("text") if proposed_profile else None
                )
                if not proposed_profile_text:
                    raise ProfileGenerationError(
                        f"Failed to generate profile text for organization {org_name}",
                        org_name,
                        "organization",
                        article_id,
                    )
            except Exception as e:
                handle_merge_error("organizations", org_name, e, "profile_generation")
                continue

            try:
                proposed_organization_embedding = embedding_manager.embed_text(
                    proposed_profile_text
                )
            except Exception:
                error = EmbeddingError(
                    f"Failed to generate embedding for organization {org_name}",
                    {
                        "embedding_model_type": embedding_model_type,
                        "organization": org_name,
                    },
                    article_id,
                )
                handle_merge_error(
                    "organizations", org_name, error, "embedding_generation"
                )
                continue

            # Find similar organization using embeddings
            try:
                similar_key, similarity_score = find_similar_organization(
                    org_name,
                    org_type,
                    proposed_organization_embedding,
                    entities,
                    similarity_threshold,
                )
            except Exception:
                error = SimilarityCalculationError(
                    f"Failed to find similar organizations for {org_name}",
                    "organizations",
                )
                handle_merge_error(
                    "organizations", org_name, error, "similarity_search"
                )
                # Continue with no similar entity found
                similar_key, similarity_score = None, 0.0

            if similar_key:
                log(
                    f"[purple]Doing a final check to see if '{org_name}' is the same as '{similar_key[0]}'[/purple]...",
                    level="info",
                )
                # First, ensure existing_org["profile"] is a dict with "text" to avoid KeyError
                existing_profile_dict = entities["organizations"][similar_key].get(
                    "profile", {}
                )
                if (
                    not isinstance(existing_profile_dict, dict)
                    or "text" not in existing_profile_dict
                ):
                    log(
                        f"Existing organization '{similar_key}' profile is missing 'text'—cannot finalize check.",
                        level="error",
                    )
                    # We'll treat it as if it doesn't match, or we can skip it:
                    continue

                try:
                    if model_type == "ollama":
                        result = local_model_check_match(
                            org_name,
                            similar_key,
                            proposed_profile_text,
                            existing_profile_dict["text"],
                            langfuse_session_id=langfuse_session_id,
                            langfuse_trace_id=langfuse_trace_id,
                        )
                    else:
                        result = cloud_model_check_match(
                            org_name,
                            similar_key,
                            proposed_profile_text,
                            existing_profile_dict["text"],
                            langfuse_session_id=langfuse_session_id,
                            langfuse_trace_id=langfuse_trace_id,
                        )
                except Exception as e:
                    handle_merge_error(
                        "organizations", org_name, e, "match_verification"
                    )
                    # Default to no match if verification fails
                    result = MatchCheckResult(
                        is_match=False, reason="Match verification failed"
                    )

                if result.is_match:
                    log(
                        f"The profiles match! Merging '{org_name}' with '{similar_key[0]}'",
                        level="success",
                    )
                else:
                    log(
                        f"The profiles do not match. Skipping merge for '{org_name}'",
                        level="error",
                    )
                    continue

                # We found a similar organization - use that instead of creating a new one
                similar_name, _ = similar_key
                log(
                    f"Merging organization '{org_name}' with existing organization '[bold]{similar_name}[/bold]' (similarity: {similarity_score:.4f})",
                    level="processing",
                )

                # Use the existing organization's key
                existing_org = entities["organizations"][similar_key]

                # Check if this article is already associated with the organization
                article_exists = any(
                    a.get("article_id") == article_id for a in existing_org["articles"]
                )

                if not article_exists:
                    existing_org["articles"].append(
                        {
                            "article_id": article_id,
                            "article_title": article_title,
                            "article_url": article_url,
                            "article_published_date": article_published_date,
                        }
                    )
                    entity_updated = True

                    # Update profile
                    if "profile" in existing_org:
                        log(
                            f"\n[yellow]Updating profile for organization:[/] {similar_name}",
                            level="info",
                        )
                        # Load existing versioned profile or create new one
                        if (
                            ENABLE_PROFILE_VERSIONING
                            and "profile_versions" in existing_org
                        ):
                            versioned_profile = VersionedProfile(
                                **existing_org["profile_versions"]
                            )
                        else:
                            versioned_profile = VersionedProfile()

                        updated_profile, versioned_profile, reflection_history = (
                            update_profile(
                                "organization",
                                similar_name,
                                existing_org["profile"],
                                versioned_profile,
                                article_content,
                                article_id,
                                model_type,
                                "guantanamo",
                                langfuse_session_id=langfuse_session_id,
                                langfuse_trace_id=langfuse_trace_id,
                            )
                        )
                        existing_org["profile"] = updated_profile
                        existing_org["profile_versions"] = (
                            versioned_profile.model_dump()
                            if ENABLE_PROFILE_VERSIONING
                            else None
                        )
                        existing_org["profile_embedding"] = (
                            embedding_manager.embed_text(updated_profile["text"])
                        )
                        existing_org.setdefault("reflection_history", [])
                        existing_org["reflection_history"].extend(reflection_history)

                        console.print(
                            Panel(
                                Markdown(updated_profile["text"]),
                                title=f"Updated Profile: {similar_name}",
                                border_style="yellow",
                            )
                        )
                    else:
                        log(
                            f"\n[green]Creating initial profile for organization:[/] {similar_name}",
                            level="info",
                        )
                        new_profile, new_versioned_profile, reflection_history = (
                            create_profile(
                                "organization",
                                similar_name,
                                article_content,
                                article_id,
                                model_type,
                                "guantanamo",
                                langfuse_session_id=langfuse_session_id,
                                langfuse_trace_id=langfuse_trace_id,
                            )
                        )
                        existing_org["profile"] = new_profile
                        existing_org["profile_versions"] = (
                            new_versioned_profile.model_dump()
                            if ENABLE_PROFILE_VERSIONING
                            else None
                        )
                        existing_org["profile_embedding"] = (
                            embedding_manager.embed_text(new_profile["text"])
                        )
                        existing_org.setdefault("reflection_history", [])
                        existing_org["reflection_history"].extend(reflection_history)

                        console.print(
                            Panel(
                                Markdown(new_profile["text"]),
                                title=f"New Profile: {similar_name}",
                                border_style="green",
                            )
                        )

                    # Store alternative names if they differ
                    if org_key != similar_key:
                        if "alternative_names" not in existing_org:
                            existing_org["alternative_names"] = []

                        alt_name_entry = {"name": org_name, "type": org_type}
                        if alt_name_entry not in existing_org["alternative_names"]:
                            existing_org["alternative_names"].append(alt_name_entry)
                            log(
                                f"Added alternative name: '{org_name}' (type: {org_type}) for '{similar_name}'",
                                level="processing",
                            )
                            entity_updated = True

                existing_timestamp = existing_org.get(
                    "extraction_timestamp", extraction_timestamp
                )
                if existing_timestamp != min(existing_timestamp, extraction_timestamp):
                    existing_org["extraction_timestamp"] = min(
                        existing_timestamp, extraction_timestamp
                    )
                    entity_updated = True

                if entity_updated:
                    write_entity_to_file("organizations", similar_key, existing_org)
                    entities["organizations"][similar_key] = existing_org
                    log(
                        f"[blue]Updated organization entity saved to file:[/] {similar_name}",
                        level="info",
                    )
            else:
                # No similar organization found - create new entry
                log(
                    f"\n[green]Creating profile for new organization:[/] {org_name}",
                    level="info",
                )

                # Reuse the proposed_profile, reflection_history, and proposed_organization_embedding
                profile = proposed_profile
                profile_embedding = proposed_organization_embedding
                reflection_history = reflection_history or []

                console.print(
                    Panel(
                        Markdown(profile["text"]),
                        title=f"New Profile: {org_name}",
                        border_style="green",
                    )
                )

                new_org = {
                    "name": org_name,
                    "type": org_type,
                    "profile": profile,
                    "profile_versions": proposed_versioned_profile.model_dump()
                    if ENABLE_PROFILE_VERSIONING
                    else None,
                    "articles": [
                        {
                            "article_id": article_id,
                            "article_title": article_title,
                            "article_url": article_url,
                            "article_published_date": article_published_date,
                        }
                    ],
                    "profile_embedding": profile_embedding,
                    "extraction_timestamp": extraction_timestamp,
                    "alternative_names": [],
                    "reflection_history": reflection_history,
                }

                entities["organizations"][org_key] = new_org
                write_entity_to_file("organizations", org_key, new_org)
                log(
                    f"[green]New organization entity saved to file:[/] {org_name}",
                    level="info",
                )

        except Exception:
            # Catch any unhandled errors for this organization
            error = EntityMergeError(
                f"Unexpected error while processing organization {org_name}",
                "organizations",
                org_name,
            )
            handle_merge_error("organizations", org_name, error, "general")


def merge_events(
    extracted_events: List[Dict[str, Any]],
    entities: Dict[str, Dict],
    article_id: str,
    article_title: str,
    article_url: str,
    article_published_date: Any,
    article_content: str,
    extraction_timestamp: str,
    model_type: str = "gemini",
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    langfuse_session_id: str = None,
    langfuse_trace_id: str = None,
):
    # Determine embedding model type based on main model type
    embedding_model_type = "local" if model_type == "ollama" else "cloud"
    embedding_manager = get_embedding_manager(model_type=embedding_model_type)

    for e in extracted_events:
        event_title = e.get("title", "")
        event_start_date = e.get("start_date", "")
        event_type = e.get("event_type", "")
        if not event_title:
            continue
        event_key = (event_title, event_start_date)

        entity_updated = False

        # Generate embedding for the event title and type
        proposed_profile, proposed_versioned_profile, reflection_history = (
            create_profile(
                "event",
                event_title,
                article_content,
                article_id,
                model_type,
                "guantanamo",
                langfuse_session_id=langfuse_session_id,
                langfuse_trace_id=langfuse_trace_id,
            )
        )
        # Extract profile text from response
        proposed_profile = extract_profile_text(proposed_profile)
        proposed_profile_text = (
            proposed_profile.get("text") if proposed_profile else None
        )
        if not proposed_profile_text:
            log(f"Failed to generate profile for event {event_title}", level="error")
            continue

        proposed_event_embedding = embedding_manager.embed_text(proposed_profile_text)

        # Find similar event using embeddings
        similar_key, similarity_score = find_similar_event(
            event_title,
            event_type,
            event_start_date,
            proposed_event_embedding,
            entities,
            similarity_threshold,
        )

        if similar_key:
            log(
                f"[purple]Doing a final check to see if '{event_title}' is the same as '{similar_key}'[/purple]...",
                level="info",
            )
            if model_type == "ollama":
                result = local_model_check_match(
                    event_title,
                    similar_key,
                    proposed_profile_text,
                    entities["events"][similar_key]["profile"]["text"],
                    langfuse_session_id=langfuse_session_id,
                    langfuse_trace_id=langfuse_trace_id,
                )
            else:
                result = cloud_model_check_match(
                    event_title,
                    similar_key,
                    proposed_profile_text,
                    entities["events"][similar_key]["profile"]["text"],
                    langfuse_session_id=langfuse_session_id,
                    langfuse_trace_id=langfuse_trace_id,
                )
            if result.is_match:
                log(
                    f"✓ The profiles match! Merging '{event_title}' with '{similar_key[0]}'",
                    level="success",
                )
            else:
                log(
                    f"The profiles do not match. Skipping merge for '{event_title}'",
                    level="error",
                )
                continue

            # We found a similar event - use that instead of creating a new one
            similar_title, _ = similar_key
            log(
                f"Merging event '{event_title}' with existing event '[bold]{similar_title}[/bold]' (similarity: {similarity_score:.4f})",
                level="processing",
            )

            # Use the existing event's key
            existing_event = entities["events"][similar_key]

            # Check if this article is already associated with the event
            article_exists = any(
                a.get("article_id") == article_id for a in existing_event["articles"]
            )

            if not article_exists:
                existing_event["articles"].append(
                    {
                        "article_id": article_id,
                        "article_title": article_title,
                        "article_url": article_url,
                        "article_published_date": article_published_date,
                    }
                )
                entity_updated = True

                # Update profile
                if "profile" in existing_event:
                    log(
                        f"\n[yellow]Updating profile for event:[/] {similar_title}",
                        level="info",
                    )
                    # Load existing versioned profile or create new one
                    if (
                        ENABLE_PROFILE_VERSIONING
                        and "profile_versions" in existing_event
                    ):
                        versioned_profile = VersionedProfile(
                            **existing_event["profile_versions"]
                        )
                    else:
                        versioned_profile = VersionedProfile()

                    updated_profile, versioned_profile, reflection_history = (
                        update_profile(
                            "event",
                            similar_title,
                            existing_event["profile"],
                            versioned_profile,
                            article_content,
                            article_id,
                            model_type,
                            "guantanamo",
                            langfuse_session_id=langfuse_session_id,
                            langfuse_trace_id=langfuse_trace_id,
                        )
                    )
                    existing_event["profile"] = updated_profile
                    existing_event["profile_versions"] = (
                        versioned_profile.model_dump()
                        if ENABLE_PROFILE_VERSIONING
                        else None
                    )
                    existing_event["profile_embedding"] = embedding_manager.embed_text(
                        updated_profile["text"]
                    )
                    existing_event.setdefault("reflection_history", [])
                    existing_event["reflection_history"].extend(reflection_history)

                    console.print(
                        Panel(
                            Markdown(updated_profile["text"]),
                            title=f"Updated Profile: {similar_title}",
                            border_style="yellow",
                        )
                    )
                else:
                    log(
                        f"\n[green]Creating initial profile for event:[/] {similar_title}",
                        level="info",
                    )
                    new_profile, new_versioned_profile, reflection_history = (
                        create_profile(
                            "event",
                            similar_title,
                            article_content,
                            article_id,
                            model_type,
                            "guantanamo",
                            langfuse_session_id=langfuse_session_id,
                            langfuse_trace_id=langfuse_trace_id,
                        )
                    )
                    existing_event["profile"] = new_profile
                    existing_event["profile_versions"] = (
                        new_versioned_profile.model_dump()
                        if ENABLE_PROFILE_VERSIONING
                        else None
                    )
                    existing_event["profile_embedding"] = embedding_manager.embed_text(
                        new_profile["text"]
                    )
                    existing_event.setdefault("reflection_history", [])
                    existing_event["reflection_history"].extend(reflection_history)

                    console.print(
                        Panel(
                            Markdown(new_profile["text"]),
                            title=f"New Profile: {similar_title}",
                            border_style="green",
                        )
                    )

                # Store alternative titles if they differ
                if event_key != similar_key:
                    if "alternative_titles" not in existing_event:
                        existing_event["alternative_titles"] = []

                    alt_title_entry = {
                        "title": event_title,
                        "start_date": event_start_date,
                        "event_type": event_type,
                    }
                    if alt_title_entry not in existing_event["alternative_titles"]:
                        existing_event["alternative_titles"].append(alt_title_entry)
                        log(
                            f"Added alternative title: '{event_title}' for '{similar_title}'",
                            level="processing",
                        )
                        entity_updated = True

            existing_timestamp = existing_event.get(
                "extraction_timestamp", extraction_timestamp
            )
            if existing_timestamp != min(existing_timestamp, extraction_timestamp):
                existing_event["extraction_timestamp"] = min(
                    existing_timestamp, extraction_timestamp
                )
                entity_updated = True

            if entity_updated:
                write_entity_to_file("events", similar_key, existing_event)
                entities["events"][similar_key] = existing_event
                log(
                    f"[blue]Updated event entity saved to file:[/] {similar_title}",
                    level="info",
                )
        else:
            # Create new entry
            log(
                f"\n[green]Creating profile for new event:[/] {event_title}",
                level="info",
            )

            # Reuse the proposed_profile, reflection_history, and proposed_event_embedding
            profile = proposed_profile
            profile_embedding = proposed_event_embedding
            reflection_history = reflection_history or []

            console.print(
                Panel(
                    Markdown(profile["text"]),
                    title=f"New Profile: {event_title}",
                    border_style="green",
                )
            )

            new_event = {
                "title": event_title,
                "description": e.get("description", ""),
                "event_type": event_type,
                "start_date": event_start_date,
                "end_date": e.get("end_date", ""),
                "is_fuzzy_date": e.get("is_fuzzy_date", False),
                "tags": e.get("tags", []),
                "profile": profile,
                "profile_versions": proposed_versioned_profile.model_dump()
                if ENABLE_PROFILE_VERSIONING
                else None,
                "profile_embedding": profile_embedding,
                "articles": [
                    {
                        "article_id": article_id,
                        "article_title": article_title,
                        "article_url": article_url,
                        "article_published_date": article_published_date,
                    }
                ],
                "extraction_timestamp": extraction_timestamp,
                "alternative_titles": [],
                "reflection_history": reflection_history,
            }

            entities["events"][event_key] = new_event
            write_entity_to_file("events", event_key, new_event)
            log(
                f"[green]New event entity saved to file:[/] {event_title}", level="info"
            )


# Generic merger wrapper functions using the new EntityMerger class
def merge_people_generic(
    extracted_people: List[Dict[str, Any]],
    entities: Dict[str, Dict],
    article_id: str,
    article_title: str,
    article_url: str,
    article_published_date: Any,
    article_content: str,
    extraction_timestamp: str,
    model_type: str = "gemini",
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    domain: str = "guantanamo",
):
    """Merge people entities using the generic EntityMerger."""
    from src.mergers import EntityMerger

    merger = EntityMerger("people")
    merger.merge_entities(
        extracted_people,
        entities,
        article_id,
        article_title,
        article_url,
        article_published_date,
        article_content,
        extraction_timestamp,
        model_type,
        similarity_threshold,
        domain,
    )


def merge_organizations_generic(
    extracted_orgs: List[Dict[str, Any]],
    entities: Dict[str, Dict],
    article_id: str,
    article_title: str,
    article_url: str,
    article_published_date: Any,
    article_content: str,
    extraction_timestamp: str,
    model_type: str = "gemini",
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    domain: str = "guantanamo",
):
    """Merge organization entities using the generic EntityMerger."""
    from src.mergers import EntityMerger

    merger = EntityMerger("organizations")
    merger.merge_entities(
        extracted_orgs,
        entities,
        article_id,
        article_title,
        article_url,
        article_published_date,
        article_content,
        extraction_timestamp,
        model_type,
        similarity_threshold,
        domain,
    )


def merge_locations_generic(
    extracted_locations: List[Dict[str, Any]],
    entities: Dict[str, Dict],
    article_id: str,
    article_title: str,
    article_url: str,
    article_published_date: Any,
    article_content: str,
    extraction_timestamp: str,
    model_type: str = "gemini",
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    domain: str = "guantanamo",
):
    """Merge location entities using the generic EntityMerger."""
    from src.mergers import EntityMerger

    merger = EntityMerger("locations")
    merger.merge_entities(
        extracted_locations,
        entities,
        article_id,
        article_title,
        article_url,
        article_published_date,
        article_content,
        extraction_timestamp,
        model_type,
        similarity_threshold,
        domain,
    )


def merge_events_generic(
    extracted_events: List[Dict[str, Any]],
    entities: Dict[str, Dict],
    article_id: str,
    article_title: str,
    article_url: str,
    article_published_date: Any,
    article_content: str,
    extraction_timestamp: str,
    model_type: str = "gemini",
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    domain: str = "guantanamo",
):
    """Merge event entities using the generic EntityMerger."""
    from src.mergers import EntityMerger

    merger = EntityMerger("events")
    merger.merge_entities(
        extracted_events,
        entities,
        article_id,
        article_title,
        article_url,
        article_published_date,
        article_content,
        extraction_timestamp,
        model_type,
        similarity_threshold,
        domain,
    )
