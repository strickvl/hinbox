from typing import Any, Dict, List, Optional, Tuple

import instructor
import litellm
import numpy as np
from pydantic import BaseModel
from rich import print
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from src.constants import (
    CLOUD_EMBEDDING_MODEL,
    GEMINI_MODEL,
    LOCAL_EMBEDDING_MODEL,
    SIMILARITY_THRESHOLD,
)
from src.embeddings import embed_text
from src.profiles import create_profile, update_profile
from src.utils import write_entity_to_file

console = Console()


class MatchCheckResult(BaseModel):
    is_match: bool
    reason: str


def cloud_model_check_match(
    new_name: str,
    existing_name: str,
    new_profile_text: str,
    existing_profile_text: str,
    model: str = GEMINI_MODEL,
) -> MatchCheckResult:
    """
    Check if a newly extracted profile refers to the same entity as an existing profile.

    This function uses an LLM to determine if two profiles refer to the same person,
    even when names might have variations or additional context is needed to establish identity.

    Args:
        new_name: The name extracted from the new article
        existing_name: The name from an existing profile in our database
        new_profile_text: The profile text generated from the new article
        existing_profile_text: The existing profile text we're comparing against
        model: The LLM model to use for comparison

    Returns:
        MatchCheckResult with is_match flag and detailed reasoning
    """
    client = instructor.from_litellm(litellm.completion)

    try:
        result = client.chat.completions.create(
            model=model,
            response_model=MatchCheckResult,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert analyst specializing in entity resolution for news articles about Guantánamo Bay.

Your task is to determine if two profiles refer to the same real-world entity (person, organization, location, or event).

Consider the following when making your determination:
1. Name variations: Different spellings, nicknames, titles, or partial names
2. Contextual information: Role, affiliations, actions, and biographical details
3. Temporal consistency: Whether the information in both profiles could apply to the same entity at different times
4. Contradictions: Whether there are any clear contradictions that would make it impossible for these to be the same entity

Even if names differ slightly, profiles may refer to the same entity if contextual details align.
Conversely, identical names might refer to different entities if contextual details clearly diverge.

Provide a detailed explanation for your decision, citing specific evidence from both profiles.""",
                },
                {
                    "role": "user",
                    "content": f"""I need to determine if these two profiles refer to the same entity:

## PROFILE FROM NEW ARTICLE:
Name: {new_name}
Profile: {new_profile_text}

## EXISTING PROFILE IN DATABASE:
Name: {existing_name}
Profile: {existing_profile_text}

Are these profiles referring to the same entity? Provide your analysis.""",
                },
            ],
            metadata={
                "project_name": "hinbox",
                "tags": ["dev", "entity_resolution"],
            },
        )
        return result
    except Exception as e:
        print(f"Error with Gemini API: {e}")
        # Return a default result indicating failure
        return MatchCheckResult(is_match=False, reason=f"API error: {str(e)}")


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
    Compute cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity score (0-1)
    """
    if not vec1 or not vec2:
        return 0.0

    # Convert to numpy arrays and normalize
    vec1_norm = normalize_vector(vec1)
    vec2_norm = normalize_vector(vec2)

    # Compute dot product (cosine similarity for normalized vectors)
    if vec1_norm.size > 0 and vec2_norm.size > 0:
        return float(np.dot(vec1_norm, vec2_norm))
    return 0.0


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
            console.print(
                f"[cyan]Exact name match for '{person_name}' with similarity: {similarity:.4f}[/cyan]"
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
        console.print(
            f"[yellow]Found similar person: '{best_match}' for '{person_name}' with similarity: {best_score:.4f}[/yellow]"
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
            console.print(
                f"[cyan]Exact match for location '{loc_name}' with similarity: {similarity:.4f}[/cyan]"
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
        console.print(
            f"[yellow]Found similar location: '{best_match[0]}' for '{loc_name}' with similarity: {best_score:.4f}[/yellow]"
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
            console.print(
                f"[cyan]Exact match for organization '{org_name}' with similarity: {similarity:.4f}[/cyan]"
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
        console.print(
            f"[yellow]Found similar organization: '{best_match[0]}' for '{org_name}' with similarity: {best_score:.4f}[/yellow]"
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
            console.print(
                f"[cyan]Exact match for event '{event_title}' with similarity: {similarity:.4f}[/cyan]"
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
        console.print(
            f"[yellow]Found similar event: '{best_match[0]}' for '{event_title}' with similarity: {best_score:.4f}[/yellow]"
        )
        return best_match, best_score

    return None, None


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
):
    embedding_model = (
        LOCAL_EMBEDDING_MODEL if model_type == "ollama" else CLOUD_EMBEDDING_MODEL
    )
    for p in extracted_people:
        person_name = p.get("name", "")
        if not person_name:
            continue

        entity_updated = False

        # Generate embedding for the person name and type
        proposed_profile_text = create_profile(
            "person", person_name, article_content, article_id, model_type
        ).get("text")
        proposed_person_embedding = embed_text(
            proposed_profile_text, model_name=embedding_model
        )

        # Find similar person using embeddings
        similar_name, similarity_score = find_similar_person(
            person_name, proposed_person_embedding, entities, similarity_threshold
        )

        if similar_name:
            console.print(
                f"[purple]Doing a final check to see if '{person_name}' is the same as '{similar_name}'[/purple]..."
            )
            result = cloud_model_check_match(
                person_name,
                similar_name,
                proposed_profile_text,
                entities["people"][similar_name]["profile"]["text"],
            )
            if result.is_match:
                console.print(
                    f"[green]The profiles match! Merging '{person_name}' with '{similar_name}'[/green]"
                )
            else:
                console.print(
                    f"[red]The profiles do not match. Skipping merge for '{person_name}'[/red]"
                )
                continue

            # We found a similar person - use that instead of creating a new one
            console.print(
                f"[blue]Merging '{person_name}' with existing person '[bold]{similar_name}[/bold]' (similarity: {similarity_score:.4f})[/blue]"
            )

            # Use the existing person's name as the key
            existing_person = entities["people"][similar_name]

            # Check if this article is already associated with the person
            article_exists = any(
                a.get("article_id") == article_id for a in existing_person["articles"]
            )

            if not article_exists:
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
                    console.print(
                        f"\n[yellow]Updating profile for person:[/] {similar_name}"
                    )
                    existing_person["profile"] = update_profile(
                        "person",
                        similar_name,
                        existing_person["profile"],
                        article_content,
                        article_id,
                        model_type,
                    )
                    existing_person["profile_embedding"] = embed_text(
                        existing_person["profile"]["text"],
                        model_name=embedding_model,
                    )

                    console.print(
                        Panel(
                            Markdown(existing_person["profile"]["text"]),
                            title=f"Updated Profile: {similar_name}",
                            border_style="yellow",
                        )
                    )
                else:
                    console.print(
                        f"\n[green]Creating initial profile for person:[/] {similar_name}"
                    )
                    existing_person["profile"] = create_profile(
                        "person", similar_name, article_content, article_id, model_type
                    )
                    existing_person["profile_embedding"] = embed_text(
                        existing_person["profile"]["text"],
                        model_name=embedding_model,
                    )
                    console.print(
                        Panel(
                            Markdown(existing_person["profile"]["text"]),
                            title=f"New Profile: {similar_name}",
                            border_style="green",
                        )
                    )

                # Store alternative names if they differ
                if person_name != similar_name:
                    if "alternative_names" not in existing_person:
                        existing_person["alternative_names"] = []

                    if person_name not in existing_person["alternative_names"]:
                        existing_person["alternative_names"].append(person_name)
                        console.print(
                            f"[blue]Added alternative name: '{person_name}' for '{similar_name}'[/blue]"
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
                write_entity_to_file("people", similar_name, existing_person)
                entities["people"][similar_name] = existing_person
                console.print(
                    f"[blue]Updated person entity saved to file:[/] {similar_name}"
                )
        else:
            # No similar person found - create new entry with initial profile
            console.print(f"\n[green]Creating profile for new person:[/] {person_name}")
            profile = create_profile(
                "person", person_name, article_content, article_id, model_type
            )
            console.print(
                Panel(
                    Markdown(profile["text"]),
                    title=f"New Profile: {person_name}",
                    border_style="green",
                )
            )

            profile_embedding = embed_text(profile["text"], model_name=embedding_model)

            new_person = {
                "name": person_name,
                "type": p.get("type", ""),
                "profile": profile,
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
                "alternative_names": [],  # Initialize empty list for alternative names
            }

            entities["people"][person_name] = new_person
            write_entity_to_file("people", person_name, new_person)
            console.print(f"[green]New person entity saved to file:[/] {person_name}")


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
):
    embedding_model = (
        LOCAL_EMBEDDING_MODEL if model_type == "ollama" else CLOUD_EMBEDDING_MODEL
    )
    for loc in extracted_locations:
        loc_name = loc.get("name", "")
        loc_type = loc.get("type", "")
        if not loc_name:
            continue
        location_key = (loc_name, loc_type)

        entity_updated = False

        # Generate embedding for the location name and type
        proposed_profile_text = create_profile(
            "location", loc_name, article_content, article_id, model_type
        ).get("text")
        proposed_location_embedding = embed_text(
            proposed_profile_text, model_name=embedding_model
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
            console.print(
                f"[purple]Doing a final check to see if '{loc_name}' is the same as '{similar_key}'[/purple]..."
            )
            result = cloud_model_check_match(
                loc_name,
                similar_key,
                proposed_profile_text,
                entities["locations"][similar_key]["profile"]["text"],
            )
            if result.is_match:
                console.print(
                    f"[green]The profiles match! Merging '{loc_name}' with '{similar_key}'[/green]"
                )
            else:
                console.print(
                    f"[red]The profiles do not match. Skipping merge for '{loc_name}'[/red]"
                )
                continue

            # We found a similar location - use that instead of creating a new one
            similar_name, similar_type = similar_key
            console.print(
                f"[blue]Merging location '{loc_name}' with existing location '[bold]{similar_name}[/bold]' (similarity: {similarity_score:.4f})[/blue]"
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
                    console.print(
                        f"\n[yellow]Updating profile for location:[/] {similar_name}"
                    )
                    existing_loc["profile"] = update_profile(
                        "location",
                        similar_name,
                        existing_loc["profile"],
                        article_content,
                        article_id,
                        model_type,
                    )
                    existing_loc["profile_embedding"] = embed_text(
                        existing_loc["profile"]["text"],
                        model_name=embedding_model,
                    )

                    console.print(
                        Panel(
                            Markdown(existing_loc["profile"]["text"]),
                            title=f"Updated Profile: {similar_name}",
                            border_style="yellow",
                        )
                    )
                else:
                    console.print(
                        f"\n[green]Creating initial profile for location:[/] {similar_name}"
                    )
                    existing_loc["profile"] = create_profile(
                        "location",
                        similar_name,
                        article_content,
                        article_id,
                        model_type,
                    )
                    existing_loc["profile_embedding"] = embed_text(
                        existing_loc["profile"]["text"],
                        model_name=embedding_model,
                    )
                    console.print(
                        Panel(
                            Markdown(existing_loc["profile"]["text"]),
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
                        console.print(
                            f"[blue]Added alternative name: '{loc_name}' (type: {loc_type}) for '{similar_name}'[/blue]"
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
                console.print(
                    f"[blue]Updated location entity saved to file:[/] {similar_name}"
                )
        else:
            # No similar location found - create new entry
            console.print(f"\n[green]Creating profile for new location:[/] {loc_name}")
            profile = create_profile(
                "location", loc_name, article_content, article_id, model_type
            )
            console.print(
                Panel(
                    Markdown(profile["text"]),
                    title=f"New Profile: {loc_name}",
                    border_style="green",
                )
            )

            profile_embedding = embed_text(profile["text"], model_name=embedding_model)

            new_location = {
                "name": loc_name,
                "type": loc_type,
                "profile": profile,
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
                "alternative_names": [],  # Initialize empty list for alternative names
            }

            entities["locations"][location_key] = new_location
            write_entity_to_file("locations", location_key, new_location)
            console.print(f"[green]New location entity saved to file:[/] {loc_name}")


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
):
    embedding_model = (
        LOCAL_EMBEDDING_MODEL if model_type == "ollama" else CLOUD_EMBEDDING_MODEL
    )
    for org in extracted_orgs:
        org_name = org.get("name", "")
        org_type = org.get("type", "")
        if not org_name:
            continue
        org_key = (org_name, org_type)

        entity_updated = False

        # Generate embedding for the organization name and type
        org_context = f"Name: {org_name}\nType: {org_type}"
        org_embedding = embed_text(org_context, model_name=embedding_model)

        # Find similar organization using embeddings
        proposed_profile_text = create_profile(
            "organization", org_name, article_content, article_id, model_type
        ).get("text")
        proposed_organization_embedding = embed_text(
            proposed_profile_text, model_name=embedding_model
        )
        similar_key, similarity_score = find_similar_organization(
            org_name,
            org_type,
            proposed_organization_embedding,
            entities,
            similarity_threshold,
        )

        if similar_key:
            console.print(
                f"[purple]Doing a final check to see if '{org_name}' is the same as '{similar_key}'[/purple]..."
            )
            result = cloud_model_check_match(
                org_name,
                similar_key,
                proposed_profile_text,
                entities["organizations"][similar_key]["profile"]["text"],
            )
            if result.is_match:
                console.print(
                    f"[green]The profiles match! Merging '{org_name}' with '{similar_key}'[/green]"
                )
            else:
                console.print(
                    f"[red]The profiles do not match. Skipping merge for '{org_name}'[/red]"
                )
                continue

            # We found a similar organization - use that instead of creating a new one
            similar_name, similar_type = similar_key
            console.print(
                f"[blue]Merging organization '{org_name}' with existing organization '[bold]{similar_name}[/bold]' (similarity: {similarity_score:.4f})[/blue]"
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
                    console.print(
                        f"\n[yellow]Updating profile for organization:[/] {similar_name}"
                    )
                    existing_org["profile"] = update_profile(
                        "organization",
                        similar_name,
                        existing_org["profile"],
                        article_content,
                        article_id,
                        model_type,
                    )
                    # Embed updated profile text using appropriate model
                    existing_org["profile_embedding"] = embed_text(
                        existing_org["profile"]["text"],
                        model_name=embedding_model,
                    )

                    console.print(
                        Panel(
                            Markdown(existing_org["profile"]["text"]),
                            title=f"Updated Profile: {similar_name}",
                            border_style="yellow",
                        )
                    )
                else:
                    console.print(
                        f"\n[green]Creating initial profile for organization:[/] {similar_name}"
                    )
                    existing_org["profile"] = create_profile(
                        "organization",
                        similar_name,
                        article_content,
                        article_id,
                        model_type,
                    )
                    console.print(
                        Panel(
                            Markdown(existing_org["profile"]["text"]),
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
                        console.print(
                            f"[blue]Added alternative name: '{org_name}' (type: {org_type}) for '{similar_name}'[/blue]"
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
                console.print(
                    f"[blue]Updated organization entity saved to file:[/] {similar_name}"
                )
        else:
            # No similar organization found - create new entry
            console.print(
                f"\n[green]Creating profile for new organization:[/] {org_name}"
            )
            profile = create_profile(
                "organization", org_name, article_content, article_id, model_type
            )
            console.print(
                Panel(
                    Markdown(profile["text"]),
                    title=f"New Profile: {org_name}",
                    border_style="green",
                )
            )

            profile_embedding = embed_text(profile["text"], model_name=embedding_model)

            new_org = {
                "name": org_name,
                "type": org_type,
                "profile": profile,
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
                "alternative_names": [],  # Initialize empty list for alternative names
            }

            entities["organizations"][org_key] = new_org
            write_entity_to_file("organizations", org_key, new_org)
            console.print(
                f"[green]New organization entity saved to file:[/] {org_name}"
            )


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
):
    embedding_model = (
        LOCAL_EMBEDDING_MODEL if model_type == "ollama" else CLOUD_EMBEDDING_MODEL
    )
    for e in extracted_events:
        event_title = e.get("title", "")
        event_start_date = e.get("start_date", "")
        event_type = e.get("event_type", "")
        if not event_title:
            continue
        event_key = (event_title, event_start_date)

        entity_updated = False

        # Generate embedding for the event title and type
        proposed_profile_text = create_profile(
            "event", event_title, article_content, article_id, model_type
        ).get("text")
        proposed_event_embedding = embed_text(
            proposed_profile_text, model_name=embedding_model
        )

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
            console.print(
                f"[purple]Doing a final check to see if '{event_title}' is the same as '{similar_key}'[/purple]..."
            )
            result = cloud_model_check_match(
                event_title,
                similar_key,
                proposed_profile_text,
                entities["events"][similar_key]["profile"]["text"],
            )
            if result.is_match:
                console.print(
                    f"[green]The profiles match! Merging '{event_title}' with '{similar_key}'[/green]"
                )
            else:
                console.print(
                    f"[red]The profiles do not match. Skipping merge for '{event_title}'[/red]"
                )
                continue

            # We found a similar event - use that instead of creating a new one
            similar_title, similar_start_date = similar_key
            console.print(
                f"[blue]Merging event '{event_title}' with existing event '[bold]{similar_title}[/bold]' (similarity: {similarity_score:.4f})[/blue]"
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
                    console.print(
                        f"\n[yellow]Updating profile for event:[/] {similar_title}"
                    )
                    existing_event["profile"] = update_profile(
                        "event",
                        similar_title,
                        existing_event["profile"],
                        article_content,
                        article_id,
                        model_type,
                    )
                    existing_event["profile_embedding"] = embed_text(
                        existing_event["profile"]["text"],
                        model_name=embedding_model,
                    )

                    console.print(
                        Panel(
                            Markdown(existing_event["profile"]["text"]),
                            title=f"Updated Profile: {similar_title}",
                            border_style="yellow",
                        )
                    )
                else:
                    console.print(
                        f"\n[green]Creating initial profile for event:[/] {similar_title}"
                    )
                    existing_event["profile"] = create_profile(
                        "event",
                        similar_title,
                        article_content,
                        article_id,
                        model_type,
                    )
                    console.print(
                        Panel(
                            Markdown(existing_event["profile"]["text"]),
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
                        console.print(
                            f"[blue]Added alternative title: '{event_title}' for '{similar_title}'[/blue]"
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
                console.print(
                    f"[blue]Updated event entity saved to file:[/] {similar_title}"
                )
        else:
            # Create new entry
            console.print(f"\n[green]Creating profile for new event:[/] {event_title}")
            profile = create_profile(
                "event", event_title, article_content, article_id, model_type
            )
            console.print(
                Panel(
                    Markdown(profile["text"]),
                    title=f"New Profile: {event_title}",
                    border_style="green",
                )
            )

            profile_embedding = embed_text(profile["text"], model_name=embedding_model)

            new_event = {
                "title": event_title,
                "description": e.get("description", ""),
                "event_type": event_type,
                "start_date": event_start_date,
                "end_date": e.get("end_date", ""),
                "is_fuzzy_date": e.get("is_fuzzy_date", False),
                "tags": e.get("tags", []),
                "profile": profile,
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
                "alternative_titles": [],  # Initialize empty list for alternative titles
            }

            entities["events"][event_key] = new_event
            write_entity_to_file("events", event_key, new_event)
            console.print(f"[green]New event entity saved to file:[/] {event_title}")
