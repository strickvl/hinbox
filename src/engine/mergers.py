"""Generic entity merger classes to eliminate code duplication."""

from typing import Any, Dict, List, Optional, Tuple, Union

from src.config_loader import DomainConfig
from src.constants import ENABLE_PROFILE_VERSIONING, SIMILARITY_THRESHOLD
from src.engine.match_checker import cloud_model_check_match, local_model_check_match
from src.engine.profiles import VersionedProfile, create_profile, update_profile
from src.logging_config import display_markdown, get_logger, log
from src.utils.embeddings.similarity import compute_similarity, get_embedding_manager
from src.utils.file_ops import write_entity_to_file
from src.utils.profiles import extract_profile_text

# Module-specific logger
logger = get_logger("mergers")


class EntityMerger:
    """Generic entity merger that handles similarity matching and profile merging."""

    # Configuration for different entity types
    ENTITY_CONFIGS = {
        "people": {
            "key_field": "name",
            "key_type": str,
            "alternative_field": "alternative_names",
            "log_color": "green",
        },
        "organizations": {
            "key_field": ("name", "type"),
            "key_type": tuple,
            "alternative_field": "alternative_names",
            "log_color": "blue",
        },
        "locations": {
            "key_field": ("name", "type"),
            "key_type": tuple,
            "alternative_field": "alternative_names",
            "log_color": "blue",
        },
        "events": {
            "key_field": ("title", "start_date"),
            "key_type": tuple,
            "alternative_field": "alternative_titles",
            "log_color": "blue",
        },
    }

    def __init__(self, entity_type: str):
        """Initialize merger for a specific entity type."""
        if entity_type not in self.ENTITY_CONFIGS:
            raise ValueError(f"Unsupported entity type: {entity_type}")

        self.entity_type = entity_type
        self.config = self.ENTITY_CONFIGS[entity_type]
        self.key_field = self.config["key_field"]
        self.key_type = self.config["key_type"]
        self.alternative_field = self.config["alternative_field"]
        self.log_color = self.config["log_color"]

    def _extract_key(self, entity_dict: Dict[str, Any]) -> Union[str, Tuple[str, str]]:
        """Extract the entity key from an entity dictionary."""
        if self.key_type == str:
            return entity_dict.get(self.key_field, "")
        # tuple key
        return tuple(entity_dict.get(field, "") for field in self.key_field)  # type: ignore[index]

    def _format_key_for_display(self, key: Union[str, Tuple]) -> str:
        """Format entity key for display in logs."""
        return key[0] if isinstance(key, tuple) else key

    def find_similar_entity(
        self,
        entity_key: Union[str, Tuple],
        entity_embedding: List[float],
        entities: Dict[str, Dict],
        similarity_threshold: float = SIMILARITY_THRESHOLD,
    ) -> Tuple[Optional[Union[str, Tuple]], Optional[float]]:
        """Find the most similar entity in the entities database using embedding similarity."""
        if not entity_embedding or not entities[self.entity_type]:
            return None, None

        best_match: Optional[Union[str, Tuple]] = None
        best_score = 0.0

        # Exact key match first
        if entity_key in entities[self.entity_type]:
            existing_entity = entities[self.entity_type][entity_key]
            if "profile_embedding" in existing_entity:
                similarity = compute_similarity(
                    entity_embedding, existing_entity["profile_embedding"]
                )
                log(
                    f"Exact match for {self.entity_type[:-1]} '{self._format_key_for_display(entity_key)}' with similarity: {similarity:.4f}",
                    level="info",
                )
                if similarity >= similarity_threshold:
                    return entity_key, similarity

        # Otherwise scan all
        for existing_key, existing_entity in entities[self.entity_type].items():
            if existing_key == entity_key:
                continue
            if "profile_embedding" in existing_entity:
                similarity = compute_similarity(
                    entity_embedding, existing_entity["profile_embedding"]
                )
                if similarity > best_score:
                    best_score = similarity
                    best_match = existing_key

        if best_match and best_score >= similarity_threshold:
            log(
                f"Found similar {self.entity_type[:-1]}: '{self._format_key_for_display(best_match)}' for '{self._format_key_for_display(entity_key)}' with similarity: {best_score:.4f}",
                level="warning",
            )
            return best_match, best_score

        return None, None

    def _add_alternative_name(
        self,
        existing_entity: Dict[str, Any],
        new_key: Union[str, Tuple],
        source_entity: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Add alternative name/title to existing entity if different.
        Matches legacy shape:
          • people: List[str]
          • locations/orgs: List[{'name','type'}]
          • events: List[{'title','start_date','event_type'}]
        """
        if self.alternative_field not in existing_entity:
            existing_entity[self.alternative_field] = []

        if self.entity_type == "people":
            if new_key not in existing_entity[self.alternative_field]:
                existing_entity[self.alternative_field].append(new_key)
                log(f"Added alternative name: '{new_key}'", level="processing")
                return True

        elif self.entity_type == "events":
            alt_entry = {
                "title": new_key[0] if isinstance(new_key, tuple) else str(new_key),
                "start_date": new_key[1]
                if isinstance(new_key, tuple) and len(new_key) > 1
                else "",
                "event_type": (source_entity or {}).get("event_type", ""),
            }
            if alt_entry not in existing_entity[self.alternative_field]:
                existing_entity[self.alternative_field].append(alt_entry)
                log(
                    f"Added alternative event: '{alt_entry['title']}'",
                    level="processing",
                )
                return True

        else:  # organizations, locations
            alt_entry = {
                "name": new_key[0] if isinstance(new_key, tuple) else str(new_key),
                "type": new_key[1]
                if isinstance(new_key, tuple) and len(new_key) > 1
                else "",
            }
            if alt_entry not in existing_entity[self.alternative_field]:
                existing_entity[self.alternative_field].append(alt_entry)
                log(
                    f"Added alternative {self.entity_type[:-1]}: '{alt_entry['name']}'",
                    level="processing",
                )
                return True

        return False

    def merge_entities(
        self,
        extracted_entities: List[Dict[str, Any]],
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
        *,
        langfuse_session_id: Optional[str] = None,
        langfuse_trace_id: Optional[str] = None,
    ) -> None:
        """Merge extracted entities with existing entities database."""
        # Preserve legacy logging of "chosen mode"; EmbeddingManager itself is obtained via shared getter.
        embedding_model_type = "local" if model_type == "ollama" else "cloud"
        embedding_manager = get_embedding_manager(domain=domain)

        log(
            f"Starting merge_{self.entity_type} with {len(extracted_entities)} {self.entity_type} to process",
            level="processing",
        )
        base_dir = DomainConfig(domain).get_output_dir()
        log(
            f"Using model: {model_type}, embedding model: {embedding_model_type}",
            level="info",
        )

        for entity_dict in extracted_entities:
            entity_key = self._extract_key(entity_dict)
            if not entity_key or (
                isinstance(entity_key, str) and not entity_key.strip()
            ):
                log(f"Skipping {self.entity_type[:-1]} with empty key", level="error")
                continue

            log(
                f"Processing {self.entity_type[:-1]}: {self._format_key_for_display(entity_key)}",
                level="processing",
            )
            entity_updated = False

            # --- Profile generation ---
            proposed_profile, proposed_versioned_profile, reflection_history = (
                create_profile(
                    self.entity_type[:-1],
                    self._format_key_for_display(entity_key),
                    article_content,
                    article_id,
                    model_type,
                    domain,
                    langfuse_session_id=langfuse_session_id,
                    langfuse_trace_id=langfuse_trace_id,
                )
            )

            proposed_profile = extract_profile_text(proposed_profile)
            if not proposed_profile or not proposed_profile.get("text"):
                log(
                    f"Failed to generate profile for {self._format_key_for_display(entity_key)}. Profile data: {proposed_profile}",
                    level="error",
                )
                continue

            # --- Embedding computation ---
            proposed_profile_text = proposed_profile["text"]
            proposed_entity_embedding = embedding_manager.embed_text_sync(
                proposed_profile_text
            )

            # --- Similarity search ---
            similar_key, similarity_score = self.find_similar_entity(
                entity_key, proposed_entity_embedding, entities, similarity_threshold
            )

            if similar_key:
                # Final model-based match check
                existing_profile_text = (
                    entities[self.entity_type][similar_key]
                    .get("profile", {})
                    .get("text", "")
                )
                if not existing_profile_text:
                    log(
                        f"No existing profile text for {self._format_key_for_display(similar_key)}",
                        level="error",
                    )
                    continue

                if model_type == "ollama":
                    result = local_model_check_match(
                        self._format_key_for_display(entity_key),
                        self._format_key_for_display(similar_key),
                        proposed_profile_text,
                        existing_profile_text,
                        langfuse_session_id=langfuse_session_id,
                        langfuse_trace_id=langfuse_trace_id,
                    )
                else:
                    result = cloud_model_check_match(
                        self._format_key_for_display(entity_key),
                        self._format_key_for_display(similar_key),
                        proposed_profile_text,
                        existing_profile_text,
                        langfuse_session_id=langfuse_session_id,
                        langfuse_trace_id=langfuse_trace_id,
                    )

                log(
                    f"Match check result: {result.is_match} - {result.reason}",
                    level="info",
                )
                if not result.is_match:
                    log(
                        f"The profiles do not match. Skipping merge for '{self._format_key_for_display(entity_key)}'",
                        level="error",
                    )
                    continue

                log(
                    f"Merging '{self._format_key_for_display(entity_key)}' with existing {self.entity_type[:-1]} "
                    f"'[bold]{self._format_key_for_display(similar_key)}[/bold]' (similarity: {similarity_score:.4f})",
                    level="processing",
                )
                existing_entity = entities[self.entity_type][similar_key]

                # Ensure article is linked
                article_exists = any(
                    a.get("article_id") == article_id
                    for a in existing_entity["articles"]
                )
                if not article_exists:
                    existing_entity["articles"].append(
                        {
                            "article_id": article_id,
                            "article_title": article_title,
                            "article_url": article_url,
                            "article_published_date": article_published_date,
                        }
                    )
                    entity_updated = True

                    # Update profile
                    if "profile" in existing_entity:
                        # Load or create versioned profile
                        versioned_profile = (
                            VersionedProfile(**existing_entity["profile_versions"])
                            if ENABLE_PROFILE_VERSIONING
                            and "profile_versions" in existing_entity
                            else VersionedProfile()
                        )

                        updated_profile, versioned_profile, reflection_history = (
                            update_profile(
                                self.entity_type[:-1],
                                self._format_key_for_display(similar_key),
                                existing_entity["profile"],
                                versioned_profile,
                                article_content,
                                article_id,
                                model_type,
                                domain,
                                langfuse_session_id=langfuse_session_id,
                                langfuse_trace_id=langfuse_trace_id,
                            )
                        )
                        existing_entity["profile"] = updated_profile
                        existing_entity["profile_versions"] = (
                            versioned_profile.model_dump()
                            if ENABLE_PROFILE_VERSIONING
                            else None
                        )
                        existing_entity["profile_embedding"] = (
                            embedding_manager.embed_text_sync(updated_profile["text"])
                        )

                        # Reflection history
                        existing_entity.setdefault("reflection_history", [])
                        existing_entity["reflection_history"].extend(reflection_history)

                        display_markdown(
                            updated_profile["text"],
                            title=f"Updated Profile: {self._format_key_for_display(similar_key)}",
                            style="yellow",
                        )
                    else:
                        new_profile, new_versioned_profile, reflection_history = (
                            create_profile(
                                self.entity_type[:-1],
                                self._format_key_for_display(similar_key),
                                article_content,
                                article_id,
                                model_type,
                                domain,
                                langfuse_session_id=langfuse_session_id,
                                langfuse_trace_id=langfuse_trace_id,
                            )
                        )
                        existing_entity["profile"] = new_profile
                        existing_entity["profile_versions"] = (
                            new_versioned_profile.model_dump()
                            if ENABLE_PROFILE_VERSIONING
                            else None
                        )
                        existing_entity["profile_embedding"] = (
                            embedding_manager.embed_text_sync(new_profile["text"])
                        )

                        existing_entity.setdefault("reflection_history", [])
                        existing_entity["reflection_history"].extend(reflection_history)

                        display_markdown(
                            new_profile["text"],
                            title=f"New Profile: {self._format_key_for_display(similar_key)}",
                            style="green",
                        )

                    # Alternative names/titles if different
                    if entity_key != similar_key:
                        if self._add_alternative_name(
                            existing_entity, entity_key, source_entity=entity_dict
                        ):
                            entity_updated = True

                # Keep earliest extraction timestamp
                existing_timestamp = existing_entity.get(
                    "extraction_timestamp", extraction_timestamp
                )
                earliest = min(existing_timestamp, extraction_timestamp)
                if existing_timestamp != earliest:
                    existing_entity["extraction_timestamp"] = earliest
                    entity_updated = True

                if entity_updated:
                    write_entity_to_file(
                        self.entity_type, similar_key, existing_entity, base_dir
                    )
                    entities[self.entity_type][similar_key] = existing_entity
                    log(
                        f"[{self.log_color}]Updated {self.entity_type[:-1]} entity saved to file:[/] "
                        f"{self._format_key_for_display(similar_key)}",
                        level="success",
                    )

            else:
                # Create new entity
                log(
                    f"Creating new {self.entity_type[:-1]} entry for: {self._format_key_for_display(entity_key)}",
                    level="success",
                )

                new_entity = self._create_new_entity(
                    entity_dict,
                    entity_key,
                    proposed_profile,
                    proposed_versioned_profile,
                    proposed_entity_embedding,
                    reflection_history,
                    article_id,
                    article_title,
                    article_url,
                    article_published_date,
                    extraction_timestamp,
                )

                entities[self.entity_type][entity_key] = new_entity
                write_entity_to_file(self.entity_type, entity_key, new_entity, base_dir)
                log(
                    f"[{self.log_color}]New {self.entity_type[:-1]} entity saved to file:[/] "
                    f"{self._format_key_for_display(entity_key)}",
                    level="info",
                )
                display_markdown(
                    proposed_profile["text"],
                    title=f"New Profile: {self._format_key_for_display(entity_key)}",
                    style="green",
                )

        log(f"Completed merge_{self.entity_type} function", level="success")

    def _create_new_entity(
        self,
        entity_dict: Dict[str, Any],
        entity_key: Union[str, Tuple],
        profile: Dict[str, Any],
        versioned_profile: VersionedProfile,
        profile_embedding: List[float],
        reflection_history: List[Any],
        article_id: str,
        article_title: str,
        article_url: str,
        article_published_date: Any,
        extraction_timestamp: str,
    ) -> Dict[str, Any]:
        """Create a new entity dictionary with appropriate structure for each entity type."""
        base_entity: Dict[str, Any] = {
            "profile": profile,
            "profile_versions": versioned_profile.model_dump()
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
            self.alternative_field: [],
            "reflection_history": reflection_history or [],
        }

        if self.entity_type == "people":
            base_entity["name"] = entity_key
        elif self.entity_type in ("organizations", "locations"):
            base_entity["name"] = entity_key[0]
            base_entity["type"] = entity_key[1]
        elif self.entity_type == "events":
            base_entity["title"] = entity_key[0]
            base_entity["start_date"] = entity_key[1]
            base_entity["description"] = entity_dict.get("description", "")
            base_entity["event_type"] = entity_dict.get("event_type", "")
            base_entity["end_date"] = entity_dict.get("end_date", "")
            base_entity["is_fuzzy_date"] = entity_dict.get("is_fuzzy_date", False)
            base_entity["tags"] = entity_dict.get("tags", [])

        return base_entity


# Convenience creators (unchanged behavior)
def create_people_merger() -> EntityMerger:
    return EntityMerger("people")


def create_organizations_merger() -> EntityMerger:
    return EntityMerger("organizations")


def create_locations_merger() -> EntityMerger:
    return EntityMerger("locations")


def create_events_merger() -> EntityMerger:
    return EntityMerger("events")
