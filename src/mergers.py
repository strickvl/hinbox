"""Generic entity merger classes to eliminate code duplication."""

from typing import Any, Dict, List, Optional, Tuple, Union

from src.constants import (
    SIMILARITY_THRESHOLD,
)
from src.logging_config import display_markdown, get_logger, log
from src.merge import (
    cloud_model_check_match,
    compute_similarity,
    local_model_check_match,
)
from src.profiles import create_profile, update_profile
from src.utils.embeddings import EmbeddingManager
from src.utils.file_ops import write_entity_to_file
from src.utils.profiles import extract_profile_text

# Get module-specific logger
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
        """Initialize merger for a specific entity type.

        Args:
            entity_type: The type of entity to merge (people, organizations, locations, events)
        """
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
        else:
            # Handle tuple keys
            if isinstance(self.key_field, tuple):
                return tuple(entity_dict.get(field, "") for field in self.key_field)
            else:
                return entity_dict.get(self.key_field, "")

    def _format_key_for_display(self, key: Union[str, Tuple]) -> str:
        """Format entity key for display in logs."""
        if isinstance(key, tuple):
            return key[0]  # Use first element (name/title) for display
        return key

    def find_similar_entity(
        self,
        entity_key: Union[str, Tuple],
        entity_embedding: List[float],
        entities: Dict[str, Dict],
        similarity_threshold: float = SIMILARITY_THRESHOLD,
    ) -> Tuple[Optional[Union[str, Tuple]], Optional[float]]:
        """Find the most similar entity in the entities database using embedding similarity.

        Args:
            entity_key: The key of the entity to find
            entity_embedding: Embedding vector of the entity
            entities: Dictionary of entities
            similarity_threshold: Minimum similarity score to consider a match

        Returns:
            Tuple of (matched_entity_key, similarity_score) or (None, None) if no match
        """
        if not entity_embedding or not entities[self.entity_type]:
            return None, None

        best_match = None
        best_score = 0.0

        # First check for exact key match
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

        # If no exact match or similarity below threshold, search all entities
        for existing_key, existing_entity in entities[self.entity_type].items():
            # Skip exact key match as we already checked it
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
        self, existing_entity: Dict, new_key: Union[str, Tuple]
    ) -> bool:
        """Add alternative name/title to existing entity if different."""
        if self.alternative_field not in existing_entity:
            existing_entity[self.alternative_field] = []

        if self.entity_type == "people":
            # For people, alternative_names is a simple list of strings
            if new_key not in existing_entity[self.alternative_field]:
                existing_entity[self.alternative_field].append(new_key)
                log(
                    f"Added alternative name: '{new_key}' for '{existing_entity.get('name', '')}'",
                    level="processing",
                )
                return True
        else:
            # For other entity types, alternatives are dictionaries
            if self.entity_type == "events":
                alt_entry = {
                    "title": new_key[0] if isinstance(new_key, tuple) else new_key,
                    "start_date": new_key[1]
                    if isinstance(new_key, tuple) and len(new_key) > 1
                    else "",
                    "event_type": "",  # This would need to be passed separately
                }
            else:
                # organizations, locations
                alt_entry = {
                    "name": new_key[0] if isinstance(new_key, tuple) else new_key,
                    "type": new_key[1]
                    if isinstance(new_key, tuple) and len(new_key) > 1
                    else "",
                }

            if alt_entry not in existing_entity[self.alternative_field]:
                existing_entity[self.alternative_field].append(alt_entry)
                log(
                    f"Added alternative {self.entity_type[:-1]}: '{self._format_key_for_display(new_key)}'",
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
    ):
        """Merge extracted entities with existing entities database.

        Args:
            extracted_entities: List of newly extracted entities
            entities: Existing entities database
            article_id: ID of the source article
            article_title: Title of the source article
            article_url: URL of the source article
            article_published_date: Publication date of the source article
            article_content: Full content of the source article
            extraction_timestamp: Timestamp of the extraction
            model_type: Model type to use ("gemini" or "ollama")
            similarity_threshold: Threshold for similarity matching
            domain: Domain configuration to use
        """
        # Determine embedding model type based on main model type
        embedding_model_type = "local" if model_type == "ollama" else "cloud"
        embedding_manager = EmbeddingManager(
            model_type=embedding_model_type, domain=domain
        )

        log(
            f"Starting merge_{self.entity_type} with {len(extracted_entities)} {self.entity_type} to process",
            level="processing",
        )
        log(
            f"Using model: {model_type}, embedding model: {embedding_model_type}",
            level="info",
        )

        for entity_dict in extracted_entities:
            entity_key = self._extract_key(entity_dict)
            if not entity_key or (isinstance(entity_key, str) and not entity_key):
                log(f"Skipping {self.entity_type[:-1]} with empty key", level="error")
                continue

            log(
                f"Processing {self.entity_type[:-1]}: {self._format_key_for_display(entity_key)}",
                level="processing",
            )
            entity_updated = False

            try:
                log(
                    f"Attempting to create profile for {self._format_key_for_display(entity_key)}...",
                    level="info",
                )
                proposed_profile, reflection_history = create_profile(
                    self.entity_type[:-1],
                    self._format_key_for_display(entity_key),
                    article_content,
                    article_id,
                    model_type,
                    domain,
                )

                # Extract profile text from response
                log("Extracting profile text from response...", level="info")
                proposed_profile = extract_profile_text(proposed_profile)
                if not proposed_profile or not proposed_profile.get("text"):
                    log(
                        f"Failed to generate profile for {self._format_key_for_display(entity_key)}. Profile data: {proposed_profile}",
                        level="error",
                    )
                    continue

                # Generate embedding for the entity
                proposed_profile_text = proposed_profile.get("text", "")
                log(
                    f"Generating embedding for profile text (length: {len(proposed_profile_text)})",
                    level="info",
                )
                proposed_entity_embedding = embedding_manager.embed_text(
                    proposed_profile_text
                )
                log(
                    f"Generated embedding of size: {len(proposed_entity_embedding)}",
                    level="info",
                )

                # Find similar entity using embeddings
                log(
                    f"Searching for similar {self.entity_type[:-1]} with similarity threshold: {similarity_threshold}",
                    level="info",
                )
                similar_key, similarity_score = self.find_similar_entity(
                    entity_key,
                    proposed_entity_embedding,
                    entities,
                    similarity_threshold,
                )

                if similar_key:
                    log(
                        f"Doing a final check to see if '{self._format_key_for_display(entity_key)}' is the same as '{self._format_key_for_display(similar_key)}'...",
                        level="info",
                    )

                    existing_profile = entities[self.entity_type][similar_key].get(
                        "profile", {}
                    )
                    existing_profile_text = existing_profile.get("text", "")
                    if not existing_profile_text:
                        log(
                            f"No existing profile text for {self._format_key_for_display(similar_key)}",
                            level="error",
                        )
                        continue

                    log("Performing model-based match check...", level="info")
                    if model_type == "ollama":
                        result = local_model_check_match(
                            self._format_key_for_display(entity_key),
                            self._format_key_for_display(similar_key),
                            proposed_profile_text,
                            existing_profile_text,
                        )
                    else:
                        result = cloud_model_check_match(
                            self._format_key_for_display(entity_key),
                            self._format_key_for_display(similar_key),
                            proposed_profile_text,
                            existing_profile_text,
                        )

                    log(
                        f"Match check result: {result.is_match} - {result.reason}",
                        level="info",
                    )

                    if result.is_match:
                        log(
                            f"The profiles match! Merging '{self._format_key_for_display(entity_key)}' with '{self._format_key_for_display(similar_key)}'",
                            level="success",
                        )
                    else:
                        log(
                            f"The profiles do not match. Skipping merge for '{self._format_key_for_display(entity_key)}'",
                            level="error",
                        )
                        continue

                    # We found a similar entity - merge with existing one
                    log(
                        f"Merging '{self._format_key_for_display(entity_key)}' with existing {self.entity_type[:-1]} '[bold]{self._format_key_for_display(similar_key)}[/bold]' (similarity: {similarity_score:.4f})",
                        level="processing",
                    )

                    # Use the existing entity
                    existing_entity = entities[self.entity_type][similar_key]

                    # Check if this article is already associated with the entity
                    article_exists = any(
                        a.get("article_id") == article_id
                        for a in existing_entity["articles"]
                    )

                    if not article_exists:
                        log(
                            f"Adding new article reference for {self._format_key_for_display(similar_key)}",
                            level="info",
                        )
                        existing_entity["articles"].append(
                            {
                                "article_id": article_id,
                                "article_title": article_title,
                                "article_url": article_url,
                                "article_published_date": article_published_date,
                            }
                        )
                        entity_updated = True

                        # Update profile with new information
                        if "profile" in existing_entity:
                            log(
                                f"\\n[yellow]Updating profile for {self.entity_type[:-1]}:[/] {self._format_key_for_display(similar_key)}",
                                level="info",
                            )
                            updated_profile, reflection_history = update_profile(
                                self.entity_type[:-1],
                                self._format_key_for_display(similar_key),
                                existing_entity["profile"],
                                article_content,
                                article_id,
                                model_type,
                                domain,
                            )
                            existing_entity["profile"] = updated_profile
                            log(
                                "Generating new embedding for updated profile...",
                                level="info",
                            )
                            existing_entity["profile_embedding"] = (
                                embedding_manager.embed_text(updated_profile["text"])
                            )
                            # Store reflection iteration history for debugging
                            existing_entity.setdefault("reflection_history", [])
                            existing_entity["reflection_history"].extend(
                                reflection_history
                            )

                            display_markdown(
                                updated_profile["text"],
                                title=f"Updated Profile: {self._format_key_for_display(similar_key)}",
                                style="yellow",
                            )
                        else:
                            log(
                                f"\\n[{self.log_color}]Creating initial profile for {self.entity_type[:-1]}:[/] {self._format_key_for_display(similar_key)}",
                                level="info",
                            )
                            new_profile, reflection_history = create_profile(
                                self.entity_type[:-1],
                                self._format_key_for_display(similar_key),
                                article_content,
                                article_id,
                                model_type,
                                domain,
                            )
                            existing_entity["profile"] = new_profile
                            log("Generating embedding for new profile...", level="info")
                            existing_entity["profile_embedding"] = (
                                embedding_manager.embed_text(new_profile["text"])
                            )
                            # Store reflection iteration history
                            existing_entity.setdefault("reflection_history", [])
                            existing_entity["reflection_history"].extend(
                                reflection_history
                            )

                            display_markdown(
                                new_profile["text"],
                                title=f"New Profile: {self._format_key_for_display(similar_key)}",
                                style="green",
                            )

                        # Store alternative names if they differ
                        if entity_key != similar_key:
                            if self._add_alternative_name(existing_entity, entity_key):
                                entity_updated = True

                    # Update extraction timestamp to earliest
                    existing_timestamp = existing_entity.get(
                        "extraction_timestamp", extraction_timestamp
                    )
                    if existing_timestamp != min(
                        existing_timestamp, extraction_timestamp
                    ):
                        existing_entity["extraction_timestamp"] = min(
                            existing_timestamp, extraction_timestamp
                        )
                        entity_updated = True

                    if entity_updated:
                        log(
                            f"Writing updated entity to file for {self._format_key_for_display(similar_key)}...",
                            level="info",
                        )
                        write_entity_to_file(
                            self.entity_type, similar_key, existing_entity
                        )
                        entities[self.entity_type][similar_key] = existing_entity
                        log(
                            f"[{self.log_color}]Updated {self.entity_type[:-1]} entity saved to file:[/] {self._format_key_for_display(similar_key)}",
                            level="success",
                        )

                else:
                    # No similar entity found - create new entry
                    log(
                        f"Creating new {self.entity_type[:-1]} entry for: {self._format_key_for_display(entity_key)}",
                        level="success",
                    )

                    # Create new entity dictionary with all required fields
                    new_entity = self._create_new_entity(
                        entity_dict,
                        entity_key,
                        proposed_profile,
                        proposed_entity_embedding,
                        reflection_history,
                        article_id,
                        article_title,
                        article_url,
                        article_published_date,
                        extraction_timestamp,
                    )

                    entities[self.entity_type][entity_key] = new_entity
                    write_entity_to_file(self.entity_type, entity_key, new_entity)
                    log(
                        f"[{self.log_color}]New {self.entity_type[:-1]} entity saved to file:[/] {self._format_key_for_display(entity_key)}",
                        level="info",
                    )

                    display_markdown(
                        proposed_profile["text"],
                        title=f"New Profile: {self._format_key_for_display(entity_key)}",
                        style="green",
                    )

            except Exception as e:
                log(
                    f"Error processing {self.entity_type[:-1]} {self._format_key_for_display(entity_key)}:",
                    level="error",
                )
                log(f"Error details: {str(e)}", level="error")
                import traceback

                log(f"Traceback:\\n{traceback.format_exc()}", level="error")
                continue

        log(f"Completed merge_{self.entity_type} function", level="success")

    def _create_new_entity(
        self,
        entity_dict: Dict[str, Any],
        entity_key: Union[str, Tuple],
        profile: Dict,
        profile_embedding: List[float],
        reflection_history: List,
        article_id: str,
        article_title: str,
        article_url: str,
        article_published_date: Any,
        extraction_timestamp: str,
    ) -> Dict[str, Any]:
        """Create a new entity dictionary with appropriate structure for the entity type."""
        base_entity = {
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
            self.alternative_field: [],
            "reflection_history": reflection_history or [],
        }

        if self.entity_type == "people":
            base_entity["name"] = entity_key
        elif self.entity_type == "organizations":
            base_entity["name"] = entity_key[0]
            base_entity["type"] = entity_key[1]
        elif self.entity_type == "locations":
            base_entity["name"] = entity_key[0]
            base_entity["type"] = entity_key[1]
        elif self.entity_type == "events":
            base_entity["title"] = entity_key[0]
            base_entity["start_date"] = entity_key[1]
            # Add additional event fields from the original dict
            base_entity["description"] = entity_dict.get("description", "")
            base_entity["event_type"] = entity_dict.get("event_type", "")
            base_entity["end_date"] = entity_dict.get("end_date", "")
            base_entity["is_fuzzy_date"] = entity_dict.get("is_fuzzy_date", False)
            base_entity["tags"] = entity_dict.get("tags", [])

        return base_entity


# Convenience functions for backward compatibility
def create_people_merger() -> EntityMerger:
    """Create a merger for people entities."""
    return EntityMerger("people")


def create_organizations_merger() -> EntityMerger:
    """Create a merger for organization entities."""
    return EntityMerger("organizations")


def create_locations_merger() -> EntityMerger:
    """Create a merger for location entities."""
    return EntityMerger("locations")


def create_events_merger() -> EntityMerger:
    """Create a merger for event entities."""
    return EntityMerger("events")
