"""Generic entity extraction classes to eliminate code duplication."""

from typing import Any, Dict, List, Optional

from src.config_loader import get_system_prompt
from src.constants import CLOUD_MODEL, OLLAMA_MODEL
from src.dynamic_models import (
    get_event_model,
    get_location_model,
    get_organization_model,
    get_person_model,
)
from src.utils.extraction import (
    extract_entities_cloud,
    extract_entities_local,
)


class EntityExtractor:
    """Generic entity extractor that handles both cloud and local models."""

    # Mapping of entity types to their model getters
    ENTITY_MODEL_GETTERS = {
        "people": get_person_model,
        "organizations": get_organization_model,
        "locations": get_location_model,
        "events": get_event_model,
    }

    # Mapping of entity types to their list attribute names
    ENTITY_LIST_ATTRIBUTES = {
        "people": "people",
        "organizations": "organizations",
        "locations": "locations",
        "events": "events",
    }

    def __init__(self, entity_type: str, domain: str = "guantanamo"):
        """Initialize extractor for a specific entity type and domain.

        Args:
            entity_type: The type of entity to extract (people, organizations,
                locations, events)
            domain: The domain configuration to use (default: guantanamo)
        """
        if entity_type not in self.ENTITY_MODEL_GETTERS:
            raise ValueError(f"Unsupported entity type: {entity_type}")

        self.entity_type = entity_type
        self.domain = domain
        self._model_getter = self.ENTITY_MODEL_GETTERS[entity_type]
        self._list_attr = self.ENTITY_LIST_ATTRIBUTES[entity_type]

    def extract_cloud(
        self,
        text: str,
        model: str = CLOUD_MODEL,
        temperature: float = 0,
        repair_hint: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Extract entities using cloud-based models (Gemini).

        Args:
            text: The text to extract entities from
            model: The cloud model to use
            temperature: Temperature for generation
            repair_hint: Optional suffix appended to the system prompt on retry

        Returns:
            List of extracted entities as dictionaries
        """
        Entity = self._model_getter(self.domain)
        system_prompt = get_system_prompt(self.entity_type, self.domain)
        if repair_hint:
            system_prompt = system_prompt + "\n\n" + repair_hint
        return extract_entities_cloud(
            text=text,
            system_prompt=system_prompt,
            response_model=List[Entity],
            model=model,
            temperature=temperature,
        )

    def extract_local(
        self,
        text: str,
        model: str = OLLAMA_MODEL,
        temperature: float = 0,
        repair_hint: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Extract entities using local models (Ollama).

        Uses the same List[Entity] response model as cloud extraction so that
        prompts (which teach bare JSON arrays) parse correctly in both modes.

        Args:
            text: The text to extract entities from
            model: The local model to use
            temperature: Temperature for generation
            repair_hint: Optional suffix appended to the system prompt on retry

        Returns:
            List of extracted entities as dictionaries
        """
        Entity = self._model_getter(self.domain)
        system_prompt = get_system_prompt(self.entity_type, self.domain)
        if repair_hint:
            system_prompt = system_prompt + "\n\n" + repair_hint
        return extract_entities_local(
            text=text,
            system_prompt=system_prompt,
            response_model=List[Entity],
            model=model,
            temperature=temperature,
        )

    def extract(
        self,
        text: str,
        model_type: str = "gemini",
        model: str = None,
        temperature: float = 0,
    ) -> List[Dict[str, Any]]:
        """Extract entities using the specified model type.

        Args:
            text: The text to extract entities from
            model_type: Either "gemini" for cloud or "ollama" for local
            model: Specific model to use (uses defaults if None)
            temperature: Temperature for generation

        Returns:
            List of extracted entities as dictionaries
        """
        if model_type == "ollama":
            return self.extract_local(
                text=text, model=model or OLLAMA_MODEL, temperature=temperature
            )
        else:
            return self.extract_cloud(
                text=text, model=model or CLOUD_MODEL, temperature=temperature
            )


# Convenience functions for backward compatibility
def create_people_extractor(domain: str = "guantanamo") -> EntityExtractor:
    """Create an extractor for people entities."""
    return EntityExtractor("people", domain)


def create_organizations_extractor(domain: str = "guantanamo") -> EntityExtractor:
    """Create an extractor for organization entities."""
    return EntityExtractor("organizations", domain)


def create_locations_extractor(domain: str = "guantanamo") -> EntityExtractor:
    """Create an extractor for location entities."""
    return EntityExtractor("locations", domain)


def create_events_extractor(domain: str = "guantanamo") -> EntityExtractor:
    """Create an extractor for event entities."""
    return EntityExtractor("events", domain)
