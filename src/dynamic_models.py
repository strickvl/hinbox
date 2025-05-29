"""Dynamic Pydantic model generation from domain configurations."""

from datetime import datetime
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel

from src.config_loader import get_domain_config


@lru_cache(maxsize=None)
def create_person_model(domain: str = "guantanamo") -> Type[BaseModel]:
    """Create a Person model for the specified domain."""
    config = get_domain_config(domain)
    person_types = config.get_entity_types("people")

    # Create enum for person types
    PersonType = Enum("PersonType", {ptype.upper(): ptype for ptype in person_types})

    class Person(BaseModel):
        name: str
        type: PersonType

    return Person


@lru_cache(maxsize=None)
def create_organization_model(domain: str = "guantanamo") -> Type[BaseModel]:
    """Create an Organization model for the specified domain."""
    config = get_domain_config(domain)
    org_types = config.get_entity_types("organizations")

    # Create enum for organization types
    OrganizationType = Enum(
        "OrganizationType", {otype.upper(): otype for otype in org_types}
    )

    class Organization(BaseModel):
        name: str
        type: OrganizationType

    return Organization


@lru_cache(maxsize=None)
def create_location_model(domain: str = "guantanamo") -> Type[BaseModel]:
    """Create a Location model for the specified domain."""
    config = get_domain_config(domain)
    location_types = config.get_entity_types("locations")

    # Create enum for location types
    LocationType = Enum(
        "LocationType", {ltype.upper(): ltype for ltype in location_types}
    )

    class Location(BaseModel):
        name: str
        type: LocationType

    return Location


@lru_cache(maxsize=None)
def create_event_model(domain: str = "guantanamo") -> Type[BaseModel]:
    """Create an Event model for the specified domain."""
    config = get_domain_config(domain)
    event_types = config.get_entity_types("events")

    # Create enum for event types
    EventType = Enum("EventType", {etype.upper(): etype for etype in event_types})

    # Try to get event tags if they exist
    try:
        categories_data = config.load_categories("events")
        event_tags = []
        if "event_tags" in categories_data:
            event_tags = list(categories_data["event_tags"].keys())

        if event_tags:
            EventTag = Enum("EventTag", {etag.upper(): etag for etag in event_tags})
        else:
            # Fallback - create a minimal tag enum
            EventTag = Enum("EventTag", {"OTHER": "other"})

    except Exception:
        # Fallback if no tags are defined
        EventTag = Enum("EventTag", {"OTHER": "other"})

    class Event(BaseModel):
        title: str
        description: str
        event_type: EventType
        start_date: datetime
        end_date: Optional[datetime] = None
        is_fuzzy_date: bool = False
        tags: List[EventTag] = []

    return Event


@lru_cache(maxsize=None)
def create_relevance_model(domain: str = "guantanamo") -> Type[BaseModel]:
    """Create an ArticleRelevance model for the specified domain."""

    class ArticleRelevance(BaseModel):
        is_relevant: bool
        reason: str

    return ArticleRelevance


# Convenience functions to get models
def get_person_model(domain: str = "guantanamo") -> Type[BaseModel]:
    """Get the Person model for a domain."""
    return create_person_model(domain)


def get_organization_model(domain: str = "guantanamo") -> Type[BaseModel]:
    """Get the Organization model for a domain."""
    return create_organization_model(domain)


def get_location_model(domain: str = "guantanamo") -> Type[BaseModel]:
    """Get the Location model for a domain."""
    return create_location_model(domain)


def get_event_model(domain: str = "guantanamo") -> Type[BaseModel]:
    """Get the Event model for a domain."""
    return create_event_model(domain)


def get_relevance_model(domain: str = "guantanamo") -> Type[BaseModel]:
    """Get the ArticleRelevance model for a domain."""
    return create_relevance_model(domain)


# Container models for list responses
def create_list_models(domain: str = "guantanamo") -> Dict[str, Type[BaseModel]]:
    """Create container models for list responses."""
    Person = get_person_model(domain)
    Organization = get_organization_model(domain)
    Location = get_location_model(domain)
    Event = get_event_model(domain)

    class ArticlePeople(BaseModel):
        people: List[Person]

    class ArticleOrganizations(BaseModel):
        organizations: List[Organization]

    class ArticleLocations(BaseModel):
        locations: List[Location]

    class ArticleEvents(BaseModel):
        events: List[Event]

    return {
        "people": ArticlePeople,
        "organizations": ArticleOrganizations,
        "locations": ArticleLocations,
        "events": ArticleEvents,
    }


# Backward compatibility with existing model imports
def get_models_for_domain(domain: str = "guantanamo") -> Dict[str, Any]:
    """Get all models for a domain (for backward compatibility)."""
    return {
        "Person": get_person_model(domain),
        "Organization": get_organization_model(domain),
        "Location": get_location_model(domain),
        "Event": get_event_model(domain),
        "ArticleRelevance": get_relevance_model(domain),
        **create_list_models(domain),
    }
