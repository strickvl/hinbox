from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field
from sqlalchemy import JSON
from sqlmodel import Field, Relationship, SQLModel


class EventType(str, Enum):
    MEETING = "meeting"
    HUNGER_STRIKE = "hunger_strike"
    COURT_HEARING = "court_hearing"
    PROTEST = "protest"
    DETAINEE_TRANSFER = "detainee_transfer"  # inter-facility movements
    DETAINEE_RELEASE = (
        "detainee_release"  # Repatriation or transfer elsewhere off the island
    )
    INSPECTION_OR_VISIT = (
        "inspection_or_visit"  # Visits by officials, NGOs, Red Cross, etc.
    )
    PRESS_CONFERENCE = "press_conference"
    POLICY_ANNOUNCEMENT = (
        "policy_announcement"  # Executive order, new legislation, etc.
    )
    DEATH_IN_CUSTODY = "death_in_custody"  # Includes suicides if known cause
    MILITARY_OPERATION = (
        "military_operation"  # Large-scale or notable operational changes
    )
    INVESTIGATION = "investigation"  # Internal or external official investigations
    MEDICAL_EMERGENCY = "medical_emergency"  # Health crises beyond hunger strikes
    LEGAL_VERDICT = "legal_verdict"  # Court decisions
    INTERROGATION = "interrogation"  # Specific questioning sessions
    FACILITY_CHANGE = "facility_change"  # Facility openings/closures/modifications
    OTHER = "other"


class Facility(str, Enum):
    CAMP_DELTA = "Camp Delta"  # Main detention facility since 2002
    CAMP_X_RAY = "Camp X-Ray"  # Original temporary facility used Jan-Apr 2002
    CAMP_ECHO = "Camp Echo"  # Used for attorney meetings and isolation
    CAMP_IGUANA = "Camp Iguana"  # Housed juvenile detainees and cleared detainees
    CAMP_5 = "Camp 5"  # Maximum security facility
    CAMP_6 = "Camp 6"  # Medium security facility
    CAMP_7 = "Camp 7"  # High-value detainee facility


class EventSource(str, Enum):
    GOV_REPORT = "government_report"
    MEDIA = "media"
    NGO = "ngo"
    LEGAL_FILING = "legal_filing"


class OrganisationType(str, Enum):
    MILITARY = "military"  # e.g. JTF-GTMO, US Army, Navy
    INTELLIGENCE = "intelligence"  # e.g. CIA, FBI
    LEGAL = "legal"  # e.g. ACLU, CCR
    NGO = "ngo"  # e.g. Amnesty International, Human Rights Watch
    MEDIA = "media"  # e.g. Miami Herald, NY Times
    GOVERNMENT = "government"  # e.g. US DoD, DoJ
    MEDICAL = "medical"  # e.g. Physicians for Human Rights
    INTERNATIONAL = "international"  # e.g. UN, Red Cross
    OTHER = "other"


class Tags(str, Enum):
    # Legal proceedings and status
    HABEAS_CORPUS = "habeas_corpus"
    MILITARY_COMMISSION = "military_commission"

    # Treatment and conditions
    TORTURE = "torture"
    FORCE_FEEDING = "force_feeding"
    ISOLATION = "isolation"
    MEDICAL_CARE = "medical_care"
    HUNGER_STRIKE = "hunger_strike"
    SUICIDE_OR_ATTEMPT = "suicide_or_attempt"

    # Afghanistan
    AFGHANISTAN = "afghanistan"
    PALESTINE = "palestine"
    IRAQ = "iraq"
    SYRIA = "syria"
    YEMEN = "yemen"
    LIBYA = "libya"
    LEBANON = "lebanon"
    SAUDI_ARABIA = "saudi_arabia"
    UAE = "uae"

    # Administrative/operational
    TRANSFER = "transfer"
    INTERROGATION = "interrogation"

    # Rights and advocacy
    PROTEST = "protest"

    # Documentation
    MEDIA_COVERAGE = "media_coverage"
    OFFICIAL_REPORT = "official_report"
    TESTIMONY = "testimony"

    # Miscellaneous
    POLICY_CHANGE = "policy_change"
    OTHER = "other"


# Association tables for many-to-many relationships
class EventDetaineeLink(SQLModel, table=True):
    event_id: UUID = Field(foreign_key="event.id", primary_key=True)
    detainee_id: UUID = Field(foreign_key="detainee.id", primary_key=True)


class EventOrganisationLink(SQLModel, table=True):
    event_id: UUID = Field(foreign_key="event.id", primary_key=True)
    organisation_id: UUID = Field(foreign_key="organisation.id", primary_key=True)


class Location(SQLModel, table=True):
    id: Optional[UUID] = Field(default=None, primary_key=True)
    facility: Optional[Facility] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    city: str = "Guantánamo Bay"
    country: str = "Cuba"

    # Relationships
    events: List["Event"] = Relationship(back_populates="location")


class Detainee(SQLModel, table=True):
    id: UUID = Field(primary_key=True)
    name: str
    nationality: str
    birth_date: datetime
    detainee_number: str = Field(unique=True, index=True)
    detention_date: datetime
    release_date: Optional[datetime] = None
    release_status: Optional[bool] = None

    # Relationships
    events: List["Event"] = Relationship(
        back_populates="involved_detainees",
        link_model=EventDetaineeLink,
    )


class Organisation(SQLModel, table=True):
    id: UUID = Field(primary_key=True)
    name: str = Field(index=True)
    type: OrganisationType
    country: str
    description: Optional[str] = None

    # Relationships
    events: List["Event"] = Relationship(
        back_populates="involved_organisations",
        link_model=EventOrganisationLink,
    )


class Event(SQLModel, table=True):
    id: UUID = Field(primary_key=True)
    title: str = Field(index=True)
    description: str
    event_type: EventType = Field(index=True)
    start: datetime = Field(index=True)
    end: Optional[datetime] = Field(default=None)

    # Foreign keys
    location_id: Optional[UUID] = Field(default=None, foreign_key="location.id")

    # Relationships
    location: Optional[Location] = Relationship(back_populates="events")
    involved_detainees: List[Detainee] = Relationship(
        back_populates="events",
        link_model=EventDetaineeLink,
    )
    involved_organisations: List[Organisation] = Relationship(
        back_populates="events",
        link_model=EventOrganisationLink,
    )

    # Store as JSON in SQLite
    sources: List[Dict] = Field(default=[], sa_type=JSON)

    # System fields
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    verification_status: str = Field(default="unverified")
    tags: List[str] = Field(default=[], sa_type=JSON)


class EventBase(SQLModel):
    title: str = Field(index=True)
    description: str
    event_type: EventType = Field(index=True)
    start: datetime = Field(index=True)
    end: Optional[datetime] = None


class EventLite(EventBase):
    # Minimal fields for initial parsing
    pass


class ArticleEvents(BaseModel):
    events: List[EventLite]


class Article(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    title: str = Field(index=True)
    url: str = Field(unique=True, index=True)
    published_date: datetime = Field(index=True)
    content: str
    scrape_timestamp: datetime
    content_scrape_timestamp: Optional[datetime] = None
    
    # System fields
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    # Optional fields for tracking processing status
    processed: bool = Field(default=False)
    processing_timestamp: Optional[datetime] = None
