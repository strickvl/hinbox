from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple
from uuid import UUID

from pydantic import BaseModel


class EventType(str, Enum):
    MEETING = "meeting"
    HUNGER_STRIKE = "hunger_strike"
    COURT_HEARING = "court_hearing"
    PROTEST = "protest"
    DETAINEE_TRANSFER = "detainee_transfer"  # Inbound or outbound
    DETAINEE_RELEASE = (
        "detainee_release"  # Repatriation or transfer to another facility
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
    TRIBUNAL = "tribunal"  # Military commission proceedings
    INTERROGATION = "interrogation"  # Specific questioning sessions
    MEDIA_ACCESS = "media_access"  # Journalist visits/documentary shoots
    ETHICS_REVIEW = "ethics_review"  # Medical ethics committee decisions
    CAMP_OPERATION = "camp_operation"  # Facility openings/closures/modifications
    OTHER = "other"


class Facility(str, Enum):
    CAMP_DELTA = "Camp Delta"  # Main detention facility since 2002
    CAMP_X_RAY = "Camp X-Ray"  # Original temporary facility used Jan-Apr 2002
    CAMP_ECHO = "Camp Echo"  # Used for attorney meetings and isolation
    CAMP_IGUANA = "Camp Iguana"  # Housed juvenile detainees and cleared detainees
    CAMP_5 = "Camp 5"  # Maximum security facility
    CAMP_6 = "Camp 6"  # Medium security facility
    CAMP_7 = "Camp 7"  # High-value detainee facility


class Location(BaseModel):  # Normalized location model
    facility: Optional[Facility]  # Camp Delta, Camp X-Ray, etc.
    coordinates: Optional[Tuple[float, float]]
    city: str = "Guantánamo Bay"
    country: str = "Cuba"


class EventSource(str, Enum):
    GOV_REPORT = "government_report"
    MEDIA = "media"
    NGO = "ngo"
    LEGAL_FILING = "legal_filing"


class Detainee(BaseModel):
    id: UUID
    name: str
    nationality: str
    birth_date: datetime
    detainee_number: str
    detention_date: datetime
    release_date: Optional[datetime]
    release_status: Optional[bool]  # whether detainee was released or not


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


class Organisation(BaseModel):
    id: UUID
    name: str
    type: OrganisationType
    country: str
    description: Optional[str]


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


class Event(BaseModel):
    id: UUID
    title: str  # More specific than 'name'
    description: str
    classification: EventType
    start: datetime  # ISO 8601 format
    end: Optional[datetime]
    location: Location  # Replaces individual location fields

    # Relationship tracking
    involved_detainees: List[Detainee] = []
    involved_organisations: List[Organisation] = []  # CIA, JTF-GTMO, etc.

    # Source tracking
    sources: List[Dict] = [
        {"type": EventSource, "url": str, "archive_url": Optional[str]}
    ]

    # System fields
    created_at: datetime = datetime.now()
    updated_at: datetime = datetime.now()
    verification_status: str = "unverified"  # workflow state
    tags: List[Tags] = []  # "human_rights", "torture", "habeas_corpus", etc.
