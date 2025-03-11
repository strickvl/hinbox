from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


class PlaceType(str, Enum):
    COUNTRY = "country"
    PROVINCE = "province"
    STATE = "state"
    DISTRICT = "district"
    CITY = "city"
    PRISON_LOCATION = "prison_location"
    OTHER = "other"


class Place(BaseModel):
    name: str
    type: PlaceType


class PersonType(str, Enum):
    DETAINEE = "detainee"
    MILITARY = "military"
    GOVERNMENT = "government"
    LAWYER = "lawyer"
    JOURNALIST = "journalist"
    OTHER = "other"


class Person(BaseModel):
    name: str
    type: PersonType


class OrganizationType(str, Enum):
    MILITARY = "military"
    INTELLIGENCE = "intelligence"
    LEGAL = "legal"
    HUMANITARIAN = "humanitarian"
    ADVOCACY = "advocacy"
    MEDIA = "media"
    GOVERNMENT = "government"
    INTERGOVERNMENTAL = "intergovernmental"
    OTHER = "other"


class Organization(BaseModel):
    name: str
    type: OrganizationType


class EventType(str, Enum):
    TRANSFER = "transfer"  # Detainee transfers between facilities
    RELEASE = "release"  # Detainee releases
    HEARING = "hearing"  # Court hearings or legal proceedings
    HUNGER_STRIKE = "hunger_strike"  # Hunger strikes by detainees
    PROTEST = "protest"  # Protests by detainees or outside groups
    POLICY_ANNOUNCEMENT = "policy_announcement"  # New policies regarding Guantánamo
    VISIT = "visit"  # Official visits to the facility
    TORTURE_INCIDENT = "torture_incident"  # Reported torture incidents
    MEDICAL_INCIDENT = "medical_incident"  # Medical issues or incidents
    MILITARY_OPERATION = "military_operation"  # Military operations at the facility
    LEGAL_DECISION = "legal_decision"  # Legal decisions affecting detainees/operations
    MEDIA_COVERAGE = "media_coverage"  # Significant media coverage events
    OTHER = "other"  # Other types of events


class EventTag(str, Enum):
    TORTURE = "torture"  # Torture allegations or incidents
    FORCE_FEEDING = "force_feeding"  # Force-feeding of hunger strikers
    HUNGER_STRIKE = "hunger_strike"  # Hunger strike events
    TRANSFER = "transfer"  # Detainee transfer events
    INTERROGATION = "interrogation"  # Interrogation sessions
    PROTEST = "protest"  # Protest actions
    POLICY_CHANGE = "policy_change"  # Changes in detention policies
    LEGAL_CHALLENGE = "legal_challenge"  # Legal challenges to detention
    HABEAS_CORPUS = "habeas_corpus"  # Habeas corpus proceedings
    MEDICAL_CARE = "medical_care"  # Medical care issues
    ISOLATION = "isolation"  # Isolation of detainees
    SUICIDE_ATTEMPT = "suicide_attempt"  # Suicide attempts
    ABUSE = "abuse"  # Detainee abuse incidents
    OFFICIAL_STATEMENT = "official_statement"  # Official statements about Guantánamo
    OTHER = "other"  # Other tags


class Event(BaseModel):
    title: str
    description: str
    event_type: EventType
    start_date: datetime
    end_date: Optional[datetime]
    is_fuzzy_date: bool
    tags: List[EventTag]
