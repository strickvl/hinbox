"""Typed records for the research store application boundary."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List
from uuid import uuid4

from pydantic import BaseModel, Field


def new_research_id(kind: str) -> str:
    """Return an opaque, namespaced identifier that carries no identity claim."""
    return f"{kind}_{uuid4().hex}"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class AnchorKind(str, Enum):
    PERSON = "person"
    ORGANIZATION = "organization"
    PLACE = "place"
    EVENT = "event"
    SOURCE = "source"


class AnchorCreate(BaseModel):
    id: str = Field(default_factory=lambda: new_research_id("anchor"))
    kind: AnchorKind
    display_name: str = Field(min_length=1)
    aliases: List[str] = Field(default_factory=list)


class Anchor(AnchorCreate):
    created_at: datetime


class RecordVersion(BaseModel):
    id: str
    logical_record_id: str
    kind: str
    payload: Dict[str, Any]
    supersedes_id: str | None = None
    actor_id: str
    status: str
    is_current: bool
    created_at: datetime


class Representation(BaseModel):
    id: str
    source_id: str
    kind: str
    checksum: str
    selected: bool
    review_state: str
    textual_form: str
    managed_path: str
    byte_length: int
