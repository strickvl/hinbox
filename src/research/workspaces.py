"""Typed workspace records and deterministic revision hashing."""

from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, Field

from src.utils.cache_utils import sha256_jsonable

Membership = Tuple[str, str, str]


class Workspace(BaseModel):
    id: str
    workspace_key: str
    title: str


class WorkspaceRevision(BaseModel):
    id: str
    workspace_id: str
    content_hash: str
    memberships: List[Membership] = Field(default_factory=list)
    reviewed_context: Dict[str, Any] = Field(default_factory=dict)


def revision_hash(
    memberships: List[Membership], reviewed_context: Dict[str, Any]
) -> str:
    return sha256_jsonable(
        {
            "memberships": sorted(memberships),
            "reviewed_context": reviewed_context,
        }
    )
