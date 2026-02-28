"""MergeCandidate model and JSONL persistence for merge evaluation pairs."""

import unicodedata
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel

from src.config_loader import DomainConfig


class MergeCandidate(BaseModel):
    """A pair of entity names to evaluate as a potential merge."""

    entity_name: str
    candidate_name: str
    entity_type: str  # person, organization, location, event
    source: str  # "seed" | "trace"
    pipeline_decision: Optional[str] = None
    pipeline_confidence: Optional[float] = None
    similarity_score: Optional[float] = None
    entity_context: Optional[str] = None  # profile snippet for entity A
    candidate_context: Optional[str] = None  # profile snippet for entity B
    run_id: Optional[str] = None


def candidate_key(c: MergeCandidate) -> str:
    """Deterministic key so A-vs-B == B-vs-A.

    Sorts the two names (lowercased, NFKC-normalized) and combines with entity_type.
    """
    a = unicodedata.normalize("NFKC", c.entity_name.strip().lower())
    b = unicodedata.normalize("NFKC", c.candidate_name.strip().lower())
    pair = tuple(sorted([a, b]))
    return f"{pair[0]}||{pair[1]}||{c.entity_type}"


def candidate_key_from_names(name_a: str, name_b: str, entity_type: str) -> str:
    """Build a candidate key from raw name strings (convenience wrapper)."""
    a = unicodedata.normalize("NFKC", name_a.strip().lower())
    b = unicodedata.normalize("NFKC", name_b.strip().lower())
    pair = tuple(sorted([a, b]))
    return f"{pair[0]}||{pair[1]}||{entity_type}"


def _eval_dir(domain: str) -> Path:
    """Resolve the eval directory for a domain."""
    config = DomainConfig(domain)
    return Path(config.get_output_dir()) / "eval"


def _candidates_path(domain: str) -> Path:
    return _eval_dir(domain) / "merge_candidates.jsonl"


def load_candidates(domain: str) -> List[MergeCandidate]:
    """Load merge candidates from the domain's JSONL file."""
    path = _candidates_path(domain)
    if not path.exists():
        return []
    candidates: List[MergeCandidate] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        candidates.append(MergeCandidate.model_validate_json(line))
    return candidates


def save_candidates(domain: str, candidates: List[MergeCandidate]) -> Path:
    """Write merge candidates to the domain's JSONL file (overwrites)."""
    path = _candidates_path(domain)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in candidates:
            f.write(c.model_dump_json() + "\n")
    return path
