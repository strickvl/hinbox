"""RelevanceCandidate model and JSONL persistence for relevance evaluation."""

from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel

from src.config_loader import DomainConfig


class RelevanceCandidate(BaseModel):
    """An article to evaluate for domain relevance."""

    article_id: str
    title: str
    content_snippet: str  # first ~600 chars of content
    url: Optional[str] = None
    published_date: Optional[str] = None
    content_length: int = 0  # full article length for context


def _eval_dir(domain: str) -> Path:
    """Resolve the eval directory for a domain."""
    config = DomainConfig(domain)
    return Path(config.get_output_dir()) / "eval"


def _candidates_path(domain: str) -> Path:
    return _eval_dir(domain) / "relevance_candidates.jsonl"


def load_relevance_candidates(domain: str) -> List[RelevanceCandidate]:
    """Load relevance candidates from the domain's JSONL file."""
    path = _candidates_path(domain)
    if not path.exists():
        return []
    candidates: List[RelevanceCandidate] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        candidates.append(RelevanceCandidate.model_validate_json(line))
    return candidates


def save_relevance_candidates(
    domain: str, candidates: List[RelevanceCandidate]
) -> Path:
    """Write relevance candidates to the domain's JSONL file (overwrites)."""
    path = _candidates_path(domain)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in candidates:
            f.write(c.model_dump_json() + "\n")
    return path
