"""GoldLabel model and append-only JSONL persistence for merge annotations."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from pydantic import BaseModel

from src.config_loader import DomainConfig

from .candidates import candidate_key_from_names


class GoldLabel(BaseModel):
    """A human annotation for whether two entity names refer to the same entity."""

    entity_name: str
    candidate_name: str
    entity_type: str
    label: str  # "yes" | "no" | "unsure"
    notes: str = ""
    pipeline_decision: Optional[str] = None
    pipeline_confidence: Optional[float] = None
    similarity_score: Optional[float] = None
    timestamp: str


def _eval_dir(domain: str) -> Path:
    config = DomainConfig(domain)
    return Path(config.get_output_dir()) / "eval"


def _gold_path(domain: str) -> Path:
    return _eval_dir(domain) / "merge_gold.jsonl"


def append_gold_label(domain: str, label: GoldLabel) -> None:
    """Append a single gold label to the domain's JSONL file."""
    path = _gold_path(domain)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(label.model_dump_json() + "\n")


def load_gold_labels(domain: str) -> Dict[str, GoldLabel]:
    """Load gold labels, returning a dict keyed by candidate_key.

    Last-write-wins: if the same pair is labelled multiple times,
    the most recent entry (last in the file) takes precedence.
    """
    path = _gold_path(domain)
    if not path.exists():
        return {}
    labels: Dict[str, GoldLabel] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        gl = GoldLabel.model_validate_json(line)
        key = candidate_key_from_names(
            gl.entity_name, gl.candidate_name, gl.entity_type
        )
        labels[key] = gl
    return labels


def make_gold_label(
    entity_name: str,
    candidate_name: str,
    entity_type: str,
    label: str,
    notes: str = "",
    pipeline_decision: Optional[str] = None,
    pipeline_confidence: Optional[float] = None,
    similarity_score: Optional[float] = None,
) -> GoldLabel:
    """Create a GoldLabel with the current UTC timestamp."""
    return GoldLabel(
        entity_name=entity_name,
        candidate_name=candidate_name,
        entity_type=entity_type,
        label=label,
        notes=notes,
        pipeline_decision=pipeline_decision,
        pipeline_confidence=pipeline_confidence,
        similarity_score=similarity_score,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
