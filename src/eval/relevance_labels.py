"""RelevanceLabel model and append-only JSONL persistence for relevance annotations."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from pydantic import BaseModel

from src.config_loader import DomainConfig


class RelevanceLabel(BaseModel):
    """A human annotation for whether an article is relevant to the domain."""

    article_id: str
    label: str  # "relevant" | "irrelevant" | "noisy"
    notes: str = ""
    timestamp: str


def _eval_dir(domain: str) -> Path:
    """Resolve the eval directory for a domain."""
    config = DomainConfig(domain)
    return Path(config.get_output_dir()) / "eval"


def _gold_path(domain: str) -> Path:
    return _eval_dir(domain) / "relevance_gold.jsonl"


def append_relevance_label(domain: str, label: RelevanceLabel) -> None:
    """Append a single relevance label to the domain's JSONL file."""
    path = _gold_path(domain)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(label.model_dump_json() + "\n")


def load_relevance_labels(domain: str) -> Dict[str, RelevanceLabel]:
    """Load relevance labels, returning a dict keyed by article_id.

    Last-write-wins: if the same article is labelled multiple times,
    the most recent entry (last in the file) takes precedence.
    """
    path = _gold_path(domain)
    if not path.exists():
        return {}
    labels: Dict[str, RelevanceLabel] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rl = RelevanceLabel.model_validate_json(line)
        labels[rl.article_id] = rl
    return labels


def make_relevance_label(
    article_id: str,
    label: str,
    notes: str = "",
) -> RelevanceLabel:
    """Create a RelevanceLabel with the current UTC timestamp."""
    return RelevanceLabel(
        article_id=article_id,
        label=label,
        notes=notes,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
