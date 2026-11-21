"""Read the per-note tags stored on a profile (``src/engine/note_relevance.py``).

Kept apart from the tagging step so the reader and the edit guards can read
tags without importing the model code.
"""

import json
from typing import Any, Dict, Iterable, List

from src.utils.corrections import sentence_key

TAGS_KEY = "note_relevance_json"  # sentence key -> {"verdict", "about"}
MODEL_KEY = "note_relevance_model"


def stored_tags(profile: Dict[str, Any]) -> Dict[str, Dict]:
    """The profile's stored tags by note key ({} when none or unreadable)."""
    raw = (profile or {}).get(TAGS_KEY)
    try:
        tags = json.loads(raw) if raw else {}
    except (TypeError, ValueError):
        return {}
    return tags if isinstance(tags, dict) else {}


def off_topic_keys(profile: Dict[str, Any]) -> List[str]:
    """Keys of the notes tagged ``other`` on this profile."""
    tags = stored_tags(profile)
    return [k for k, v in tags.items() if (v or {}).get("verdict") == "other"]


def is_off_topic(sentence: str, off_keys: Iterable[str]) -> bool:
    """A profile sentence lies inside a note tagged ``other`` (a note can
    hold more than one sentence, so containment, not equality)."""
    key = sentence_key(sentence)
    return len(key) >= 20 and any(key in k for k in off_keys)
