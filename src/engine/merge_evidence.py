"""Cheap, deterministic evidence text for entity merging, plus batch embedding.

The merger embeds this evidence text (not an LLM-written profile) to search
for existing matches, so the expensive profile call only runs once an entity
is known to be new or merged.
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Union

from src.utils.embeddings.manager import EmbeddingManager
from src.utils.step_timing import timed_step


def batch_embed_texts(
    embedding_manager: EmbeddingManager,
    texts: List[str],
    batch_size: int,
    step: str = "embed",
) -> Tuple[List[List[float]], Optional[str], Optional[int], Optional[str]]:
    """Batch-embed texts in chunks, returning (vectors, model, dim, fingerprint).

    Vectors are aligned 1:1 with the input texts list. ``step`` names the
    timing entry (see ``src/utils/step_timing.py``).
    """
    if not texts:
        return [], None, None, None

    all_embeddings: List[List[float]] = []
    model: Optional[str] = None
    dim: Optional[int] = None
    fingerprint: Optional[str] = None

    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        with timed_step(step):
            result = embedding_manager.embed_batch_result_sync(chunk)
        all_embeddings.extend(result.embeddings)
        model = result.model
        dim = result.dimension or (
            len(result.embeddings[0]) if result.embeddings else None
        )
        fingerprint = EmbeddingManager.fingerprint_from_result(result)

    return all_embeddings, model, dim, fingerprint


def extract_context_windows(
    article_content: str,
    needles: List[str],
    window_chars: int = 240,
    max_windows: int = 3,
) -> str:
    """Extract text windows around entity mentions in the article.

    Finds case-insensitive occurrences of each needle in article_content
    and returns up to max_windows non-overlapping snippets of ~window_chars
    each. Falls back to the first N characters of the article if no
    mentions are found.
    """
    if not article_content:
        return ""

    # Normalise needles: strip, dedupe, drop empties
    seen: set = set()
    clean_needles: List[str] = []
    for n in needles:
        n = n.strip()
        if n and n.lower() not in seen:
            seen.add(n.lower())
            clean_needles.append(n)

    if not clean_needles:
        return article_content[: window_chars * max_windows]

    # Collect match positions across all needles
    positions: List[Tuple[int, int]] = []
    lower_content = article_content.lower()
    for needle in clean_needles:
        for m in re.finditer(re.escape(needle.lower()), lower_content):
            positions.append((m.start(), m.end()))

    positions.sort()

    if not positions:
        return article_content[: window_chars * max_windows]

    # Build non-overlapping windows
    half = window_chars // 2
    windows: List[str] = []
    last_end = -1

    for start, end in positions:
        if len(windows) >= max_windows:
            break
        win_start = max(0, start - half)
        win_end = min(len(article_content), end + half)
        if win_start < last_end:
            continue  # overlaps previous window
        windows.append(article_content[win_start:win_end].strip())
        last_end = win_end

    return "\n\n---\n\n".join(windows)


def build_evidence_text(
    entity_type: str,
    alternative_field: str,
    entity_key: Union[str, Tuple],
    entity_dict: Dict[str, Any],
    article_content: str,
    max_chars: int = 1500,
    window_chars: int = 240,
    max_windows: int = 3,
) -> str:
    """Build a deterministic pseudo-profile from entity data + article context.

    This is the cheap alternative to create_profile() — no LLM calls,
    just string concatenation from extractor-provided structured fields
    plus context windows around entity mentions in the article.
    """
    # Assemble search needles from key + any alternative names
    needles: List[str] = []
    if isinstance(entity_key, tuple):
        needles.append(str(entity_key[0]))
    else:
        needles.append(str(entity_key))

    # Include extractor-provided alternates when present
    for alt in entity_dict.get(alternative_field, []):
        if isinstance(alt, dict):
            needles.append(str(alt.get("name", alt.get("title", ""))))
        elif isinstance(alt, str):
            needles.append(alt)

    context = extract_context_windows(
        article_content, needles, window_chars=window_chars, max_windows=max_windows
    )

    # Build header by entity type
    parts: List[str] = []

    if entity_type == "people":
        parts.append(f"Name: {entity_key}")
    elif entity_type in ("organizations", "locations"):
        name = entity_key[0] if isinstance(entity_key, tuple) else str(entity_key)
        etype = (
            entity_key[1]
            if isinstance(entity_key, tuple) and len(entity_key) > 1
            else ""
        )
        parts.append(f"Name: {name}")
        if etype:
            parts.append(f"Type: {etype}")
    elif entity_type == "events":
        title = entity_key[0] if isinstance(entity_key, tuple) else str(entity_key)
        start_date = (
            entity_key[1]
            if isinstance(entity_key, tuple) and len(entity_key) > 1
            else ""
        )
        parts.append(f"Title: {title}")
        if start_date:
            parts.append(f"Start date: {start_date}")
        event_type = entity_dict.get("event_type", "")
        if event_type:
            parts.append(f"Event type: {event_type}")
        description = entity_dict.get("description", "")
        if description:
            parts.append(f"Description: {description}")

    if context:
        parts.append(f"\nCONTEXT:\n{context}")

    text = "\n".join(parts)
    return text[:max_chars]
