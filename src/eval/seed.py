"""Generate merge candidates from existing entity Parquet data.

Two sources produce a balanced mix of likely-positive and likely-negative pairs:
  A) Fuzzy-matched entity names (score 40-90) — interesting near-misses
  B) Alias-based pairs — names the pipeline already merged (likely positives)
"""

import random
from typing import Any, Dict, List, Optional, Tuple

from rapidfuzz import fuzz

from src.frontend.data_access import get_domain_data

from .candidates import MergeCandidate, candidate_key, save_candidates

# Entity type key in parquet -> entity_type string used in MergeCandidate
_TYPE_MAP = {
    "people": "person",
    "organizations": "organization",
    "locations": "location",
    "events": "event",
}


def _get_name(entity: Dict[str, Any], entity_type_key: str) -> str:
    """Extract the display name from an entity dict."""
    if entity_type_key == "events":
        return entity.get("title", entity.get("name", ""))
    return entity.get("name", "")


def _get_profile_snippet(entity: Dict[str, Any], max_len: int = 200) -> str:
    """Extract a short profile text snippet for annotator context."""
    profile = entity.get("profile")
    if isinstance(profile, dict):
        text = profile.get("text", "")
    elif isinstance(profile, str):
        text = profile
    else:
        return ""
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


def _get_aliases(entity: Dict[str, Any]) -> List[str]:
    """Get all alternative names / aliases for an entity."""
    aliases: List[str] = []
    for field in ("aliases", "alternative_names"):
        val = entity.get(field)
        if isinstance(val, list):
            aliases.extend([str(a) for a in val if a])
    return aliases


def _fuzzy_candidates(
    entities: List[Dict[str, Any]],
    entity_type_key: str,
    min_score: float = 40.0,
    max_score: float = 90.0,
) -> List[MergeCandidate]:
    """Source A: generate pairs from fuzzy name matching."""
    names = [_get_name(e, entity_type_key) for e in entities]
    name_to_entity = {_get_name(e, entity_type_key): e for e in entities}
    candidates: List[MergeCandidate] = []
    seen_keys: set = set()
    mc_type = _TYPE_MAP[entity_type_key]

    for i, name_a in enumerate(names):
        if not name_a:
            continue
        for j in range(i + 1, len(names)):
            name_b = names[j]
            if not name_b:
                continue
            score = fuzz.token_sort_ratio(name_a.lower(), name_b.lower())
            if min_score <= score <= max_score:
                c = MergeCandidate(
                    entity_name=name_a,
                    candidate_name=name_b,
                    entity_type=mc_type,
                    source="seed",
                    similarity_score=round(score, 1),
                    entity_context=_get_profile_snippet(name_to_entity[name_a]),
                    candidate_context=_get_profile_snippet(name_to_entity[name_b]),
                )
                key = candidate_key(c)
                if key not in seen_keys:
                    seen_keys.add(key)
                    candidates.append(c)
    return candidates


def _alias_candidates(
    entities: List[Dict[str, Any]],
    entity_type_key: str,
) -> List[MergeCandidate]:
    """Source B: generate pairs from alias / alternative_names fields."""
    candidates: List[MergeCandidate] = []
    seen_keys: set = set()
    mc_type = _TYPE_MAP[entity_type_key]

    for entity in entities:
        name = _get_name(entity, entity_type_key)
        if not name:
            continue
        for alias in _get_aliases(entity):
            if alias.lower() == name.lower():
                continue
            c = MergeCandidate(
                entity_name=name,
                candidate_name=alias,
                entity_type=mc_type,
                source="seed",
                similarity_score=None,
                entity_context=_get_profile_snippet(entity),
                candidate_context=_get_profile_snippet(entity),
            )
            key = candidate_key(c)
            if key not in seen_keys:
                seen_keys.add(key)
                candidates.append(c)
    return candidates


def generate_seed_candidates(
    domain: str = "guantanamo",
    max_pairs: int = 200,
    seed: Optional[int] = 42,
) -> List[MergeCandidate]:
    """Generate and save seed merge candidates for annotation.

    Returns the full list of candidates (may be fewer than max_pairs if
    there aren't enough interesting pairs in the data).
    """
    if seed is not None:
        random.seed(seed)

    domain_data = get_domain_data(domain)
    all_fuzzy: List[MergeCandidate] = []
    all_alias: List[MergeCandidate] = []

    for type_key in _TYPE_MAP:
        entities = domain_data.get(type_key, [])
        if not entities:
            continue
        all_fuzzy.extend(_fuzzy_candidates(entities, type_key))
        all_alias.extend(_alias_candidates(entities, type_key))

    # Deduplicate across sources (alias pairs take priority)
    alias_keys = {candidate_key(c) for c in all_alias}
    unique_fuzzy = [c for c in all_fuzzy if candidate_key(c) not in alias_keys]

    # Combine: all alias pairs + sample of fuzzy pairs
    combined = list(all_alias)
    random.shuffle(unique_fuzzy)
    remaining = max_pairs - len(combined)
    if remaining > 0:
        combined.extend(unique_fuzzy[:remaining])
    else:
        combined = combined[:max_pairs]

    random.shuffle(combined)
    return combined


def seed_and_save(
    domain: str = "guantanamo",
    max_pairs: int = 200,
    seed: Optional[int] = 42,
) -> Tuple[int, str]:
    """Generate candidates and write them to disk. Returns (count, path)."""
    candidates = generate_seed_candidates(domain, max_pairs, seed)
    path = save_candidates(domain, candidates)
    return len(candidates), str(path)
