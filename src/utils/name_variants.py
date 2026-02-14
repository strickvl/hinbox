"""Deterministic name-variant utilities for entity deduplication.

Provides normalization, acronym detection/generation, and equivalence
expansion used by both within-article QC and the merge pipeline.
No LLM calls — everything here is fast and deterministic.
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import FrozenSet, List, Optional, Set, Tuple

# Default stopwords stripped when computing acronyms from long-form names
DEFAULT_ACRONYM_STOPWORDS: FrozenSet[str] = frozenset(
    {"the", "of", "for", "and", "to", "in", "on", "a", "an", "at", "by"}
)

# Punctuation that is removed during match normalization (not display)
_MATCH_STRIP_RE = re.compile(r"[''\".,;:!?()\[\]{}&/\\–—-]")


# ──────────────────────────────────────────────
# Core normalization
# ──────────────────────────────────────────────


def normalize_display(name: str) -> str:
    """Normalize a name for display: strip/collapse whitespace, NFC unicode.

    This is the same normalization already used in quality_controls.normalize_name.
    """
    text = str(name or "").strip()
    text = " ".join(text.split())
    return unicodedata.normalize("NFC", text)


def normalize_for_match(name: str) -> str:
    """Normalize a name for matching: lowercase, strip punctuation, collapse whitespace.

    More aggressive than display normalization — used for comparisons, not storage.
    """
    text = normalize_display(name).lower()
    text = _MATCH_STRIP_RE.sub(" ", text)
    text = " ".join(text.split())
    return text


# ──────────────────────────────────────────────
# Acronym utilities
# ──────────────────────────────────────────────


def is_acronym_form(name: str, *, min_len: int = 2, max_len: int = 10) -> bool:
    """Check if a string looks like an acronym (all uppercase letters, optional dots).

    Examples: "ICE", "FBI", "U.N.", "DoD" (mixed-case acronyms handled separately).
    """
    cleaned = name.replace(".", "").replace(" ", "").strip()
    if not cleaned:
        return False
    if len(cleaned) < min_len or len(cleaned) > max_len:
        return False
    # All uppercase letters (standard acronym form)
    if cleaned.isalpha() and cleaned.isupper():
        return True
    # Mixed-case short forms like "DoD", "DoJ" — 2-4 chars with at least
    # half uppercase
    if cleaned.isalpha() and len(cleaned) <= 4:
        upper_count = sum(1 for c in cleaned if c.isupper())
        if upper_count >= len(cleaned) / 2:
            return True
    return False


def compute_acronym(
    long_name: str,
    *,
    stopwords: FrozenSet[str] = DEFAULT_ACRONYM_STOPWORDS,
) -> Optional[str]:
    """Derive the acronym from a long-form name by taking first letters of significant words.

    Example: "Immigration and Customs Enforcement" -> "ICE"
             "Department of Homeland Security" -> "DHS"
             "Federal Bureau of Investigation" -> "FBI"

    Returns None if the result is too short (< 2 chars) or the input is already short.
    """
    words = long_name.split()
    if len(words) < 2:
        return None

    significant = [w for w in words if w.lower() not in stopwords]
    if len(significant) < 2:
        return None

    acronym = "".join(w[0].upper() for w in significant if w)
    if len(acronym) < 2:
        return None

    return acronym


def acronym_matches(
    short: str,
    long_name: str,
    *,
    stopwords: FrozenSet[str] = DEFAULT_ACRONYM_STOPWORDS,
) -> bool:
    """Check if `short` is the acronym of `long_name`.

    Case-insensitive on the acronym side. Tries both with and without
    dots/periods in the short form.
    """
    if not is_acronym_form(short):
        return False

    derived = compute_acronym(long_name, stopwords=stopwords)
    if derived is None:
        return False

    short_clean = short.replace(".", "").replace(" ", "").strip().upper()
    return short_clean == derived


# ──────────────────────────────────────────────
# Substring / containment checks
# ──────────────────────────────────────────────


def is_name_contained(
    short_name: str, long_name: str, *, min_short_len: int = 4
) -> bool:
    """Check if the shorter name is meaningfully contained in the longer name.

    Avoids false positives from very short strings (e.g., "US" in "US Army").
    Requires word-boundary containment to prevent "ICE" matching "Service".
    """
    short_match = normalize_for_match(short_name)
    long_match = normalize_for_match(long_name)

    if len(short_match) < min_short_len:
        return False

    if short_match == long_match:
        return True

    # Check word-boundary containment: "homeland security" in "department of homeland security"
    # Use word-boundary regex to avoid partial word matches
    pattern = r"\b" + re.escape(short_match) + r"\b"
    return bool(re.search(pattern, long_match))


# ──────────────────────────────────────────────
# Name signature (bundle of derived variants)
# ──────────────────────────────────────────────


@dataclass(frozen=True)
class NameSignature:
    """Bundle of derived name variants for a single entity name."""

    display: str  # normalize_display() output
    match: str  # normalize_for_match() output
    acronym: Optional[str] = None  # computed acronym if this is a long-form name
    tokens: Tuple[str, ...] = ()  # match-normalized tokens

    @property
    def is_acronym(self) -> bool:
        """Whether the display name looks like an acronym."""
        return is_acronym_form(self.display)


def build_signature(
    name: str,
    *,
    stopwords: FrozenSet[str] = DEFAULT_ACRONYM_STOPWORDS,
) -> NameSignature:
    """Build a NameSignature with all derived variants for a name."""
    display = normalize_display(name)
    match = normalize_for_match(name)
    tokens = tuple(match.split())
    acronym = compute_acronym(display, stopwords=stopwords)

    return NameSignature(
        display=display,
        match=match,
        acronym=acronym,
        tokens=tokens,
    )


# ──────────────────────────────────────────────
# Equivalence group expansion
# ──────────────────────────────────────────────


def expand_equivalents(
    name: str,
    *,
    equivalence_groups: List[List[str]],
) -> Set[str]:
    """Return {name} plus all configured synonyms in the same equivalence group.

    Matching is case-insensitive using normalize_for_match().
    Multiple groups can match (though typically a name belongs to at most one).
    """
    result: Set[str] = {name}
    name_match = normalize_for_match(name)

    for group in equivalence_groups:
        group_match = {normalize_for_match(g): g for g in group}
        if name_match in group_match:
            result.update(group)

    return result


# ──────────────────────────────────────────────
# High-level: are two names likely the same entity?
# ──────────────────────────────────────────────


def names_likely_same(
    name_a: str,
    name_b: str,
    *,
    entity_type: str = "organizations",
    equivalence_groups: Optional[List[List[str]]] = None,
    stopwords: FrozenSet[str] = DEFAULT_ACRONYM_STOPWORDS,
) -> bool:
    """Determine if two entity names likely refer to the same real-world entity.

    Uses deterministic heuristics only (no LLM):
    1. Exact match after normalization
    2. Acronym match (one is acronym of the other)
    3. Substring containment (one is contained in the other at word boundaries)
    4. Equivalence group membership (config-driven synonyms)

    Conservative for people (only exact match + equivalence groups).
    More aggressive for organizations/locations.
    """
    sig_a = build_signature(name_a, stopwords=stopwords)
    sig_b = build_signature(name_b, stopwords=stopwords)

    # 1. Exact match after normalization
    if sig_a.match == sig_b.match:
        return True

    # For people, be conservative — only exact match and equivalence groups
    if entity_type == "people":
        if equivalence_groups:
            expanded = expand_equivalents(name_a, equivalence_groups=equivalence_groups)
            if normalize_for_match(name_b) in {
                normalize_for_match(e) for e in expanded
            }:
                return True
        return False

    # 2. Acronym match (one is acronym of the other)
    if sig_a.is_acronym and not sig_b.is_acronym:
        if acronym_matches(sig_a.display, sig_b.display, stopwords=stopwords):
            return True
    elif sig_b.is_acronym and not sig_a.is_acronym:
        if acronym_matches(sig_b.display, sig_a.display, stopwords=stopwords):
            return True

    # 3. Substring containment at word boundaries
    if len(sig_a.match) != len(sig_b.match):
        short, long = (
            (sig_a, sig_b) if len(sig_a.match) < len(sig_b.match) else (sig_b, sig_a)
        )
        if is_name_contained(short.display, long.display):
            return True

    # 4. Equivalence group membership
    if equivalence_groups:
        expanded = expand_equivalents(name_a, equivalence_groups=equivalence_groups)
        if normalize_for_match(name_b) in {normalize_for_match(e) for e in expanded}:
            return True

    return False
