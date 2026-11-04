"""Deterministic name-variant utilities for entity deduplication.

Provides normalization, acronym detection/generation, and equivalence
expansion used by both within-article QC and the merge pipeline.
No LLM calls — everything here is fast and deterministic.
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Callable, FrozenSet, List, Optional, Set, Tuple

from rapidfuzz.distance import Levenshtein

# Default stopwords stripped when computing acronyms from long-form names
DEFAULT_ACRONYM_STOPWORDS: FrozenSet[str] = frozenset(
    {"the", "of", "for", "and", "to", "in", "on", "a", "an", "at", "by"}
)

# Punctuation that is removed during match normalization (not display)
_MATCH_STRIP_RE = re.compile(r"['\u2018\u2019\u201c\u201d\".,;:!?()\[\]{}&/\\–—-]")


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


def fold_accents(text: str) -> str:
    """Drop accents so "Guantánamo" and "Guantanamo" compare equal."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def normalize_for_match(name: str) -> str:
    """Normalize a name for matching: drop accents, lowercase, strip punctuation.

    More aggressive than display normalization — used for comparisons, not storage.
    """
    text = fold_accents(normalize_display(name)).lower()
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


# ──────────────────────────────────────────────
# Name quality assessment
# ──────────────────────────────────────────────

# Generic plural heads that indicate a category, not a specific entity.
# Only flagged when they appear as the LAST word (head noun) of the name.
_GENERIC_PLURAL_HEADS: FrozenSet[str] = frozenset(
    {
        "departments",
        "agencies",
        "officials",
        "authorities",
        "forces",
        "organizations",
        "institutions",
        "offices",
        "committees",
        "groups",
        "teams",
        "units",
        "branches",
        "divisions",
    }
)

# Preposition patterns that signal a descriptive phrase rather than a proper noun.
# e.g. "military base in Guantánamo Bay", "prison near Havana"
_DESCRIPTIVE_LOCATION_RE = re.compile(
    r"(?i)^(?:(?:u\.s\.?|american|cuban|military)\s+)?"
    r"(?:military\s+)?(?:base|prison|facility|camp|detention\s+center|jail|compound|complex|site)"
    r"\s+(?:in|at|near|outside|on)\s+",
)

# Leading articles to strip for scoring purposes
_LEADING_ARTICLE_RE = re.compile(r"(?i)^the\s+")


def is_low_quality_name(name: str, *, entity_type: str = "organizations") -> bool:
    """Check if a name is likely a generic/descriptive phrase rather than a proper entity name.

    Returns True for:
    - Generic plurals used as the head noun: "Defense departments", "security agencies"
    - Descriptive location phrases: "U.S. military base in Guantánamo Bay"

    Conservative: only flags clear-cut cases to avoid false positives.
    """
    if not name or not name.strip():
        return False

    cleaned = name.strip()
    words = cleaned.split()

    # Generic plural head noun (last word is a generic plural)
    if len(words) >= 2 and words[-1].lower() in _GENERIC_PLURAL_HEADS:
        return True

    # Descriptive location phrase (entity_type-agnostic check)
    if _DESCRIPTIVE_LOCATION_RE.match(cleaned):
        return True

    return False


def strip_leading_article(name: str) -> str:
    """Strip a leading 'the' from a name for scoring.

    'the Pentagon' → 'Pentagon', 'The New York Times' → 'New York Times'
    """
    return _LEADING_ARTICLE_RE.sub("", name).strip()


# Metonymic / contextual location suffixes.
# These indicate a colloquial reference ("U.S. soil", "Cuban waters")
# rather than a proper name.
_CONTEXTUAL_SUFFIXES: FrozenSet[str] = frozenset(
    {"soil", "territory", "waters", "border", "grounds", "arena", "area"}
)


def is_shouting_name(name: str) -> bool:
    """A name in ALL CAPS that isn't an acronym, often copied from a dateline:
    "GUANTANAMO BAY NAVY BASE" (yes), "JTF GTMO" / "FBI" (no)."""
    words = [w for w in re.split(r"[^A-Za-z]+", fold_accents(name)) if w]
    return (
        len(words) >= 2
        and not is_acronym_form(name)
        and all(w.isupper() for w in words)
        and any(len(w) >= 5 for w in words)
    )


def score_canonical_name(name: str) -> float:
    """Score how 'canonical' a name is.  Higher = better.

    Shared scoring function used by both within-article variant collapse
    and the merge-time canonical key picker.

    Signals:
    - Length bonus (normalized, capped at 50 chars): longer = more formal
    - Acronym penalty (-2.0): abbreviations like "ICE", "DHS"
    - Contextual suffix penalty (-3.0): metonymic references like "U.S. soil"
    - Low-quality name penalty (-4.0): generic plurals and descriptive phrases
    - ALL CAPS penalty (-0.5): "GUANTANAMO BAY NAVY BASE" loses to the same
      name in normal case, but not to a shorter alias like "Gitmo"
    """
    score = 0.0

    # Prefer longer names (more specific / formal)
    score += min(len(name) / 50.0, 1.0)

    # Penalize pure acronym forms (ICE, DHS, FBI ...)
    if is_acronym_form(name):
        score -= 2.0

    # Penalize metonymic / contextual location suffixes
    words = name.lower().split()
    if words and words[-1] in _CONTEXTUAL_SUFFIXES:
        score -= 3.0

    # Penalize generic plurals and descriptive phrases
    if is_low_quality_name(name):
        score -= 4.0

    if is_shouting_name(name):
        score -= 0.5

    return score


# ──────────────────────────────────────────────
# Spelling variants worth a match check
# ──────────────────────────────────────────────

# Words that only say what kind of place or body a name is ("Broward County"
# is "Broward"). Dropped from the end of location/organization names.
_TRAILING_DESIGNATORS: FrozenSet[str] = frozenset({"county", "city"})
# Prefixes that don't change which entity is meant ("U.S. Navy" is "Navy").
_LEADING_DESIGNATORS: Tuple[Tuple[str, ...], ...] = (("the",), ("u", "s"), ("us",))
# Titles in front of a person's name ("President Trump", "Judge Pohl").
_PERSON_TITLES: FrozenSet[str] = frozenset(
    {"mr", "mrs", "ms", "dr", "judge", "president", "sen", "senator", "rep", "gov"}
    | {"governor", "gen", "general", "col", "colonel", "capt", "lt", "sgt", "adm"}
)


# Words that start a trailing description ("Navy base in southeast Cuba").
_QUALIFIER_WORDS: FrozenSet[str] = frozenset({"in", "at", "near", "outside", "on"})
_ORDINAL_RE = re.compile(r"^(\d+)(?:st|nd|rd|th)$")


def _place_core(name: str) -> str:
    """Match form without leading "the"/"U.S.", a trailing "County"/"City",
    or a trailing "in/at/near …" description after two or more words."""
    tokens = normalize_for_match(name).split()
    for prefix in _LEADING_DESIGNATORS:
        if len(tokens) > len(prefix) and tuple(tokens[: len(prefix)]) == prefix:
            tokens = tokens[len(prefix) :]
            break
    cut = next((i for i, t in enumerate(tokens) if i >= 2 and t in _QUALIFIER_WORDS), 0)
    if cut:
        tokens = tokens[:cut]
    if len(tokens) > 1 and tokens[-1] in _TRAILING_DESIGNATORS:
        tokens = tokens[:-1]
    return " ".join(tokens)


def _same_word_spelled_differently(a: str, b: str, *, edits: bool) -> bool:
    """Whether two words are one word spelled two ways.

    Plural, ordinal or shortened forms: "Commission"/"Commissions", "11"/"11th",
    "Counterterror"/"Counterterrorism". With ``edits``, also one typo apart:
    "Shelley"/"Shelly", "Rosa"/"Rose" (used for people's names only).
    """
    if a == b:
        return True
    if _ORDINAL_RE.sub(r"\1", a) == _ORDINAL_RE.sub(r"\1", b):
        return True
    if not (a.isalpha() and b.isalpha()) or a[0] != b[0]:
        return False
    short, long = sorted((a, b), key=len)
    if len(short) >= 4 and long.startswith(short):
        return True
    return edits and len(short) >= 4 and Levenshtein.distance(a, b) <= 1


def _one_word_respelled(a: List[str], b: List[str]) -> bool:
    """Two names of two or more words that differ only in one word's spelling."""
    if len(a) != len(b) or len(a) < 2:
        return False
    diffs = [(x, y) for x, y in zip(a, b) if x != y]
    return len(diffs) == 1 and _same_word_spelled_differently(*diffs[0], edits=False)


def _middle_words_dropped(a: List[str], b: List[str]) -> bool:
    """Same first and last word, the shorter name's words all in the longer one
    in order: "Guantánamo Navy base" / "Guantánamo Bay Navy Base"."""
    short, long = sorted((a, b), key=len)
    if len(short) < 2 or len(short) == len(long):
        return False
    if short[0] != long[0] or short[-1] != long[-1]:
        return False
    rest = iter(long[1:-1])
    return all(word in rest for word in short[1:-1])


def _same_group(
    a: str, b: str, groups: Optional[List[List[str]]], key: Callable[[str], str]
) -> bool:
    ka, kb = key(a), key(b)
    return any(
        ka in keys and kb in keys
        for keys in ({key(n) for n in group} for group in groups or [])
    )


# One spelling for words that English-language papers transliterate several
# ways ("Khalid Sheik Mohammed" / "Khalid Shaikh Mohammed").
_TRANSLITERATIONS = {
    **dict.fromkeys(("sheik", "shaikh", "shaykh", "sheikh"), "sheikh"),
    **dict.fromkeys(
        ("mohammed", "mohammad", "mohamed", "muhammed", "muhammad"), "muhammad"
    ),
    **dict.fromkeys(("qaida", "qaeda"), "qaida"),
}
# Arabic name particles: never a name on their own ("al" in "Abd al Hadi al Iraqi").
_NAME_PARTICLES: FrozenSet[str] = frozenset(
    {"al", "el", "abd", "abu", "bin", "ibn", "bint", "ben"}
)


def _match_words(name: str) -> List[str]:
    """Match-normalized words with one spelling per transliterated word."""
    return [_TRANSLITERATIONS.get(t, t) for t in normalize_for_match(name).split()]


def person_name_tokens(name: str) -> List[str]:
    """Name words without titles or single-letter initials ("Donald J. Trump")."""
    tokens = [t for t in _match_words(name) if len(t) > 1]
    while len(tokens) > 1 and tokens[0] in _PERSON_TITLES:
        tokens = tokens[1:]
    return tokens


def _people_variants(name_a: str, name_b: str) -> bool:
    """Same first and last name (one of them may be respelled: "Clay"/"Clayton",
    "Shelley"/"Shelly"), or a bare surname that ends the full name."""
    a, b = person_name_tokens(name_a), person_name_tokens(name_b)
    if not a or not b:
        return False
    if len(a) >= 2 and len(b) >= 2:
        # "Khalid Sheik Mohammed" / "Khalid Shaikh Mohammed"
        first, last = a[0] == b[0], a[-1] == b[-1]
        return (
            (first and last)
            or _ends_full_name(a, b)
            or (last and _same_word_spelled_differently(a[0], b[0], edits=True))
            or (first and _same_word_spelled_differently(a[-1], b[-1], edits=True))
        )
    short, long = (a, b) if len(a) < len(b) else (b, a)
    # "Trump" / "Donald Trump"; a lone first name doesn't count here (see
    # ``lone_name_in_full_name``)
    return len(short) == 1 and len(long) >= 2 and short[0] == long[-1]


def _ends_full_name(a: List[str], b: List[str]) -> bool:
    """A name of two or more real words that ends a longer one:
    "Hadi al Iraqi" / "Abd al Hadi al Iraqi"."""
    short, long = sorted((a, b), key=len)
    real = [t for t in short if t not in _NAME_PARTICLES]
    return len(real) >= 2 and len(short) < len(long) and long[-len(short) :] == short


def lone_name_in_full_name(name_a: str, name_b: str) -> bool:
    """A one-word name that is a first or middle name of the other name:
    "Lara" / "Lara Tur", "Hadi" / "Abd al Hadi al Iraqi".

    Weaker than ``names_are_variants``: "Daniel" is many people, so the merge
    only sends such a pair to the match check when exactly one entity in the
    table fits (``name_variant_candidate``).
    """
    a, b = person_name_tokens(name_a), person_name_tokens(name_b)
    short, long = (a, b) if len(a) < len(b) else (b, a)
    if len(short) != 1 or len(long) < 2:
        return False
    word = short[0]
    return len(word) >= 3 and word not in _NAME_PARTICLES and word in long[:-1]


# Head words of a body named after a person: "Bush administration".
_PERSON_LED_HEADS: FrozenSet[str] = frozenset({"administration", "government"})


def _person_led_variants(a: List[str], b: List[str]) -> bool:
    """ "Bush administration" / "George W. Bush administration": the same head
    word after two spellings of one person's name."""
    if len(a) < 2 or len(b) < 2 or a[-1] != b[-1] or a[-1] not in _PERSON_LED_HEADS:
        return False
    return _people_variants(" ".join(a[:-1]), " ".join(b[:-1]))


def _words_inserted(a: List[str], b: List[str], *, max_extra: int) -> bool:
    """The longer list is the shorter one with one run of at most ``max_extra``
    words added before its last word: "Waterboarding of KSM" / "CIA
    waterboarding of KSM". Words added in two places ("Easing of lockdown
    conditions in Camp 6" / "Lockdown of Camp 6") don't count."""
    short, long = sorted((a, b), key=len)
    extra = len(long) - len(short)
    if len(short) < 3 or not 0 < extra <= max_extra:
        return False
    return any(
        long[:i] == short[:i] and long[i + extra :] == short[i:]
        for i in range(len(short))
    )


_EVENT_STOPWORDS: FrozenSet[str] = DEFAULT_ACRONYM_STOPWORDS | {"as", "from", "with"}


def _event_words(name: str) -> List[str]:
    return [t for t in _match_words(name) if t not in _EVENT_STOPWORDS]


def names_are_variants(
    name_a: str,
    name_b: str,
    *,
    entity_type: str,
    equivalence_groups: Optional[List[List[str]]] = None,
) -> bool:
    """Whether two names could be spellings of one entity, worth a match check.

    Looser than ``names_likely_same``: "Guantánamo Bay" / "Guantanamo Bay",
    "Broward" / "Broward County", "Trump" / "Donald Trump". It only decides
    which pairs the match check sees; the match check decides the merge.
    """
    if normalize_for_match(name_a) == normalize_for_match(name_b):
        return True
    if _same_group(name_a, name_b, equivalence_groups, normalize_for_match):
        return True
    if entity_type == "people":
        return _people_variants(name_a, name_b)
    if entity_type in ("locations", "organizations"):
        core_a, core_b = _place_core(name_a), _place_core(name_b)
        if not core_a or not core_b:
            return False
        if core_a == core_b or _same_group(
            core_a, core_b, equivalence_groups, _place_core
        ):
            return True
        a, b = core_a.split(), core_b.split()
        return (
            _one_word_respelled(a, b)
            or _middle_words_dropped(a, b)
            or _person_led_variants(a, b)
        )
    if entity_type == "events":
        # "Firing of Convening Authority Harvey Rishikof" /
        # "Firing of Harvey Rishikof as convening authority"
        a, b = _event_words(name_a), _event_words(name_b)
        return (
            (len(a) >= 2 and sorted(a) == sorted(b))
            or _one_word_respelled(a, b)
            or _words_inserted(a, b, max_extra=2)
        )
    return False
