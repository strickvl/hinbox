"""Stub profiles: no model call until a second article mentions the entity.

On a 30-article local run most entities were mentioned in one article only,
yet each got a model-written profile when first seen (events alone: 163
profile calls, ~24 minutes). A stub is built in code instead: the article's
own sentences that name the entity (an event's extracted description first),
each cited to the article. It costs no model call, and every claim in it is
the article's own words.

The real profile is written when a second article merges into the entity:
``update_or_fill()`` creates the profile from the stub's article (looked up by
ID in ``article_texts``) exactly as the create path would have, then updates
it with the new article. That is the same two calls the entity would have had
without stubs, so a profile built from two articles comes out the same; only
entities that stay at one article are cheaper. If the first article's text
can't be found, the stub itself is updated (its sentences are kept and the
new article's facts added).

``HINBOX_PROFILE_STUBS=off`` writes a full profile for every new entity again.
Stubs are left out of the version history; the first real profile is
version 1.
"""

import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from src.constants import ENTITY_SINGULAR
from src.engine.profile_create import CITATION_RE, type_tag
from src.engine.profiles import VersionedProfile, create_profile, update_profile
from src.logging_config import log
from src.utils.quality_controls import citation_ids

MAX_STUB_SENTENCES = 3
MAX_SENTENCE_CHARS = 600
# Sentence ends: ., ! or ? (optionally closing a quote/bracket) before a space,
# and paragraph breaks
_SENTENCE_END = re.compile(
    r"(?:(?<=[.!?])|(?<=[.!?][\"')\]]))\s+(?=[A-Z0-9\"'(\[])|\n\s*\n"
)
_TITLES = {"mr", "mrs", "ms", "dr", "gen", "lt", "col", "sgt", "capt", "maj", "adm", "cmdr"}  # fmt: skip


def stubs_enabled() -> bool:
    """True unless ``HINBOX_PROFILE_STUBS`` is off/0/false/no."""
    value = os.getenv("HINBOX_PROFILE_STUBS", "on").strip().lower()
    return value not in ("off", "0", "false", "no")


def is_stub(profile: Optional[Dict[str, Any]]) -> bool:
    return bool((profile or {}).get("stub"))


_ABBREVIATIONS = _TITLES | {
    "sen", "rep", "gov", "st", "jr", "sr", "no", "vs", "u.s", "inc", "corp",
    "jan", "feb", "aug", "sept", "oct", "nov", "dec",
}  # fmt: skip


def _ends_with_abbreviation(piece: str) -> bool:
    """'Col.', 'U.S.' or an initial ('George W.') ends the piece, not a sentence."""
    if not piece.endswith("."):
        return False
    last = piece.split()[-1].rstrip(".").lower() if piece.split() else ""
    return last in _ABBREVIATIONS or (len(last) == 1 and last.isalpha())


def split_sentences(text: str) -> List[str]:
    """Split article text into sentences (good enough for news prose)."""
    sentences: List[str] = []
    for part in _SENTENCE_END.split(text or ""):
        part = " ".join((part or "").split())
        if not part:
            continue
        if sentences and _ends_with_abbreviation(sentences[-1]):
            sentences[-1] += " " + part
        else:
            sentences.append(part)
    return sentences


def _surname(entity_type: str, name: str) -> Optional[str]:
    """A person's last name ("Mohammed"), or None for short names and non-people."""
    if entity_type != "people":
        return None
    words = [w for w in name.split() if w.lower().strip(".") not in _TITLES]
    return words[-1] if len(words) > 1 and len(words[-1]) > 3 else None


def _pattern(needle: str) -> "re.Pattern[str]":
    return re.compile(rf"(?<!\w){re.escape(needle)}(?!\w)", re.I)


def _words(name: str) -> List[str]:
    return re.findall(r"[\w’']+", name.lower())


def _broader_than(alias: str, name: str) -> bool:
    """True when every word of ``alias`` is in ``name`` and ``name`` has more.

    Such an alias names something bigger: "Miami" for "Miami Shores", "AECOM"
    for "AECOM Tishman", "U.S. District Court" for one district's court. Its
    sentences are about that bigger thing, so they don't go in the stub.
    """
    alias_words, name_words = set(_words(alias)), set(_words(name))
    return bool(alias_words) and alias_words < name_words


# Capitalized words that can stand right before a surname without being
# another first name: ranks, offices and sentence openers.
_NOT_FIRST_NAMES = _ABBREVIATIONS | {
    "judge", "president", "secretary", "senator", "justice", "attorney",
    "prosecutor", "chief", "director", "ambassador", "professor", "prof",
    "marine", "army", "navy", "air", "general", "colonel", "captain",
    "commander", "admiral", "lieutenant", "sergeant", "major", "private",
    "officer", "agent", "the", "a", "an", "and", "but", "when", "while",
    "after", "before", "then", "said", "says", "by", "of", "for", "to", "as",
}  # fmt: skip


def _surname_shared(article_text: str, name: str, surname: str) -> bool:
    """True when the article also names someone else with ``surname``.

    "Marine Col. Keith Parrella" in an article about Jennifer Parrella: the
    word before the surname is a first name that isn't one of hers, so
    sentences with only "Parrella" may be about him.
    """
    own = set(_words(name))
    # A first name has no full stop after it, and sits on the surname's line:
    # "Kenya.\n\nWiltsie" is two sentences, not a "Kenya Wiltsie"
    before = re.compile(
        rf"\b([A-Z][\w’'-]*)[ \t]+(?:[A-Z]\.[ \t]+)?{re.escape(surname)}\b"
    )
    for match in before.finditer(article_text or ""):
        word = match.group(1).lower()
        if word not in own and word not in _NOT_FIRST_NAMES:
            return True
    return False


def mention_sentences(
    article_text: str,
    entity_type: str,
    names: List[str],
    limit: int = MAX_STUB_SENTENCES,
) -> List[str]:
    """Up to ``limit`` article sentences naming the entity, in article order.

    Sentences with the full name (or an alias) come first; sentences with
    only a person's surname fill any remaining places. An alias made only of
    words from the name ("Miami" for "Miami Shores") is not searched for, and
    the surname is not used when the article names another person with it.
    """
    names = list(dict.fromkeys(n.strip() for n in names if n and n.strip()))
    if not names:
        return []
    names = names[:1] + [n for n in names[1:] if not _broader_than(n, names[0])]
    full = [_pattern(n) for n in names]
    surname = _surname(entity_type, names[0])
    if surname and _surname_shared(article_text, names[0], surname):
        surname = None
    short = [_pattern(surname)] if surname else []
    sentences = [
        s for s in split_sentences(article_text) if len(s) <= MAX_SENTENCE_CHARS
    ]
    strong = [i for i, s in enumerate(sentences) if any(p.search(s) for p in full)]
    weak = [
        i
        for i, s in enumerate(sentences)
        if i not in strong and any(p.search(s) for p in short)
    ]
    picked = sorted(strong[:limit] + weak[: max(0, limit - len(strong))])
    return [sentences[i] for i in picked]


def _display_name(entity_key: Union[str, Tuple]) -> str:
    return str(entity_key[0] if isinstance(entity_key, tuple) else entity_key)


def _alt_names(entity_dict: Dict[str, Any], alternative_field: str) -> List[str]:
    names = []
    for alt in entity_dict.get(alternative_field) or []:
        if isinstance(alt, dict):
            names.append(str(alt.get("name") or alt.get("title") or ""))
        elif isinstance(alt, str):
            names.append(alt)
    return names + [a for a in entity_dict.get("aliases") or [] if isinstance(a, str)]


def build_stub_profile(
    entity_type: str,
    entity_key: Union[str, Tuple],
    entity_dict: Dict[str, Any],
    article_text: str,
    article_id: str,
    alternative_field: str = "alternative_names",
) -> Dict[str, Any]:
    """A cited profile made of the article's own sentences; no model call.

    ``entity_type`` is the plural table name ("people", "events"…).
    """
    marker = f"^[{article_id}]"
    name = _display_name(entity_key)
    blocks: List[str] = []
    description = str(entity_dict.get("description") or "").strip()
    if entity_type == "events" and description:
        blocks.append(f"### Summary\n\n{description}{marker}")
    sentences = mention_sentences(
        article_text, entity_type, [name] + _alt_names(entity_dict, alternative_field)
    )
    if sentences:
        blocks.append(
            "### Mentions\n\n" + "\n\n".join(f"{s}{marker}" for s in sentences)
        )
    if not blocks:
        blocks.append(f"{name} is mentioned in this article.{marker}")
    tags = [t for t in entity_dict.get("tags") or [] if isinstance(t, str) and t]
    tag = type_tag(entity_type)
    return {
        "text": "\n\n".join(blocks),
        "tags": list(dict.fromkeys(tags + [tag])),
        "confidence": 0.5,
        "sources": [article_id],
        "stub": True,
        "qc": {"passed": True, "flags": ["stub"]},
    }


def first_profile(
    entity_type: str,
    entity_key: Union[str, Tuple],
    entity_dict: Dict[str, Any],
    article_text: str,
    article_id: str,
    model_type: str,
    domain: str,
    *,
    alternative_field: str,
    model: Optional[str] = None,
    create: Callable[..., Tuple[Dict, VersionedProfile, list]] = create_profile,
) -> Tuple[Dict, VersionedProfile, list]:
    """A new entity's profile: a stub, or with stubs off ``create(...)``.

    The merger passes its own ``create_profile`` as ``create``.
    """
    if stubs_enabled():
        stub = build_stub_profile(
            entity_type,
            entity_key,
            entity_dict,
            article_text,
            article_id,
            alternative_field,
        )
        return stub, VersionedProfile(), []
    return create(
        ENTITY_SINGULAR[entity_type],
        _display_name(entity_key),
        article_text,
        article_id,
        model_type,
        domain,
        model=model,
    )


def stub_sources(profile: Dict[str, Any]) -> List[str]:
    """The stub's article IDs, in the order they were cited."""
    ids = list(profile.get("sources") or [])
    cited = [
        i
        for m in CITATION_RE.findall(profile.get("text") or "")
        for i in citation_ids(m)
    ]
    return list(dict.fromkeys(ids + cited))


def fill_stub(
    entity_type: str,
    entity_name: str,
    stub: Dict[str, Any],
    article_texts: Dict[str, str],
    model_type: str,
    domain: str,
    model: Optional[str] = None,
    article_dates: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict, VersionedProfile, list]:
    """Write the real profile from the stub's article(s).

    Creates it from the first article, then updates it with any later one.
    Returns the stub unchanged (with an empty history) when the first
    article's text isn't in ``article_texts``.
    """
    sources = stub_sources(stub)
    first = sources[0] if sources else None
    if not first or not article_texts.get(first):
        log(
            f"Stub for '{entity_name}': article {first} not found, "
            "updating the stub itself",
            level="warning",
        )
        return stub, VersionedProfile(), []
    profile, versions, history = create_profile(
        entity_type,
        entity_name,
        article_texts[first],
        first,
        model_type,
        domain,
        model=model,
    )
    for later in sources[1:]:
        if article_texts.get(later):
            profile, versions, more = update_profile(
                entity_type,
                entity_name,
                profile,
                versions,
                article_texts[later],
                later,
                model_type,
                domain,
                model=model,
                article_dates=article_dates,
            )
            history = list(history) + list(more)
    return profile, versions, history


def update_or_fill(
    entity_type: str,
    entity_name: str,
    profile: Dict[str, Any],
    versioned_profile: VersionedProfile,
    article_text: str,
    article_id: str,
    model_type: str,
    domain: str,
    *,
    article_texts: Optional[Dict[str, str]] = None,
    model: Optional[str] = None,
    article_dates: Optional[Dict[str, Any]] = None,
    entity: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict, VersionedProfile, list]:
    """``update_profile()``, writing the real profile first if this is a stub.

    ``entity_type`` is singular ("person"), as ``update_profile`` takes it.
    When ``entity`` (the stored entity dict) is of a kind listed under
    ``profiles.link_only`` in the domain config (countries, regions), no
    model call is made and the text is left as it is: the article is only
    linked (``hub_profiles.py``), stub or not.
    """
    from src.engine.hub_profiles import is_hub, link_article

    if entity is not None and is_hub(domain, entity_type, entity.get("type")):
        log(f"Linked article to hub '{entity_name}' without a profile update")
        return link_article(profile, article_id), versioned_profile, []
    history: list = []
    if is_stub(profile):
        log(f"Writing full profile for '{entity_name}' (second article)", level="info")
        profile, versioned_profile, history = fill_stub(
            entity_type,
            entity_name,
            profile,
            article_texts or {},
            model_type,
            domain,
            model=model,
            article_dates=article_dates,
        )
    updated, versioned_profile, more = update_profile(
        entity_type,
        entity_name,
        profile,
        versioned_profile,
        article_text,
        article_id,
        model_type,
        domain,
        model=model,
        article_dates=article_dates,
    )
    # A failed update hands back the profile it was given; the caller sees
    # that this article isn't in its sources and keeps the stub
    updated = {k: v for k, v in updated.items() if k != "stub"}
    return updated, versioned_profile, list(history) + list(more)
