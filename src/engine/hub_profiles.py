"""Freeze the profiles of low-value hubs (countries, regions): link, don't write.

A hub such as "United States" is named in passing by most articles. In the
7 Nov 150-article pilot it was linked to 83 articles and each one cost a
profile-update call; the profile reached 47,945 characters, 41,658 of them
under one heading, a history of US detention policy filed under a country.

Storing one copied sentence per article instead was tried and dropped: over
all 1,620 articles only about 5 of 30 sampled sentences said anything about
the country itself ("the U.S. Navy base in southeast Cuba", "off Aden,
Yemen"), and the United States list would have reached ~370,000 characters.

So for an entity whose type is listed under ``profiles.link_only`` in the
domain config, a later article makes no model call and changes no text. The
article is still linked to the entity (``articles``), counted in the
profile's ``sources`` (the merger treats a profile without the article as a
failed update) and recorded in ``profile["linked_only"]``, so audits can tell
it apart from an article the updater forgot to cite. The profile keeps
whatever it had: a stub's copied sentences, or an older written profile.
``HINBOX_HUB_LINK_ONLY=off`` turns this off.
"""

import os
from typing import Any, Dict, List, Optional

from src.constants import ENTITY_SINGULAR

_PLURAL = {singular: plural for plural, singular in ENTITY_SINGULAR.items()}


def link_only_enabled() -> bool:
    return os.getenv("HINBOX_HUB_LINK_ONLY", "on").strip().lower() != "off"


def link_only_types(config: Dict[str, Any], entity_type: str) -> List[str]:
    """The kinds of ``entity_type`` (plural or singular) that are only linked."""
    table = _PLURAL.get(entity_type, entity_type)
    listed = ((config.get("profiles") or {}).get("link_only") or {}).get(table)
    return [str(k) for k in listed or []]


def is_hub(domain: str, entity_type: str, kind: Optional[str]) -> bool:
    """True when entities of this type and kind are linked, not updated."""
    if not kind or not link_only_enabled():
        return False
    from src.config_loader import get_domain_config

    # Older tables hold some types as enum reprs ("LocationType.COUNTRY")
    kind = str(kind).rsplit(".", 1)[-1].lower()
    return kind in link_only_types(get_domain_config(domain).load_config(), entity_type)


def link_article(profile: Dict[str, Any], article_id: str) -> Dict[str, Any]:
    """A copy of ``profile`` with the article in ``sources`` and
    ``linked_only``; the text is not touched."""
    sources = list(profile.get("sources") or [])
    linked = list(profile.get("linked_only") or [])
    return {
        **profile,
        "sources": sources + ([article_id] if article_id not in sources else []),
        "linked_only": linked + ([article_id] if article_id not in linked else []),
    }
