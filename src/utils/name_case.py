"""Put an ALL CAPS name back in the case the article uses elsewhere.

Datelines print places in capitals ("GUANTANAMO BAY NAVY BASE, Cuba —"), and
models copy them. If the article also writes the name in normal case
("Guantánamo Bay Navy Base"), that spelling is used. Otherwise a location
name is title-cased ("Guantanamo Bay Navy Base"); other types are left as
they are, since "ACLU OF FLORIDA" would come out wrong. No LLM calls.
"""

import re
from typing import Any, Dict, List

from src.utils.name_variants import fold_accents, is_shouting_name


def _fold_keep_length(text: str) -> str:
    """Accent-folded text with one character per original character."""
    return "".join(fold_accents(c)[:1] or c for c in text)


def recase_from_article(name: str, article_text: str) -> str:
    """The article's normal-case spelling of an ALL CAPS ``name`` (else ``name``)."""
    if not is_shouting_name(name) or not article_text:
        return name
    pattern = re.compile(
        r"\b" + re.escape(_fold_keep_length(name)) + r"\b", re.IGNORECASE
    )
    for found in pattern.finditer(_fold_keep_length(article_text)):
        original = article_text[found.start() : found.end()]
        if not is_shouting_name(original):
            return original
    return name


_SMALL_WORDS = frozenset({"of", "the", "and", "de", "del", "al", "el", "on", "in"})


def _title_case_word(word: str, first: bool) -> str:
    if not first and word.lower() in _SMALL_WORDS:
        return word.lower()
    if "." in word or not re.search(r"[AEIOUY]", word):  # "U.S.", "USS", "JTF"
        return word
    return "-".join(p[:1] + p[1:].lower() for p in word.split("-"))


def title_case_place(name: str) -> str:
    """Title-case an ALL CAPS place: "USS COLE" becomes "USS Cole"."""
    if not is_shouting_name(name):
        return name
    return " ".join(_title_case_word(w, i == 0) for i, w in enumerate(name.split()))


def recase_entity_names(
    entities: Dict[str, List[Dict[str, Any]]], article_text: str
) -> List[Dict[str, str]]:
    """Recase ALL CAPS names in place; return [{"from", "to"}] for each change."""
    changed: List[Dict[str, str]] = []
    for entity_type, items in entities.items():
        for item in items:
            field = "name" if "name" in item else "title"
            old = item.get(field)
            if not isinstance(old, str):
                continue
            new = recase_from_article(old, article_text)
            if new == old and entity_type == "locations":
                new = title_case_place(old)
            if new != old:
                item[field] = new
                changed.append({"from": old, "to": new})
    return changed
