"""Training claims whose right verdict is known without any judge.

Take a sentence copied word for word from an article ("Qadir was flown to
the base in 2006."): the article plainly says it, so it is "supported". Then
change one detail in code: "Qadir was flown to the base in 2008." The
article now says something different about the same fact, so the claim is
"partial" (the main fact is there, one detail is wrong), exactly as the
judge's prompt defines it. Turn "was flown" into "was not flown" and it is
"not_supported". No model and no person decides these labels, so a tuned
checker can learn from them to be stricter than the teachers whose verdicts
make up the rest of its training data.

Kinds of change (``KINDS``):

- ``year``: a year moved by 1-3 ("2006" -> "2008").
- ``number``: another number ("14 men" -> "17 men").
- ``name``: a two-word name swapped for another two-word name from the
  same article ("said Col. Ann Lee" -> "said Col. Raza Khan"); never the
  entity's own name.
- ``negation``: "not" added after the first was/is/had/did..., or removed.
- ``hedge``: "allegedly", "reportedly", "suspected" and friends dropped,
  turning a report into a flat statement.

A change is kept only if the new claim is not itself in the article (case
and spacing aside), and for years and numbers only if the new value appears
nowhere in the article, so a planted error can't be accidentally true.
"""

import random
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.citecheck.items import article_split, item_id
from src.engine.profile_stubs import split_sentences
from src.utils.grounding import GROUNDING_MAX_ARTICLE_CHARS

KINDS = ("year", "number", "name", "negation", "hedge")
LEVEL = {
    "copy": "supported",
    "year": "partial",
    "number": "partial",
    "name": "partial",
    "hedge": "partial",
    "negation": "not_supported",
}
MIN_CHARS, MAX_CHARS = 60, 320
QUOTE_WORDS = 25

_YEAR = re.compile(r"\b(19[5-9]\d|20[0-3]\d)\b")
_NUMBER = re.compile(r"(?<![\w.,])(\d{1,3}(?:,\d{3})*|\d+)(?![\w.,]\d)")
_NAME = re.compile(r"\b([A-Z][a-z’'\-]+(?:\s(?:al-|bin\s|el-)?[A-Z][a-z’'\-]+)+)\b")
_AUX = re.compile(
    r"\b(is|was|were|are|has|had|have|will|would|could|can|did|does|do)\b(?!\s+not\b)"
)
_NEGATED = re.compile(
    r"\b(is|was|were|are|has|had|have|will|would|could|can|did|does|do)\s+not\b"
)
_NT = re.compile(
    r"\b(isn't|wasn't|weren't|aren't|hasn't|hadn't|haven't|didn't|doesn't|don't)\b"
)
_HEDGE = re.compile(
    r"\b(allegedly|reportedly|purportedly|apparently|possibly|suspected|alleged)\s+",
    re.I,
)
_SPACE = re.compile(r"\s+")
_SENTENCE_START_WORDS = {"The", "A", "An", "In", "On", "At", "He", "She", "It", "They"}


def _squash(text: str) -> str:
    return _SPACE.sub(" ", text).strip().lower()


def _names_entity(sentence: str, entity_name: str) -> bool:
    words = set(re.findall(r"\w+", sentence.lower()))
    return any(
        w in words for w in re.findall(r"\w+", entity_name.lower()) if len(w) >= 4
    )


def _quote(sentence: str, near: int = 0) -> str:
    """The sentence, or a run of ``QUOTE_WORDS`` of its words around
    character ``near``, word for word, so the quote is found in the article."""
    words = sentence.split()
    if len(words) <= QUOTE_WORDS:
        return sentence
    at = len(sentence[:near].split())
    start = max(0, min(at - QUOTE_WORDS // 2, len(words) - QUOTE_WORDS))
    return " ".join(words[start : start + QUOTE_WORDS])


def change_year(s: str, article: str, rng: random.Random) -> Optional[Tuple[str, str]]:
    found = list(_YEAR.finditer(s))
    if not found:
        return None
    m = rng.choice(found)
    old = int(m.group(1))
    for step in rng.sample([-3, -2, -1, 1, 2, 3], 6):
        new = str(old + step)
        if new not in article:
            return (
                s[: m.start()] + new + s[m.end() :],
                f"the claim gives {new} instead of {old}",
            )
    return None


def change_number(
    s: str, article: str, rng: random.Random
) -> Optional[Tuple[str, str]]:
    found = [m for m in _NUMBER.finditer(s) if not _YEAR.fullmatch(m.group(1))]
    if not found:
        return None
    m = rng.choice(found)
    old = int(m.group(1).replace(",", ""))
    candidates = [old + d for d in (1, 2, 3, 5, 10)] + [
        old - d for d in (1, 2, 3) if old - d > 0
    ]
    rng.shuffle(candidates)
    for new_value in candidates:
        new = f"{new_value:,}" if "," in m.group(1) else str(new_value)
        if not re.search(rf"(?<![\d,]){re.escape(new)}(?![\d,])", article):
            return (
                s[: m.start()] + new + s[m.end() :],
                f"the claim gives {new} instead of {m.group(1)}",
            )
    return None


def _article_names(article: str) -> List[str]:
    names = []
    for m in _NAME.finditer(article):
        name = m.group(1)
        if name.split()[0] in _SENTENCE_START_WORDS:
            continue
        if name not in names:
            names.append(name)
    return names


def change_name(
    s: str, article: str, rng: random.Random, entity_name: str
) -> Optional[Tuple[str, str]]:
    entity_words = {w.lower() for w in re.findall(r"\w+", entity_name) if len(w) >= 3}

    def is_entity(name: str) -> bool:
        return any(w.lower() in entity_words for w in re.findall(r"\w+", name))

    inside = [
        m
        for m in _NAME.finditer(s)
        if m.group(1).split()[0] not in _SENTENCE_START_WORDS
        and not is_entity(m.group(1))
    ]
    if not inside:
        return None
    m = rng.choice(inside)
    others = [
        n
        for n in _article_names(article)
        if n not in s and not is_entity(n) and len(n.split()) == len(m.group(1).split())
    ]
    if not others:
        return None
    new = rng.choice(others)
    return (
        s[: m.start()] + new + s[m.end() :],
        f"the claim names {new}, not {m.group(1)}",
    )


def change_negation(
    s: str, article: str, rng: random.Random
) -> Optional[Tuple[str, str]]:
    m = _NEGATED.search(s)
    if m:
        return s[: m.start()] + m.group(1) + s[m.end() :], "the claim says the opposite"
    if _NT.search(s):
        return None  # "didn't" -> "did" changes the verb form; leave it
    m = _AUX.search(s)
    if not m:
        return None
    return s[: m.end()] + " not" + s[m.end() :], "the claim says the opposite"


def drop_hedge(s: str, article: str, rng: random.Random) -> Optional[Tuple[str, str]]:
    m = _HEDGE.search(s)
    if not m:
        return None
    rest = s[m.end() :]
    if m.start() == 0 and rest:
        rest = rest[0].upper() + rest[1:]
    return s[: m.start()] + rest, f'the claim drops "{m.group(1).lower()}"'


def _change(kind: str, s: str, article: str, rng: random.Random, entity: str):
    if kind == "year":
        return change_year(s, article, rng)
    if kind == "number":
        return change_number(s, article, rng)
    if kind == "name":
        return change_name(s, article, rng, entity)
    if kind == "negation":
        return change_negation(s, article, rng)
    return drop_hedge(s, article, rng)


def _changed_at(original: str, changed: str) -> int:
    for i, (a, b) in enumerate(zip(original, changed)):
        if a != b:
            return i
    return min(len(original), len(changed))


def planted_claims(
    pairs: Iterable[Tuple[str, str]],
    texts: Dict[str, str],
    per_pair: int = 2,
    seed: int = 7,
    splits: Tuple[str, ...] = ("train", "tune"),
) -> List[Dict[str, Any]]:
    """Copied and changed claims for (entity name, article id) pairs whose
    article is in ``splits``: for each pair, up to ``per_pair`` copied
    sentences naming the entity, each kept as "supported" and changed once
    by a kind it allows (the rarer kinds tried first)."""
    rng = random.Random(seed)
    out: List[Dict[str, Any]] = []
    done = set()
    for entity_name, article_id in sorted(set(pairs)):
        split = article_split(article_id)
        article = texts.get(article_id) or ""
        if split not in splits or not article:
            continue
        flat = _squash(article)
        # Only sentences from the part of the article the judge is shown
        # (``build_examples()`` cuts the same way), so a "supported" copy is
        # never one the training prompt leaves out
        sentences = [
            s
            for s in split_sentences(article[:GROUNDING_MAX_ARTICLE_CHARS])
            if MIN_CHARS <= len(s) <= MAX_CHARS and _names_entity(s, entity_name)
        ]
        rng.shuffle(sentences)
        for sentence in sentences[:per_pair]:
            entity = entity_name
            base = {"entity_name": entity, "article_id": article_id, "split": split}
            key = (entity, article_id, sentence)
            if key in done:
                continue
            done.add(key)
            out.append(
                {
                    **base,
                    "item_id": item_id(f"planted/{entity}", article_id, sentence),
                    "claim": sentence,
                    "kind": "copy",
                    "level": LEVEL["copy"],
                    "reasoning": f'The article says "{_quote(sentence)}".',
                    "source": "planted",
                }
            )
            for kind in ("hedge", "name", "negation", "number", "year"):
                made = _change(kind, sentence, article, rng, entity)
                if not made:
                    continue
                changed, why = made
                if changed == sentence or _squash(changed) in flat:
                    continue
                quote = _quote(sentence, _changed_at(sentence, changed))
                out.append(
                    {
                        **base,
                        "item_id": item_id(f"planted/{entity}", article_id, changed),
                        "claim": changed,
                        "kind": kind,
                        "level": LEVEL[kind],
                        "reasoning": f'The article says "{quote}"; {why}.',
                        "source": "planted",
                    }
                )
                break
    return out
