"""The article words behind a grounding verdict, kept only when they really
are in the article.

The grounding judge writes a short reasoning for each claim, and its prompt
asks it to quote the passage it compared the claim with: 'The article says
"Qadir was flown to the base in 2006", not 2007.' With
``HINBOX_GROUNDING_PASSAGE=on`` it also copies that passage into a field of
its own. Either way the model may misquote, so nothing it wrote is stored as
it wrote it. Each candidate is looked up in the article text the judge saw;
what is stored is the article's own characters at the place it was found.
A quote that isn't in the article (a reworded sentence, a quote of the claim,
an invented line) is dropped, and the reader then shows the whole article
with nothing highlighted, as before.

The lookup forgives only typing differences: letter case, runs of spaces and
line breaks, curly against straight quote marks, and the three kinds of
dash. Every word, number and punctuation mark in between must match.
"""

import re
import threading
from collections import Counter
from functools import lru_cache
from typing import Iterable, Iterator, List, Optional, Tuple

# A quote shorter than this, or with fewer words, can match in the wrong
# place ("the base" appears in most Guantánamo articles), so it isn't kept
MIN_PASSAGE_CHARS = 20
MIN_PASSAGE_WORDS = 4
# Passages kept per verdict: enough for "X ... and later Y" reasonings
MAX_PASSAGES = 3

_FOLD = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "„": '"',
        "«": '"',
        "»": '"',
        "‘": "'",
        "’": "'",
        "‚": "'",
        "′": "'",
        "–": "-",
        "—": "-",
        "−": "-",
        " ": " ",
    }
)
_SPACES = re.compile(r"\s+")
# Characters trimmed off both ends of a quote before it is looked up: the
# model often keeps the sentence's full stop or a stray quote mark
_EDGE = " \t\n.,;:!?\"'()[]"
_ELLIPSIS = re.compile(r"\s*(?:\[\s*(?:\.\s*){3}\]|\.\s*\.\s*\.|…)\s*")
# A quoted run in a reasoning: double quotes of either kind, or single quotes
# that don't touch a letter on the outside (so "Qadir's" isn't an opening)
# An apostrophe between two letters ("Hamdan's", "don't") is part of a
# single-quoted run, not its end: Qwen often quotes whole sentences that way
_QUOTED = re.compile(
    r"\"([^\"\n]+?)\"|“([^”\n]+?)”"
    r"|(?<![\w])'((?:[^'\n]|(?<=\w)'(?=\w))+?)'(?![\w])"
    r"|(?<![\w])‘((?:[^’\n]|(?<=\w)’(?=\w))+?)’(?![\w])"
)

# How candidates fared, for the bench: model_kept, model_rejected,
# reasoning_kept, reasoning_rejected
_stats: Counter = Counter()
_stats_lock = threading.Lock()


def passage_stats(reset: bool = False) -> Counter:
    """Counts of candidate passages kept and dropped since the last reset."""
    with _stats_lock:
        out = Counter(_stats)
        if reset:
            _stats.clear()
    return out


def _count(key: str, n: int = 1) -> None:
    if n:
        with _stats_lock:
            _stats[key] += n


@lru_cache(maxsize=64)
def _folded(text: str) -> Tuple[str, List[int]]:
    """``text`` lowercased, typographic marks made plain and each run of
    spaces turned into one space, with the position in ``text`` of every
    character kept, so a match can be mapped back."""
    chars: List[str] = []
    where: List[int] = []
    in_space = False
    for i, ch in enumerate(text.translate(_FOLD)):
        if ch.isspace():
            if not in_space and chars:
                chars.append(" ")
                where.append(i)
            in_space = True
            continue
        in_space = False
        # Lowercasing can give two characters ("İ" -> "i" + a combining
        # dot): each gets this position, so later matches still map back
        for low in ch.lower():
            chars.append(low)
            where.append(i)
    return "".join(chars), where


def _clean(quote: str) -> str:
    return _SPACES.sub(" ", quote.translate(_FOLD)).strip(_EDGE).lower()


def long_enough(quote: str) -> bool:
    q = _clean(quote)
    return len(q) >= MIN_PASSAGE_CHARS and len(q.split(" ")) >= MIN_PASSAGE_WORDS


def _matches(text: str, quote: str) -> Iterator[Tuple[int, int]]:
    """Every place in ``text`` where ``quote`` appears, typing differences
    aside (see the module notes), as start and end in ``text``. A match must
    start and end on word edges, so "ran to" doesn't match inside "Iran
    today"."""
    q = _clean(quote)
    if not q:
        return
    folded, where = _folded(text)
    start = 0
    while True:
        i = folded.find(q, start)
        if i < 0:
            return
        j = i + len(q)
        left_ok = i == 0 or not (folded[i - 1].isalnum() and q[0].isalnum())
        right_ok = j == len(folded) or not (folded[j].isalnum() and q[-1].isalnum())
        if left_ok and right_ok:
            yield where[i], where[j - 1] + 1
        start = i + 1


def find_passage(text: str, quote: str) -> Optional[Tuple[int, int]]:
    """Start and end in ``text`` of the first place ``quote`` appears, or
    None."""
    return next(_matches(text, quote), None)


def find_once(text: str, quote: str) -> Optional[Tuple[int, int]]:
    """Where ``quote`` appears in ``text``, only if it appears exactly once.
    A phrase the article uses twice ("the maximum-security Camp 7") can't
    say which sentence the judge read, so it is not kept."""
    found = _matches(text, quote)
    first = next(found, None)
    return first if first is not None and next(found, None) is None else None


def quotes_in(reasoning: Optional[str]) -> List[str]:
    """The quoted runs in a reasoning, each cut at any ellipsis, in order."""
    out: List[str] = []
    for m in _QUOTED.finditer(reasoning or ""):
        run = next(g for g in m.groups() if g is not None)
        out.extend(p for p in _ELLIPSIS.split(run) if p.strip())
    return out


def verified_passages(article: str, candidates: Iterable[str]) -> List[str]:
    """The candidates found in ``article`` exactly once, each as the
    article's own text at that place, long enough to be told apart, without
    repeats or a passage lying inside another kept one, at most
    ``MAX_PASSAGES``. A candidate the article has in two places is dropped,
    since the reader could not tell which one the judge meant."""
    spans: List[Tuple[int, int]] = []
    for c in candidates:
        if not long_enough(c):
            continue
        span = find_once(article, c)
        if span is None:
            continue
        if any(s <= span[0] and span[1] <= e for s, e in spans):
            continue
        spans = [(s, e) for s, e in spans if not (span[0] <= s and e <= span[1])]
        spans.append(span)
    return [article[s:e] for s, e in spans[:MAX_PASSAGES]]


def passages_for(
    article: str, reasoning: Optional[str], model_passage: Optional[str] = None
) -> Optional[List[str]]:
    """The verified passages behind one verdict: the judge's own copied
    passage first (split at any ellipsis), then the quotes in its reasoning.
    None when nothing it quoted is in ``article`` (the text the judge saw)."""
    copied = [p for p in _ELLIPSIS.split(model_passage or "") if p.strip()]
    quoted = quotes_in(reasoning)
    for name, items in (("model", copied), ("reasoning", quoted)):
        tried = [p for p in items if long_enough(p)]
        places = [sum(1 for _ in _matches(article, p)) for p in tried]
        _count(f"{name}_kept", sum(1 for n in places if n == 1))
        _count(f"{name}_rejected", sum(1 for n in places if n == 0))
        _count(f"{name}_ambiguous", sum(1 for n in places if n > 1))
    return verified_passages(article, copied + quoted) or None
