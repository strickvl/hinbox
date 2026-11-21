"""The checks a written page's sentences go through (``written_page.py``).

1. ``code_check()``: every number, month and capitalised word of a
   sentence must be in the notes it cites (with their article dates) or in
   the entity's name (``missing_details()``).
2. ``drop_repeats()``: a section sentence saying again what an earlier one
   said is dropped.
3. ``model_check()``: one call per 8 sentences judges each against its
   notes: supported, partial, wrong or not supported; the last two drop it.
4. ``summary_check()``: an opening sentence summing up the sections is
   checked against the sections instead of one note.
"""

import re
import unicodedata
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field

from src.engine.written_page import (
    CHECK_BATCH,
    CHECK_NOTE_CHARS,
    MAX_NOTES_SHOWN,
    Ask,
    Note,
    Written,
)


class SentenceCheck(BaseModel):
    model_config = ConfigDict(extra="ignore")
    sentence: int
    starts_with: str = Field(description="The sentence's first three words, copied")
    problem: str = Field(description="What the notes lack or say otherwise, or 'none'")
    verdict: str = Field(description="supported, partial, wrong or not_supported")


class SentenceChecks(BaseModel):
    model_config = ConfigDict(extra="ignore")
    checks: List[SentenceCheck]


CHECK = """You check sentences of an encyclopedia entry about "{name}" \
against the notes each one was written from. The notes are sentences taken \
from news articles.

For each numbered sentence:
1. Copy its first three words.
2. Say in a few words what its notes lack or say differently, or "none".
3. Give one verdict:
- supported: everything the sentence says is in its notes. A sentence may \
leave out details of its notes, or put them in other words; that is fine.
- partial: the facts are right, but a claim the notes attribute to someone \
is stated as fact, or a word goes further than the notes ("the first" for \
"one of the first", "always" for once).
- wrong: a name, date, number or place differs from the notes, a fact \
the notes give about one person or event is given to another, or facts \
from different dates are joined as if they happened at one time.
- not_supported: its notes do not say this.

A sentence may count or describe its own notes ("one account says ...; \
three later reports give ..."): that is supported when its notes match. \
Each note starts with its article's date in brackets. A sentence may use \
that date for when something was said, reported or true ("In May 2016, he \
said ...", "By 2016 ..."); that counts as in the notes. But that date put on \
an older report, document or event the note only looks back on ("In April \
2018, the Senate report said ..." from an April 2018 article) is wrong. Judge only by the \
notes shown with each sentence, not by what you know."""


_MONTHS = (
    "january february march april may june july august september october "
    "november december jan feb mar apr jun jul aug sep sept oct nov dec"
).split()
_NUMBER = re.compile(r"\d[\d,.]*\d|\d")
_CAP = re.compile(r"\b[A-Z][\w'’-]*")
# Words that can start a sentence or a clause without being a name
_COMMON = set(
    """a an the in on at by for from of to with after before during since
    until while when where its his her their he she they it this that these
    those there some several many both other others among according later
    earlier however although despite as under over also and but or no not
    one two three four five six seven eight nine ten first second third
    then thus today yet each all most more few such between about around
    within without into through because if though once until former
    detainee detainees prisoner prisoners""".split()
)


# Titles a writer puts before a name the notes give ("President Trump")
_TITLES = set(
    """president former judge general gen senator sen secretary colonel col
    captain capt lieutenant lt sergeant sgt admiral adm major maj governor
    commander rear vice chief prime minister sheikh sheik dr mr mrs ms""".split()
)

# "U.S." in the notes covers "United States" and "American" in a sentence
_US_WORDS = {"united", "states", "america", "american", "americans", "u", "s"}


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.replace("’", "'").lower()


def missing_details(sentence: str, sources: Iterable[str], name: str) -> List[str]:
    """Numbers, months and capitalised words of ``sentence`` that are in
    neither its notes nor the entity's name. A word counts as present when
    one of the notes' words shares its first five letters ("Pakistani" for
    "Pakistan")."""
    hay = _fold(" ".join(sources) + " " + name)
    digits = hay.replace(",", "")
    words = set(re.findall(r"[\w']+", hay))
    words |= {w[:-2] for w in words if w.endswith("'s")}
    stems = {w[:5] for w in words if len(w) >= 5}
    if re.search(r"\bu\.?s\b|\bunited states\b", hay):
        words |= _US_WORDS

    def present(low: str) -> bool:
        if len(low) < 2 or low in _COMMON or low in _MONTHS or low in words:
            return True
        if low.endswith("'s") and low[:-2] in words:
            return True
        if low in _TITLES:
            return True
        if "-" in low:  # "Bush-era": each part
            return all(present(part) for part in low.split("-") if part)
        return len(low) >= 5 and low[:5] in stems

    missing = []
    for num in _NUMBER.findall(sentence):
        if num.replace(",", "").rstrip(".") not in digits:
            missing.append(num)
    for m in _CAP.finditer(sentence):
        tok = m.group(0)
        # A first name before a surname the notes give ("Donald Trump" for
        # "Trump"); an invented full name ("Clive Stafford Smith") still fails
        nxt = re.match(r"\s+([A-Z][\w'’-]*)", sentence[m.end() :])
        if nxt and _fold(nxt.group(1)).strip("'-") in words and len(tok) > 2:
            continue
        # A sentence's first word is capitalised anyway ("Reports say ...",
        # "Regarding Walid ..."), so it is left to the model check
        if not sentence[: m.start()].strip(" \"'“‘("):
            continue
        if not present(_fold(tok).strip("'-")):
            missing.append(tok)
    # Who did it: "Pakistani forces captured", "CIA agents captured" need
    # that actor beside a force word in a note, not just the country or the
    # agency somewhere in it (21 Nov: al Sharabi's note names no captor; the
    # CIA page's note on al Hawsawi says Pakistani authorities arrested him)
    for m in _ACTOR.finditer(sentence):
        actor = _fold(m.group(1)).rstrip(".")
        actor = actor[:6] if len(actor) > 6 else actor  # "Pakistani", "Pakistan's"
        near = rf"\b{re.escape(actor)}\w*\W+(?:\w+\W+){{0,2}}(?:{_FORCES})\b"
        if not re.search(near, hay):
            missing.append(f"{m.group(1)} {m.group(2)}")
    return list(dict.fromkeys(missing))


_FORCES = (
    "forces|force|authorities|agents|officers|officials|police|troops|"
    "intelligence|operatives|soldiers|security"
)
_ACTOR = re.compile(
    rf"\b([A-Z][\w.]*(?:-[A-Z][\w.]*)?)\s+({_FORCES})\s+(?:\w+\s+){{0,2}}?"
    r"(?:captured|arrested|seized|detained|raided|apprehended|abducted|caught|took)\b"
)


def code_check(written: List[Written], notes: Sequence[Note], name: str) -> None:
    by_n = {n.n: n for n in notes}
    for w in written:
        cited = [by_n[i] for i in w.notes if i in by_n]
        if not cited:
            w.dropped = "cites no note"
            continue
        w.notes = [n.n for n in cited]
        # A note's article date may be named ("a 2012 report")
        gaps = missing_details(w.text, [f"{n.text} {n.date}" for n in cited], name)
        if gaps:
            w.dropped = "not in its notes: " + ", ".join(gaps[:5])
        elif _OLD_REPORT.match(w.text) and from_article_dates(w.text, cited):
            w.dropped = (
                "check: it puts the article's date on an older report; "
                "leave the report undated unless its notes date it"
            )


# "In April 2018, the Senate torture report said": an article's date put on
# a report it looks back on (GPT-6 Luna, CIA page, 21 Nov)
_OLD_REPORT = re.compile(
    r"(?:In|By|On)\s+(?:[A-Z][a-z]+\.?\s+)?(?:\d{1,2},\s+)?\d{4},\s+(?:the|a|an)\s+"
    r"(?:[\w'’-]+\s+){0,4}(?:report|study|review|investigation|memo|document|"
    r"survey|inquiry)s?\b"
)

# "a news account described", "another account says": sources narrated
# where nothing is disputed
NARRATES = re.compile(
    r"\b(?:a|an|another|one|a separate|a later|other)\s+(?:\d{4}\s+)?"
    r"(?:news\s+|later\s+|separate\s+|press\s+)?(?:account|report|article|source|"
    r"story)s?\b(?!\s+(?:by|from|of)\s+[A-Z])",
    re.I,
)


def _content(text: str) -> set:
    return {
        w for w in re.findall(r"[\w']+", _fold(text)) if len(w) > 2 and w not in _COMMON
    }


_DAY = re.compile(
    r"\b(" + "|".join(_MONTHS) + r")\.?\s+(\d{1,2}(?:\s*[-–]\s*\d{1,2})?)\b", re.I
)


def _numbers(text: str) -> set:
    """Numbers of a sentence, a day with its month as one ("sep 10-11", so
    "Sept. 11" is not taken as said by "Sept. 10-11")."""
    days = {
        f"{m.group(1)[:3].lower()} {m.group(2).replace(' ', '')}"
        for m in _DAY.finditer(text)
    }
    rest = _NUMBER.findall(_DAY.sub(" ", text))
    return days | set(rest)


def drop_repeats(written: List[Written], share: float = 0.8) -> None:
    """Drop a section sentence whose words are nearly all in one earlier
    section sentence (a one-sentence section saying again what the first
    section said; on 21 Nov Luna's Karachi page said one fact three times),
    or that cites only notes earlier sentences of its section cited and
    whose words are nearly all in those sentences (local Qwen told one
    arrest three times in three wordings)."""
    seen: List[set] = []
    section: Dict[str, Tuple[set, set]] = {}
    nums_said: set = set()
    for w in written:
        if w.dropped or not w.section:
            continue
        words = _content(w.text)
        nums = _numbers(w.text)
        said, cited = section.setdefault(w.section, (set(), set()))
        again = words and any(len(words & old) >= share * len(words) for old in seen)
        told = (
            words and set(w.notes) <= cited and len(words & said) >= 0.75 * len(words)
        )
        # A number or date not said yet makes it new (Qwen's Sept. 11, 2002
        # capture of bin Attash was dropped as a repeat of his April 2003 one)
        if (again or told) and nums <= nums_said:
            w.dropped = "repeats an earlier sentence"
            continue
        seen.append(words)
        said |= words
        cited |= set(w.notes)
        nums_said |= nums


# An opening sentence about the page instead of its subject
_ABOUT_PAGE = re.compile(
    r"\b(?:this|the|another|a later|the next) (?:entry|page|article|record|sections?)\b"
    r"|\bthese notes\b"
    r"|\bthe notes\b",
    re.IGNORECASE,
)


# An opening summary about the sources, not the subject: "Reports described
# arrests and intelligence raids in Karachi ...; other accounts placed a
# detainee there" (GPT-6 Luna, 21 Nov)
_ABOUT_REPORTS = re.compile(
    r"\b(?:reports|accounts|articles|sources|news accounts)\s+(?:also\s+)?"
    r"(?:describe|described|say|said|cover|covered|mention|mentioned|place|placed|"
    r"note|noted|discuss|discussed|detail|detailed)\b",
    re.IGNORECASE,
)


def summary_check(written: List[Written], notes: Sequence[Note], name: str) -> None:
    """An opening sentence summing up the sections is checked against the
    sections' kept sentences and their notes, not its own notes alone: a
    summary in general words is never in one note (on 21 Nov most were
    dropped that way, and the openings were left one sentence long)."""
    by_n = {n.n: n for n in notes}
    body = [w for w in written if w.section and not w.dropped]
    sources = [w.text for w in body] + [
        f"{by_n[n].text} {by_n[n].date}" for w in body for n in w.notes if n in by_n
    ]
    for w in written:
        if not w.summary or w.dropped:
            continue
        if _ABOUT_PAGE.search(w.text) or _ABOUT_REPORTS.search(w.text):
            w.dropped = "talks about the page"
            continue
        words = _content(w.text)
        first = next((o for o in written if not o.section and not o.summary), None)
        if first is not None and first is not w and _content(first.text) <= words:
            w.dropped = "repeats the first sentence"
            continue
        if any(
            len(words & _content(o.text)) >= 0.8 * len(words)
            for o in written
            if o is not w and not o.dropped and not o.summary
        ):
            w.dropped = "copies a sentence of the page"
            continue
        w.notes = [n for n in w.notes if n in by_n] or [
            n for b in body for n in b.notes[:1]
        ][:6]
        gaps = missing_details(w.text, sources, name)
        if gaps:
            w.dropped = "not in the sections: " + ", ".join(gaps[:5])
        else:
            w.check = "summary"


_MONTH_N = {m: i for i, m in enumerate(_MONTHS[:12], 1)}
_WHEN = re.compile(
    r"\b(?:(" + "|".join(_MONTHS[:12]) + r")\s+)?((?:19|20)\d\d)\b", re.I
)


def from_article_dates(sentence: str, shown: Sequence[Note]) -> List[str]:
    """The sentence's "May 2017" or "2017" that no note's text gives but a
    note's article date does: the writer was told it may date a statement
    by its article, and on 21 Nov the check dropped such sentences anyway."""
    text = _fold(" ".join(n.text for n in shown))
    out = []
    for m in _WHEN.finditer(sentence):
        month, year = (m.group(1) or "").lower(), m.group(2)
        if year in text and (not month or month in text):
            continue
        for n in shown:
            if n.date[:4] == year and (
                not month or n.date[5:7] == f"{_MONTH_N[month]:02d}"
            ):
                out.append(m.group(0))
                break
    return out


def check_messages(
    batch: Sequence[Written], notes: Sequence[Note], name: str
) -> List[Dict[str, str]]:
    by_n = {n.n: n for n in notes}
    parts = []
    for i, w in enumerate(batch, 1):
        shown = [by_n[n] for n in w.notes[:MAX_NOTES_SHOWN] if n in by_n]
        lines = "\n".join(
            f"  - [{n.date or 'no date'}] {n.text[:CHECK_NOTE_CHARS]}" for n in shown
        )
        dated = from_article_dates(w.text, shown)
        hint = (
            f"\nDates taken from its articles' dates (accept them): {', '.join(dated)}"
            if dated
            else ""
        )
        parts.append(f"Sentence {i}: {w.text}\nIts notes:\n{lines}{hint}")
    return [
        {"role": "system", "content": CHECK.format(name=name)},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def _first_words(text: str, n: int = 3) -> str:
    return " ".join(re.findall(r"[\w']+", _fold(text))[:n])


def model_check(
    written: List[Written], notes: Sequence[Note], name: str, ask: Ask,
    report: Dict[str, Any],
) -> None:  # fmt: skip
    """Drop sentences the model calls wrong or not supported by their notes.
    Sentences left without a usable answer are asked again once, four at a
    time (on 21 Nov local Qwen answered for 26 of 54 sentences)."""
    todo = [w for w in written if not w.dropped and not w.summary and not w.check]
    _check(todo, notes, name, ask, report, CHECK_BATCH)
    again = [w for w in todo if not w.check]
    _check(again, notes, name, ask, report, 4)
    for w in again:
        w.check = w.check or "unchecked"


def _check(
    kept: List[Written], notes: Sequence[Note], name: str, ask: Ask,
    report: Dict[str, Any], size: int,
) -> None:  # fmt: skip
    for start in range(0, len(kept), size):
        batch = kept[start : start + size]
        try:
            reply = ask(check_messages(batch, notes, name), SentenceChecks, 4096)
        except Exception as exc:
            report.setdefault("check_errors", []).append(str(exc)[:200])
            for w in batch:
                w.check = "error"
            continue
        report["calls"] = report.get("calls", 0) + 1
        for c in reply.checks or []:
            if not 1 <= c.sentence <= len(batch):
                continue
            w = batch[c.sentence - 1]
            if c.starts_with and _first_words(c.starts_with, 2) != _first_words(
                w.text, 2
            ):
                continue  # the answer is about another sentence
            verdict = str(c.verdict or "").strip().lower().replace(" ", "_")
            if verdict not in ("supported", "partial", "wrong", "not_supported"):
                continue
            w.check, w.problem = verdict, str(c.problem or "")[:300]
            if verdict in ("wrong", "not_supported"):
                w.dropped = "check: " + w.problem
