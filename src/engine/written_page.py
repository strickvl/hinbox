"""Write a readable encyclopedia page from a profile's notes.

Why: the stored profile is a notebook. Each article that touches a page adds
or changes a few cited sentences ("notes"), in the order the articles came.
On 21 Nov the Karachi page, after its off-topic notes were hidden, was still
23 sentences of who was arrested there, one per article, under the headings
"Background", "Role" and "Karachi"; the CIA page was 456,000 characters with
a section named after one detainee. Hiding notes cannot turn that into a
page someone would read. This module writes a second text from the notes,
and leaves the notes as they are, underneath it, as the evidence.

What happens for one page:

1. ``page_notes()`` numbers its notes (N1, N2 ...), each with its cited
   articles and the oldest article's date. Notes tagged ``other`` by the
   note-relevance step are left out.
2. When the notes are too long for one call (``WRITE_CHARS``), ``condense()``
   asks for the main facts of each run of notes, each fact naming the notes
   it came from, and repeats on the facts until they fit. A fact keeps the
   note numbers, so every written sentence still leads back to notes.
3. ``write()`` asks for an opening that says what the thing is, sections
   under plain headings (the kind's starter headings from the domain config
   are offered), and prose, each sentence listing the notes it uses.
4. ``code_check()`` drops a sentence that cites no real note, or holds a
   number, month or capitalised name its notes don't have (allowing
   "Pakistani" for "Pakistan"). ``model_check()`` then shows the model each
   sentence beside its notes and drops one it calls ``not_supported``;
   ``partial`` is kept and recorded.
5. ``render()`` turns what is left into Markdown with one citation marker per
   sentence, naming the articles of its notes, so the reader's citations
   still open the sources.

Stored as a JSON string under ``profile["written_json"]``: one entry per
model, with the text, each sentence's note keys (``sentence_key()``) and
check results, and counts. The notes' text never changes. A prototype: it
runs on a copy of a finished run (``scripts/write_pages.py``), not in the
pipeline. Once written, a page is meant to change only by small recorded
edits as articles arrive; that part is not built yet.
"""

import json
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field

from src.utils.claim_spans import parse_claims
from src.utils.corrections import sentence_key
from src.utils.note_tags import stored_tags

WRITTEN_KEY = "written_json"
WRITE_CHARS = 30000  # notes shown to one writing call
CONDENSE_CHARS = 12000  # notes shown to one condensing call
FACTS_PER_RUN = 10  # facts asked for per condensing call
MAX_ROUNDS = 4
CHECK_BATCH = 8  # sentences per model-check call
CHECK_NOTE_CHARS = 1500  # each note shown to the check at most this long
MAX_NOTES_SHOWN = 8  # notes shown per sentence to the check

_HEADING = re.compile(r"^[ \t]*(#{1,6})[ \t]+(.+?)[ \t#]*$", re.MULTILINE)


# --- notes ----------------------------------------------------------------


@dataclass
class Note:
    n: int  # 1-based, as the model sees it
    key: str
    text: str
    articles: List[str]
    date: str = ""
    heading: str = ""
    tag: str = ""
    keys: List[str] = field(default_factory=list)  # the notes a fact came from


def _day(value: Any) -> str:
    """ "2016-08-12" from an ISO date, a datetime or epoch seconds."""
    if value is None or value == "":
        return ""
    if hasattr(value, "isoformat"):
        return value.isoformat()[:10]
    text = str(value).strip()
    if re.fullmatch(r"\d{9,11}(\.\d+)?", text):
        return datetime.fromtimestamp(float(text), timezone.utc).date().isoformat()
    return text[:10]


def _dates(entity: Dict[str, Any]) -> Dict[str, str]:
    out = {}
    for a in entity.get("articles") or []:
        if isinstance(a, dict) and a.get("article_id"):
            out[str(a["article_id"])] = _day(a.get("article_published_date"))
    return out


def page_notes(entity: Dict[str, Any], keep_context: bool = True) -> List[Note]:
    """The page's notes in text order, without those tagged ``other`` (and
    ``context`` ones when ``keep_context`` is false)."""
    profile = entity.get("profile") or {}
    text = str(profile.get("text") or "")
    tags = stored_tags(profile)
    dates = _dates(entity)
    heads = [(m.start(), m.group(2).strip()) for m in _HEADING.finditer(text)]
    claims, _ = parse_claims(text)
    out: List[Note] = []
    seen = set()
    for c in claims:
        key = sentence_key(c.text)
        if not key or key in seen:
            continue
        seen.add(key)
        tag = str((tags.get(key) or {}).get("verdict") or "")
        if tag == "other" or (tag == "context" and not keep_context):
            continue
        heading = ""
        for at, title in heads:
            if at <= c.start:
                heading = title
        days = sorted(d for d in (dates.get(a, "") for a in c.articles) if d)
        out.append(
            Note(len(out) + 1, key, key, list(c.articles), days[0] if days else "",
                 heading, tag, [key])
        )  # fmt: skip
    return out


def notes_block(notes: Sequence[Note]) -> str:
    return "\n".join(
        f"N{n.n} ({n.date or 'no date'}{', ' + n.heading if n.heading else ''}): {n.text}"
        for n in notes
    )


# --- replies --------------------------------------------------------------


class Fact(BaseModel):
    model_config = ConfigDict(extra="ignore")
    text: str = Field(description="One fact, in a plain sentence")
    notes: List[int] = Field(description="Numbers of the notes it comes from")


class Facts(BaseModel):
    model_config = ConfigDict(extra="ignore")
    facts: List[Fact]


class WSentence(BaseModel):
    model_config = ConfigDict(extra="ignore")
    text: str = Field(description="One sentence of the page, without note numbers")
    notes: List[int] = Field(description="Numbers of the notes this sentence uses")


class WParagraph(BaseModel):
    model_config = ConfigDict(extra="ignore")
    sentences: List[WSentence]


class WSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    heading: str
    paragraphs: List[WParagraph]


class WBody(BaseModel):
    model_config = ConfigDict(extra="ignore")
    paragraphs: List[WParagraph]


class WPage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    opening: List[WSentence] = Field(description="One to three sentences")
    sections: List[WSection]


class SentenceCheck(BaseModel):
    model_config = ConfigDict(extra="ignore")
    sentence: int
    starts_with: str = Field(description="The sentence's first three words, copied")
    problem: str = Field(description="What the notes lack or say otherwise, or 'none'")
    verdict: str = Field(description="supported, partial, wrong or not_supported")


class SentenceChecks(BaseModel):
    model_config = ConfigDict(extra="ignore")
    checks: List[SentenceCheck]


Ask = Callable[[List[Dict[str, str]], Any, int], Any]  # messages, model, max_tokens


# --- prompts --------------------------------------------------------------

CONDENSE = """You prepare notes for a short encyclopedia entry about the \
{what} "{name}". The notes below are part of a long file of sentences taken \
from news articles, in the order the articles came in.

List at most {k} facts from these notes that an encyclopedia entry about \
{name} could use: what it is, what it did, decided or said, what was done \
to it or happened there, and its part in the events.

Rules:
- Merge notes that say the same thing into one fact.
- Keep dates, numbers and names exactly as the notes give them.
- Keep who said what: a claim, allegation or assessment stays attributed \
("the Pentagon says ...", "his lawyers said ...").
- Leave out notes mainly about someone or something else, where {name} is \
only a setting or a passing mention.
- Each fact lists the numbers of the notes it comes from.
- Use only what the notes say."""

WRITE = """You write the encyclopedia entry for the {what} "{name}", for a \
reference work built from news reporting. You are given numbered notes \
(sentences taken from articles, with each article's date). Write the entry \
from them.

1. Opening: one to three sentences. The first says plainly what {name} is \
("X is a city in ...", "X is a ... held at ..."), from the notes only. Then \
say why {name} matters in these notes.
2. Sections: after the opening, write sections under short plain headings. \
Headings that suit this kind of page:
{headings}
Use those that fit, in that order, and leave out any with nothing to say; \
you may add up to two of your own. A heading names a topic, never another \
person, a single article or a date.
3. Write as a careful historian for a general reader: plain, neutral words, \
with dates in the sentences. Join related facts: several similar facts \
(arrests in one city, a series of hearings) become one sentence or one \
paragraph, not one sentence each. Keep events in date order within a \
section.
4. Use only what the notes say. Never add a name, number, date, place, \
cause or motive the notes don't give. Don't guess.
5. Keep who said what. A claim, allegation or assessment stays attributed \
("the Pentagon describes him as ...", "an intelligence assessment said ..."). \
Loaded words (terrorist, jihadist, mastermind, extremist) appear only inside \
quotation marks or attributed to whoever used them. Quoted words stay \
exactly as quoted.
6. Leave out notes mainly about someone or something else, unless a reader \
needs them to follow a sentence about {name}. A page about a place is about \
the place and what happened there, not the later lives of the people who \
passed through it.
7. When notes give different dates or numbers for the same thing, say so in \
one sentence that gives both and cites both. Don't set two different things \
against each other.
8. Every sentence lists the numbers of the notes it uses (at least one) in \
its notes field; never write note numbers into the sentence itself.
9. Length: about {words} words in all. Leave out minor details before you \
leave out a main fact."""

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
- wrong: a name, date, number or place differs from the notes, or a fact \
the notes give about one person or event is given to another.
- not_supported: its notes do not say this.

Each note starts with its article's date in brackets. A sentence may use \
that date for when something was said, reported or true ("In May 2016, he \
said ...", "By 2016 ..."); that counts as in the notes. Judge only by the \
notes shown with each sentence, not by what you know."""


def _what(entity_type: str, kind: Optional[str]) -> str:
    base = {"people": "person", "organizations": "organization",
            "locations": "place", "events": "event"}.get(entity_type, entity_type)  # fmt: skip
    kind = str(kind or "").rsplit(".", 1)[-1].replace("_", " ").strip()
    return f"{base} ({kind})" if kind and kind != "other" else base


def target_words(n_notes: int) -> int:
    """About 60 words for 3 notes, 250 for 25, 900 for 200, at most 1200."""
    return int(max(60, min(1200, 22 * n_notes**0.75)))


# --- condensing -----------------------------------------------------------


def _runs(notes: Sequence[Note], limit: int) -> Iterable[List[Note]]:
    run: List[Note] = []
    size = 0
    for note in notes:
        if run and size + len(note.text) > limit:
            yield run
            run, size = [], 0
        run.append(note)
        size += len(note.text) + 20
    if run:
        yield run


def _numbered(notes: Sequence[Note]) -> List[Note]:
    return [
        Note(i + 1, n.key, n.text, n.articles, n.date, n.heading, n.tag, n.keys)
        for i, n in enumerate(notes)
    ]


def facts_to_notes(reply: Facts, shown: Sequence[Note]) -> List[Note]:
    """Each fact as a note holding the keys, articles and oldest date of the
    shown notes it names; a fact naming none of them is dropped."""
    by_n = {n.n: n for n in shown}
    out: List[Note] = []
    for fact in reply.facts or []:
        src = [by_n[i] for i in dict.fromkeys(fact.notes or []) if i in by_n]
        text = " ".join(str(fact.text or "").split())
        if not src or not text:
            continue
        keys = list(dict.fromkeys(k for s in src for k in s.keys))
        arts = list(dict.fromkeys(a for s in src for a in s.articles))
        days = sorted(s.date for s in src if s.date)
        out.append(Note(0, text, text, arts, days[0] if days else "", "", "", keys))
    return out


def condense(
    notes: List[Note], name: str, what: str, ask: Ask, report: Dict[str, Any],
    workers: int = 1,
) -> List[Note]:  # fmt: skip
    """Facts drawn from the notes until they fit one writing call. A run whose
    call fails, or whose facts name none of its notes, keeps its notes."""
    from concurrent.futures import ThreadPoolExecutor

    system = CONDENSE.format(what=what, name=name, k=FACTS_PER_RUN)

    def one(run: List[Note]) -> List[Note]:
        shown = _numbered(run)
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": f"Notes:\n{notes_block(shown)}"}]  # fmt: skip
        try:
            reply = ask(msgs, Facts, 4096)
        except Exception as exc:
            report.setdefault("condense_errors", []).append(str(exc)[:200])
            return run
        report["calls"] = report.get("calls", 0) + 1
        return facts_to_notes(reply, shown) or run

    rounds = 0
    while sum(len(n.text) + 20 for n in notes) > WRITE_CHARS and rounds < MAX_ROUNDS:
        rounds += 1
        runs = list(_runs(notes, CONDENSE_CHARS))
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            out = [n for facts in pool.map(one, runs) for n in facts]
        if len(out) >= len(notes):
            break
        notes = _numbered(out)
    report["condense_rounds"] = rounds
    return _numbered(notes)


# --- writing --------------------------------------------------------------


def write_messages(
    notes: Sequence[Note], name: str, what: str, headings: Sequence[Tuple[str, str]],
    words: int = 0,
) -> List[Dict[str, str]]:  # fmt: skip
    heads = "\n".join(f"- {h}: {note}" if note else f"- {h}" for h, note in headings)
    system = WRITE.format(
        what=what, name=name, headings=heads or "- (choose your own)",
        words=words or target_words(len(notes)),
    )  # fmt: skip
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Notes about {name}:\n{notes_block(notes)}"},
    ]


@dataclass
class Written:
    text: str  # one sentence, as written
    notes: List[int]
    section: str = ""  # "" for the opening
    para: int = 0
    dropped: str = ""  # why it was dropped, "" when kept
    check: str = ""  # the model check's verdict
    problem: str = ""
    summary: bool = False  # an opening sentence summing up the sections
    revised: str = ""  # the dropped sentence this one rewrites


# A note reference written into the sentence: "[N1, N2]", "(N12)", "{11, 14}",
# "(notes 2, 3)", or bare numbers in parentheses, "(14, 3)", when all are
# note numbers
_REF = re.compile(
    r"\s*([\[({])\s*((?:notes?\s*|N\s*)?\d{1,3}"
    r"(?:\s*(?:,|;|and|&|-|–)\s*(?:N\s*)?\d{1,3})*)\s*[\])}]",
    re.IGNORECASE,
)


def strip_refs(text: str, max_n: int = 999) -> Tuple[str, List[int]]:
    """``text`` without the note references a model wrote into it, and the
    note numbers they named. Luna's section calls did this on 21 Nov ("...
    in 2002 [N1]."), and the code check then dropped 64 of 90 sentences for
    names and numbers "not in its notes"."""
    found: List[int] = []

    def cut(m: "re.Match[str]") -> str:
        body = m.group(2)
        nums = [int(x) for x in re.findall(r"\d+", body)]
        marked = m.group(1) in "[{" or re.match(r"(?i)notes?|n", body)
        if not marked and not all(1 <= x <= max_n for x in nums):
            return m.group(0)
        found.extend(nums)
        return ""

    out = _REF.sub(cut, text)
    out = re.sub(r"\s+([.,;:])", r"\1", " ".join(out.split()))
    return out, found


# A sentence end: ". " before a capital, except after an initial or a short
# abbreviation ("U.S. Army", "Gen. Smith", "J. Doe")
_END = re.compile(r"(?<=[.!?])[\"”’)]?\s+(?=[\"“]?[A-Z])")
_ABBR = re.compile(
    r"(?:\b[A-Z]|\b[A-Z]\.[A-Z]|\b(?:Mr|Mrs|Ms|Dr|Gen|Col|Lt|Sgt|Capt|Adm|Maj|Gov|Sen|Rep|St|Jr|Sr|No|Vol|Co|Inc|Ltd|vs))\.$"
)


def split_sentences(text: str) -> List[str]:
    """One item a model gave as several sentences, cut into sentences."""
    out, start = [], 0
    for m in _END.finditer(text):
        if _ABBR.search(text[start : m.start()]):
            continue
        out.append(text[start : m.start()].strip())
        start = m.end()
    out.append(text[start:].strip())
    return [t for t in out if t]


def _pieces(w: "Written", max_n: int) -> List["Written"]:
    """``w`` cut into its sentences, each without note references; a
    sentence that named notes keeps those, else it keeps the item's."""
    raw = " ".join(str(w.text or "").split())
    parts = [strip_refs(t, max_n) for t in split_sentences(raw)] or [("", [])]
    named = [n for _, refs in parts for n in refs]
    out = []
    for text, refs in parts:
        notes = list(dict.fromkeys(refs or w.notes or named))
        out.append(Written(text, notes, w.section, w.para))
    return out


def flatten(page: WPage, max_n: int = 999, summary: bool = False) -> List[Written]:
    """The page's sentences in order. With ``summary``, opening sentences
    after the first are marked as summing up the sections."""
    out = _flatten(page, max_n)
    if summary:
        for w in [w for w in out if not w.section][1:]:
            w.summary = True
    return out


def _flatten(page: WPage, max_n: int) -> List[Written]:
    out = [Written(s.text, list(s.notes or []), "", 0) for s in page.opening or []]
    p = 1
    for sec in page.sections or []:
        heading = " ".join(str(sec.heading or "").strip("# ").split())
        for para in sec.paragraphs or []:
            out += [
                Written(s.text, list(s.notes or []), heading, p)
                for s in para.sentences or []
            ]
            p += 1
    out = [piece for w in out for piece in _pieces(w, max_n)]
    return [w for w in out if w.text]


# --- checks ---------------------------------------------------------------

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
    stems = {w[:5] for w in words if len(w) >= 5}
    if re.search(r"\bu\.?s\b|\bunited states\b", hay):
        words |= _US_WORDS

    def present(low: str) -> bool:
        if len(low) < 2 or low in _COMMON or low in _MONTHS or low in words:
            return True
        if low.endswith("'s") and low[:-2] in words:
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
        # A sentence's first word is capitalised anyway ("Reports say ...",
        # "Regarding Walid ..."), so it is left to the model check
        if not sentence[: m.start()].strip(" \"'“‘("):
            continue
        if not present(_fold(tok).strip("'-")):
            missing.append(tok)
    return list(dict.fromkeys(missing))


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


def _content(text: str) -> set:
    return {
        w for w in re.findall(r"[\w']+", _fold(text)) if len(w) > 2 and w not in _COMMON
    }


def drop_repeats(written: List[Written], share: float = 0.8) -> None:
    """Drop a section sentence whose words are nearly all in one earlier
    section sentence (a one-sentence section saying again what the first
    section said; on 21 Nov Luna's Karachi page said one fact three times)."""
    seen: List[set] = []
    for w in written:
        if w.dropped or not w.section:
            continue
        words = _content(w.text)
        if words and any(len(words & old) >= share * len(words) for old in seen):
            w.dropped = "repeats an earlier sentence"
            continue
        seen.append(words)


# An opening sentence about the page instead of its subject
_ABOUT_PAGE = re.compile(
    r"\b(?:this|the) (?:entry|page|article|record)\b|\bthese notes\b|\bthe notes\b",
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
        if _ABOUT_PAGE.search(w.text):
            w.dropped = "talks about the page"
            continue
        w.notes = [n for n in w.notes if n in by_n] or [
            n for b in body for n in b.notes[:1]
        ][:6]
        gaps = missing_details(w.text, sources, name)
        if gaps:
            w.dropped = "not in the sections: " + ", ".join(gaps[:5])
        else:
            w.check = "summary"


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
        parts.append(f"Sentence {i}: {w.text}\nIts notes:\n{lines}")
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


# --- output ---------------------------------------------------------------


def _articles(w: Written, by_n: Dict[int, Note]) -> List[str]:
    return list(dict.fromkeys(a for n in w.notes for a in by_n[n].articles))


def render(written: List[Written], notes: Sequence[Note]) -> str:
    """Markdown: opening paragraph, then ``## heading`` sections; each kept
    sentence followed by one marker naming its notes' articles."""
    by_n = {n.n: n for n in notes}
    blocks: List[str] = []
    heading: Optional[str] = None
    para: Optional[int] = None
    line: List[str] = []
    for w in (w for w in written if not w.dropped):
        if para is not None and (w.para != para or w.section != heading):
            blocks.append(" ".join(line))
            line = []
        if w.section != heading and w.section:
            blocks.append(f"## {w.section}")
        heading, para = w.section, w.para
        text = w.text if w.text[-1] in ".!?\"”'" else w.text + "."
        line.append(f"{text}^[{', '.join(_articles(w, by_n))}]")
    if line:
        blocks.append(" ".join(line))
    return "\n\n".join(blocks) + "\n"


def write_page(
    entity: Dict[str, Any], entity_type: str, model: str, ask: Ask,
    headings: Sequence[Tuple[str, str]] = (), check: bool = True, workers: int = 1,
    approach: str = "plan",
) -> Dict[str, Any]:  # fmt: skip
    """The written page for one entity, with its record, or a record with an
    ``error`` and no text. ``approach`` "plan" plans the page and writes it a
    section at a time (``written_plan.py``); "single" writes it in one call.
    The record's ``model`` is the model, with "#plan" after it for the
    planned approach, so both can be stored side by side."""
    from src.engine.written_plan import write_planned
    from src.engine.written_revise import revise

    name = str(entity.get("name") or entity.get("title") or "")
    what = _what(entity_type, entity.get("type") or entity.get("event_type"))
    key = model if approach == "single" else f"{model}#{approach}"
    report: Dict[str, Any] = {"calls": 0, "approach": approach}
    notes = page_notes(entity)
    report["notes"] = len(notes)
    if not notes:
        return {"model": key, "error": "no notes", "report": report}
    shown = condense(notes, name, what, ask, report, workers)
    report["shown"] = len(shown)
    words = target_words(len(notes))
    if approach == "plan":
        page = write_planned(
            shown, name, what, headings, words, ask, report, notes_block,
            WPage, WBody, WParagraph,
        )  # fmt: skip
        if page is None:
            return {"model": key, "error": "; ".join(report.get("errors") or ["no plan"]),
                    "report": report}  # fmt: skip
    else:
        try:
            page = ask(write_messages(shown, name, what, headings, words), WPage, 8192)
        except Exception as exc:
            return {"model": key, "error": str(exc)[:300], "report": report}
        report["calls"] += 1
    written = flatten(
        page, max((n.n for n in shown), default=0), summary=approach == "plan"
    )
    code_check([w for w in written if not w.summary], shown, name)
    drop_repeats(written)
    if check:
        model_check(written, shown, name, ask, report)
        written = revise(written, shown, name, ask, report)
    summary_check(written, shown, name)
    kept = [w for w in written if not w.dropped]
    by_n = {n.n: n for n in shown}
    used = {k for w in kept for n in w.notes for k in by_n[n].keys}
    report.update(
        sentences=len(written), kept=len(kept),
        dropped_code=sum(1 for w in written if w.dropped and not w.dropped.startswith("check")),
        dropped_check=sum(1 for w in written if w.dropped.startswith("check")),
        partial=sum(1 for w in kept if w.check == "partial"),
        notes_used=len(used), words=sum(len(w.text.split()) for w in kept),
        headings=list(dict.fromkeys(w.section for w in kept if w.section)),
    )  # fmt: skip
    return {
        "model": key,
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "text": render(written, shown) if kept else "",
        "sentences": [
            {
                "text": w.text,
                "section": w.section,
                "notes": [k for n in w.notes for k in by_n[n].keys]
                if w.notes and all(n in by_n for n in w.notes)
                else [],
                "dropped": w.dropped,
                "check": w.check,
                "problem": w.problem,
                "revised": w.revised,
            }
            for w in written
        ],  # fmt: skip
        "report": report,
    }


def stored_pages(profile: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Written pages by model; ``{}`` when none or unreadable."""
    raw = (profile or {}).get(WRITTEN_KEY)
    if not raw:
        return {}
    try:
        pages = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except (TypeError, ValueError):
        return {}
    return {
        str(m): p
        for m, p in (pages or {}).items()
        if isinstance(p, dict) and p.get("text")
    }


def store_page(profile: Dict[str, Any], page: Dict[str, Any]) -> None:
    raw = profile.get(WRITTEN_KEY)
    try:
        pages = json.loads(raw) if isinstance(raw, str) and raw else {}
    except ValueError:
        pages = {}
    pages[page["model"]] = page
    profile[WRITTEN_KEY] = json.dumps(pages, ensure_ascii=False)


def make_ask(model: str) -> Ask:
    from src.utils.llm import cloud_generation, local_generation
    from src.utils.step_timing import timed_step

    generate = local_generation if model.startswith("ollama/") else cloud_generation

    def ask(msgs: List[Dict[str, str]], response_model: Any, max_tokens: int) -> Any:
        with timed_step("profile.written_page"):
            reply = generate(
                messages=msgs, response_model=response_model, model=model,
                max_tokens=max_tokens,
            )  # fmt: skip
        if reply is None:
            raise RuntimeError("written page step returned nothing")
        return reply

    return ask
