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


# A note that only comments on the one before ("This highlights the
# practical hurdles ...", "The city represents the endpoint of his journey")
_COMMENT = re.compile(
    r"(?:this|these|that|it|the (?:city|case|episode|move|decision)) "
    r"(?:highlights?|illustrates?|underscores?|represents?|demonstrates?|reflects?|"
    r"shows?|exemplifies|symboli[sz]es?)\b",
    re.IGNORECASE,
)


# The old profile's own "sources disagree" sentences: on 21 Nov "one article
# identifies the detainee rebranded as a 'forever prisoner' as Suhayl al
# Sharabi ... rather than Paracha" came back on the Karachi page in two
# models' runs as if it were a fact about Karachi
_OLD_DISPUTE = re.compile(
    r"^(?:sources|accounts|reports) (?:disagree|differ)|^another report differs|"
    r"\bidentif(?:y|ies|ied)\b[^.]*\brather than\b",
    re.IGNORECASE,
)


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
        if _COMMENT.match(key) or _OLD_DISPUTE.search(key):
            continue  # the old profile's own comment or "sources disagree", no fact
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
    filled: bool = False  # written for planned notes no sentence used


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

    out = _REF.sub(cut, _brackets(text))
    # A note's date prefix copied into the sentence: "[2017-05-20; 2011-05-01]"
    out = re.sub(r"\s*\[\d{4}-\d\d-\d\d(?:\s*[;,]\s*\d{4}-\d\d-\d\d)*\]", "", out)
    out = re.sub(r"\s+([.,;:])", r"\1", " ".join(out.split()))
    out = re.sub(r"\.\.$", ".", out)
    return close_quotes(out), found


def close_quotes(text: str) -> str:
    """``text`` with an opened quotation closed at its end (GPT-6 Luna's
    Zaki page, 21 Nov: '... “let out their steam on paper instead of
    taking it out on the guard force.' and no closing mark)."""
    if text.count("“") > text.count("”"):
        return text + "”"
    if text.count('"') % 2:
        return text + '"'
    return text


def _brackets(text: str) -> str:
    """``text`` without a model's square-bracket asides. On 21 Nov a fill
    reply's "[The existing section already notes ... adds no distinct
    fact.]" was printed on a page, and a section writer marked a name it
    put in as "[Clarke] described ...". Outside quotation marks a short
    bracket (up to three words) keeps its words and loses the brackets, a
    longer one goes; inside a quotation an editor's "[the men]" stays.
    Note references ("[N3]", "[2017-05-20]") are left for the code below."""
    out, quoted = [], False
    for piece in re.split(r'(["“”]|\[[^\[\]]*\])', text):
        if piece in ('"', "“", "”"):
            quoted = piece == "“" or (piece == '"' and not quoted)
            out.append(piece)
        elif piece.startswith("[") and piece.endswith("]") and not quoted:
            body = piece[1:-1].strip()
            if not body or (
                re.search(r"\d", body)
                and re.fullmatch(r"(?i)(?:notes?|and|n|[\d\s,;&–-])+", body)
            ):
                out.append(piece)  # a note reference or date: cut below
            elif len(body.split()) <= 3:
                out.append(body)
        else:
            out.append(piece)
    return "".join(out)


# A sentence end: ". " before a capital, except after an initial or a short
# abbreviation ("U.S. Army", "Gen. Smith", "J. Doe")
_END = re.compile(r"(?<=[.!?])[\"”’)]?\s+(?=[\"“]?[A-Z])")
_ABBR = re.compile(
    r"(?:\b[A-Z]|\b[A-Z]\.[A-Z]|\b(?:Mr|Mrs|Ms|Dr|Gen|Gens|Brig|Col|Lt|Sgt|Capt|Cmdr|Cmdt|Adm|Maj|Pvt|Cpl|Spc|Gov|Sen|Rep|Prof|Rev|Hon|Det|Insp|Supt|Ft|Mt|St|Jr|Sr|No|Vol|Co|Corp|Inc|Ltd|vs|v|Jan|Feb|Mar|Apr|Aug|Sept|Sep|Oct|Nov|Dec))\.$"
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


# A sentence opening by narrating its source ("The article states that ...")
_NARRATES = re.compile(
    r"^(?:the|this|one) (?:article|report|note|source)s? (?:states?|says?|"
    r"reports?|notes?|mentions?) that\s+",
    re.IGNORECASE,
)


def _unnarrate(text: str) -> str:
    out = _NARRATES.sub("", text, count=1)
    return out[:1].upper() + out[1:] if out != text else text


def _pieces(w: "Written", max_n: int) -> List["Written"]:
    """``w`` cut into its sentences, each without note references; a
    sentence that named notes keeps those, else it keeps the item's."""
    raw = " ".join(str(w.text or "").split())
    parts = [strip_refs(t, max_n) for t in split_sentences(_brackets(raw))] or [
        ("", [])
    ]
    named = [n for _, refs in parts for n in refs]
    out = []
    for text, refs in parts:
        notes = list(dict.fromkeys(refs or w.notes or named))
        out.append(Written(_unnarrate(text), notes, w.section, w.para))
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


def give_identity_a_note(
    written: List[Written], report: Dict[str, Any], shown: Sequence[Note]
) -> None:
    """The opening's first sentence ("X is a Yemeni former Guantánamo
    detainee.") cites the plan's notes on what X is when it names none it
    may cite: on 21 Nov one Luna page lost it as "cites no note" and never
    said who he was."""
    known = {n.n for n in shown}
    first = next((w for w in written if not w.section), None)
    if first is None or any(n in known for n in first.notes):
        return
    plan = report.get("plan") or {}
    first.notes = [n for n in plan.get("opening_notes") or [] if n in known][:3] or [
        n.n for n in shown[:1]
    ]


def write_page(
    entity: Dict[str, Any], entity_type: str, model: str, ask: Ask,
    headings: Sequence[Tuple[str, str]] = (), check: bool = True, workers: int = 1,
    approach: str = "plan",
    label: str = "",
    draws: int = 1,
) -> Dict[str, Any]:  # fmt: skip
    """The written page for one entity, with its record, or a record with an
    ``error`` and no text. ``approach`` "plan" plans the page and writes it a
    section at a time (``written_plan.py``); "single" writes it in one call.
    The record's ``model`` is the model, with "#plan" after it for the
    planned approach, so both can be stored side by side. With ``draws`` over
    1 the plan is made once and the page written that many times, each
    section taken from the draw that scores best (``written_draws.py``)."""
    from src.engine.written_draws import finish
    from src.engine.written_plan import make_plan, write_from_plan

    name = str(entity.get("name") or entity.get("title") or "")
    what = _what(entity_type, entity.get("type") or entity.get("event_type"))
    key = model if approach == "single" else f"{model}#{approach}"
    key += f"@{label}" if label else ""
    report: Dict[str, Any] = {"calls": 0, "approach": approach}
    notes = page_notes(entity)
    report["notes"] = len(notes)
    if not notes:
        return {"model": key, "error": "no notes", "report": report}
    shown = condense(notes, name, what, ask, report, workers)
    report["shown"] = len(shown)
    words = target_words(len(notes))
    if approach == "plan":
        plan = make_plan(shown, name, what, headings, ask, report, notes_block)
        page = None
        if plan is not None and draws <= 1:
            page = write_from_plan(
                plan, shown, name, what, words, ask, report, notes_block,
                WPage, WBody, WParagraph,
            )  # fmt: skip
        if plan is None or (draws <= 1 and page is None):
            return {"model": key, "error": "; ".join(report.get("errors") or ["no plan"]),
                    "report": report}  # fmt: skip
    else:
        try:
            page = ask(write_messages(shown, name, what, headings, words), WPage, 8192)
        except Exception as exc:
            return {"model": key, "error": str(exc)[:300], "report": report}
        report["calls"] += 1
    if approach == "plan" and draws > 1:
        from src.engine.written_draws import write_drawn

        written = write_drawn(
            plan, shown, name, what, words, ask, report, check, draws,
            local=model.startswith("ollama/"),
        )  # fmt: skip
    else:
        written = finish(page, shown, name, what, approach, check, ask, report)
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
                "filled": w.filled,
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
