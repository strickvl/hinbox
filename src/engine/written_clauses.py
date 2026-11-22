"""Repeats smaller than a sentence, found by code after the page is put
together: a fact told plainly once and again as a clause of a longer
sentence, and a fact told again as what a later report said.

Why: in round 28 (22 Nov) the chosen sections still had 16 repeated pairs.
Half were a clause: "On November 3, Rishikof suspended Brig. Gen. John
Baker's 21-day contempt sentence" and, two sentences on, "In February
2018, Rishikof supported the war court judge's view ..., and suspended
Baker's contempt sentence." Others were a repeat wrapped as a report under
its article's date, which the one-time-event check can't see: "In May
2018, reports confirmed that Brown was dismissed alongside Rishikof."
"""

import re
from typing import Any, Dict, List, Sequence, Tuple

from src.engine.written_page import Written

# "In May 2018, reports confirmed that X", "An August 2018 report said X",
# "X was reported to have been resettled"
_REPORT = re.compile(
    r"^(?:(?:In|By|On)\s+[^,]{3,30},\s+)?(?:an?\s+|the\s+|later\s+)?"
    r"(?:[A-Z][a-z]+\s+)?(?:\d{4}\s+)?(?:news\s+)?(?:reports?|accounts?|articles?)\s+"
    r"(?:said|confirmed|noted|stated|reported|added|showed)\s+(?:that\s+)?",
    re.I,
)
_REPORTED = re.compile(r"\b(?:was|were) reported to have (?:been )?", re.I)


_NUMBER_WORDS = re.compile(
    r"\b(two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|"
    r"fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|dozens?|"
    r"hundreds?)\b",
    re.I,
)


def _facts(text: str, name: str) -> Tuple[set, set, set]:
    """Words, numbers (number words too: round 30 cut "two were
    hospitalized" as told by a sentence about three) and people."""
    from src.engine.written_checks import _content, _numbers
    from src.engine.written_plan import _surnames

    nums = _numbers(text) | {m.lower() for m in _NUMBER_WORDS.findall(text)}
    return _content(text), nums, _surnames(text, name)


def _when_of(text: str) -> Any:
    from src.engine.written_groups import text_when

    return text_when(text)


def _within(
    part: str, earlier: List[str], name: str, share: float, when: Any = None
) -> bool:
    """Every number and person of ``part``, and ``share`` of its words, in
    one earlier sentence. A part with a number told under another month
    (``when``) is not the same count: round 31 cut "two captives were
    hospitalized for intravenous drips" in April for a "two" of March 19."""
    words, nums, people = _facts(part, name)
    if len(words) < 3:
        return False
    for old in earlier:
        w, n, p = _facts(old, name)
        other = _when_of(old)
        if nums and when and other and when[:2] != other[:2]:
            continue
        if nums <= n and people <= p and len(words & w) >= share * len(words):
            return True
    return False


def unwrap_report(text: str) -> str:
    """``text`` without "In May 2018, reports confirmed that" or "was
    reported to have been"; ``text`` itself when it has neither."""
    m = _REPORT.match(text)
    out = text[m.end() :] if m else text
    return _REPORTED.sub("was " if "was" in text else "were ", out)


def drop_report_repeats(written: List[Written], name: str) -> List[str]:
    """Drop a section sentence that only tells, as a later report, what an
    earlier kept sentence of the page already tells; the dropped texts."""
    dropped: List[str] = []
    earlier: List[str] = []
    for w in written:
        if w.dropped:
            continue
        inner = unwrap_report(w.text)
        # the report's own date went with its wrapper: it is not a fact of
        # what the report says
        if w.section and inner != w.text and _within(inner, earlier, name, 0.6):
            w.dropped = "tells again, as a later report, an earlier sentence"
            dropped.append(w.text)
            continue
        earlier.append(w.text)
    return dropped


def _clauses(text: str) -> List[Tuple[str, str]]:
    """``text`` as (join, clause) pairs, the first join "". A clause starts
    lower case after ";" or ", and"; the last item of a list ("upheld X,
    suspended Y, and struck down Z") stays with its list, since round 30 cut
    "upheld Baker's conviction, suspended confinement" and left "and struck
    down the $1,000 fine" on the sentence before."""
    out: List[Tuple[str, str]] = []
    for k, part in enumerate(re.split(r";\s+(?=[a-z])", text)):
        join = "; " if k else ""
        bits = re.split(r"(,\s+and\s+)(?=[a-z])", part)
        piece = bits[0]
        for sep, rest in zip(bits[1::2], bits[2::2]):
            if re.search(r",\s+[a-z]", piece):  # a list: keep it whole
                piece += sep + rest
                continue
            out.append((join, piece))
            join, piece = sep, rest
        out.append((join, piece))
    return out


def _small_count(text: str) -> bool:
    """A count that two facts share by chance: a number word or a number
    under 10. Three rounds running (30-32) the clause cut lost "two
    detainees were hospitalized for intravenous drips" for another "two",
    each time by a different route; a distinctive count ("about 140 Navy
    doctors") is still cut when an earlier sentence has it."""
    if _NUMBER_WORDS.search(text):
        return True
    return any(int(n) < 10 for n in re.findall(r"(?<![\d,.$])\d+(?![\d,.%])", text))


def drop_repeat_clauses(written: List[Written], name: str) -> List[List[str]]:
    """Cut from a section sentence a clause that an earlier kept sentence of
    its section tells (its numbers, people and 60% of its words); each cut
    as [before, after]."""
    cut: List[List[str]] = []
    said: dict = {}
    for w in written:
        if w.dropped or not w.section:
            continue
        before = said.setdefault(w.section, [])
        parts = _clauses(w.text)
        if len(parts) > 1 and before:
            keep = [parts[0]] + [
                (j, p) for j, p in parts[1:]
                if _small_count(p)
                or not _within(p, before, name, 0.6, _when_of(p) or _when_of(w.text))
            ]  # fmt: skip
            if len(keep) < len(parts):
                text = "".join(j + p for j, p in keep).rstrip(" ,;")
                if not text.endswith((".", "!", "?", '"', "”")):
                    text += "."
                if text != w.text:
                    cut.append([w.text, text])
                    w.text = text
        before.append(w.text)
    return cut


def _told_by(a: str, b: str, name: str) -> bool:
    """Everything ``a`` tells is in ``b``: no person or number ``b`` lacks,
    60% of its words in ``b`` and at most two words more."""
    (wa, na, pa), (wb, nb, pb) = _facts(a, name), _facts(b, name)
    return (
        len(wa) >= 3 and na <= nb and pa <= pb
        and len(wa & wb) >= 0.6 * len(wa) and len(wa - wb) <= 2
    )  # fmt: skip


def drop_section_repeats(written: List[Written], name: str) -> List[str]:
    """Drop a section sentence whose whole content another sentence of its
    section tells: the later one when the earlier tells it (round 29 kept
    "In April 2013, most of the men awaiting transfer were Yemenis,
    including Hamiri." right after a sentence saying so), the earlier one
    when the later, fuller one tells it (round 30). A sentence that adds a
    fact (three words or more) is left for the review's trim."""
    dropped: List[str] = []
    said: Dict[str, List[Written]] = {}
    for w in written:
        if w.dropped or not w.section:
            continue
        before = said.setdefault(w.section, [])
        old = next((o for o in before if _told_by(w.text, o.text, name)), None)
        if old is not None:
            w.dropped = "repeats an earlier sentence of its section"
            dropped.append(w.text)
            continue
        for o in [o for o in before if _told_by(o.text, w.text, name)]:
            o.dropped = "told again, more fully, later in its section"
            dropped.append(o.text)
            w.notes = list(dict.fromkeys(o.notes + w.notes))
            before.remove(o)
        before.append(w)
    return dropped


# A sentence that opens on its own date: "In June 2011, ...", "On April 13,
# 2013, ...", "By late March 2013, ..."
# (round 32: "On April 21, 2013," has a comma before its year, so it was not
# seen as dated and the camp's April sentences stayed out of order)
_OPENS_DATED = re.compile(
    r"^\W*(?:In|On|By|As of|Since|From|During|Early|Late|Until)\b"
    r"[^,]{0,30}?(?:\d,\s+)?\b(?:19|20)\d\d\b"
)


_PART = re.compile(r"\b(late|early|mid)\b-?\s*[A-Z][a-z]+\.?\s+(?:19|20)\d\d")
_PART_DAY = {"early": 5, "mid": 15, "late": 25}


def _sort_when(text: str) -> Any:
    """``_when_of`` for sorting: a month with no day goes after the month's
    dated days, "early"/"mid"/"late" at the 5th/15th/25th. Round 33 put
    "by late March 2013" before "On March 15, 2013" and "In late April
    2013" before "On April 11, 2013", since a missing day sorted as the 1st."""
    when = _when_of(text)
    if not when or not when[1] or when[2]:
        return when
    m = _PART.search(text)
    return (when[0], when[1], _PART_DAY[m.group(1).lower()] if m else 32)


def _caps(text: str) -> set:
    """Capitalised words after the first, four letters or more."""
    return {w for w in re.findall(r"(?<=\s)[A-Z][\w’'-]{3,}", text)}


def _rehome(runs: List[List[Written]], name: str) -> None:
    """Move an undated sentence that names nothing its run names, and does
    not lean on the sentence before ("He", "That month", "The X also"), to
    the end of the run sharing most of its names. Round 33 carried "Rishikof
    had also been in contact with a deputy of Sessions" behind a March 2018
    sentence it followed in the draft, away from the October 2017 one naming
    Sessions."""
    from src.engine.written_draws import _PRONOUN, _THAT_TIME, _THE_ALSO

    own = {w.lower() for w in re.findall(r"[\w’'-]+", name)}

    def names(t: str) -> set:
        return {w for w in _caps(t) if w.lower() not in own}

    for run in runs:
        for w in list(run[1:]):
            if (
                _PRONOUN.match(w.text)
                or _THAT_TIME.match(w.text)
                or _THE_ALSO.match(w.text)
            ):
                continue
            mine = names(w.text)
            if not mine or mine & set().union(
                *(names(o.text) for o in run if o is not w)
            ):
                continue
            best = max(
                (r for r in runs if r is not run),
                key=lambda r: len(mine & set().union(*(names(o.text) for o in r))),
                default=None,
            )
            if best and mine & set().union(*(names(o.text) for o in best)):
                run.remove(w)
                best.append(w)


def sort_paragraphs(written: List[Written], name: str = "") -> List[str]:
    """Put each paragraph's runs in date order: a run starts at a sentence
    that opens on its own date and takes the undated sentences after it, so
    "He ..." and "That month ..." move with the sentence they lean on; text
    before the first such sentence stays first. Round 31's camp section
    opened April 2013, June 2013, June 2011, September 2012, then February
    2013. The first sentence of each moved run, as a record."""
    moved: List[str] = []
    groups: Dict[Tuple[str, int], List[int]] = {}
    for i, w in enumerate(written):
        if not w.dropped and w.section:
            groups.setdefault((w.section, w.para), []).append(i)
    for idx in groups.values():
        runs: List[List[Written]] = []
        for i in idx:
            w = written[i]
            if _OPENS_DATED.match(w.text) or not runs:
                runs.append([w])
            else:
                runs[-1].append(w)
        head = [] if _OPENS_DATED.match(runs[0][0].text) else [runs.pop(0)]
        before = [list(r) for r in runs]
        _rehome(runs, name)
        order = sorted(runs, key=lambda r: _sort_when(r[0].text) or (9999, 0, 0))
        if order == runs and runs == before:
            continue
        moved += [r[0].text for k, r in enumerate(order) if r is not runs[k]]
        moved += [w.text for r, b in zip(runs, before) for w in r if w not in b]
        for i, w in zip(idx, [w for r in head + order for w in r]):
            written[i] = w
    return moved


def sort_section_paragraphs(written: List[Written]) -> List[str]:
    """Put each section's paragraphs in date order by the first sentence of
    each that opens on its own date; a paragraph with none stays after the
    one before it, and a first paragraph with none stays first. Round 32's
    camp section put a 2011-2013 paragraph before "The strike had been under
    way since early February 2013", and a Rishikof section opened in
    February 2018 before June 2017. The first sentence of each moved
    paragraph, as a record."""
    moved: List[str] = []
    sections: Dict[str, List[List[int]]] = {}
    last: Dict[str, Any] = {}
    for i, w in enumerate(written):
        if w.dropped or not w.section:
            continue
        paras = sections.setdefault(w.section, [])
        if not paras or last[w.section] != w.para:
            paras.append([])
        paras[-1].append(i)
        last[w.section] = w.para
    for paras in sections.values():
        keys: List[Any] = []
        for para in paras:
            first = next(
                (written[i].text for i in para if _OPENS_DATED.match(written[i].text)),
                None,
            )
            when = _sort_when(first) if first else None
            keys.append(when or (keys[-1] if keys else (0, 0, 0)))
        order = [p for _, p in sorted(zip(keys, paras), key=lambda kp: kp[0])]
        if order == paras:
            continue
        moved += [written[p[0]].text for k, p in enumerate(order) if p is not paras[k]]
        slots = sorted(i for p in paras for i in p)
        items = [written[i] for p in order for i in p]
        for i, w in zip(slots, items):
            written[i] = w
    return moved


def tidy_page(
    written: List[Written], notes: Sequence[Any], name: str, report: Dict[str, Any]
) -> None:
    """The code passes run on a finished page and again on a picked one;
    what each changed is added to the report, so nothing is lost from view."""
    from src.engine.written_dates import own_dates

    for key, found in (
        ("own_dates", own_dates(written, notes)),
        ("paragraphs_sorted", sort_paragraphs(written, name)),
        ("section_paragraphs_sorted", sort_section_paragraphs(written)),
        ("section_repeats", drop_section_repeats(written, name)),
        ("report_repeats", drop_report_repeats(written, name)),
        ("clauses_cut", drop_repeat_clauses(written, name)),
    ):
        report[key] = list(report.get(key) or []) + found
