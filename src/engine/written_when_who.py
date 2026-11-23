"""One narrow check on the finished page: is each date the date of what it
dates, and is each act given to whoever did it.

Why: on 22 Nov the blind readers of 16 held-out pages (pages never used
while tuning) named, more than any other factual fault, an article's
publication date put on an older event it only looks back on (five of 16
readers) and an act or statement given to the wrong person, or a related
case's detail given to this one (three). The general sentence check asks
about both among many things and let them through; by the stages rule a
step that struggles is split, so this asks only these two questions. It
runs on the page as it will be shown, after the drafts are picked and the
code passes have dated and moved sentences.

A wrong sentence is mended only when the model's fix changes a few words,
brings in no name, number or month its notes lack, and takes out no date
its notes write or the date pass worked out; nothing is dropped. Round 34
dropped 12 true sentences and 0 false ones on the tuning pages (some with
the model's own reasoning ending "so correct"), and 3 of its 5 wrong mends
took out a year or day the date pass had rightly worked out ("on Monday" in
a Monday article). A sentence it would have dropped, or whose fix is
refused, goes under ``report["when_who_flagged"]``; every mend under
``report["when_who"]``. Only sentences code finds a question about
(``suspect()``) are asked.
"""

import difflib
import re
from datetime import date
from typing import Any, Callable, Dict, List, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

WHEN_WHO = """You check two things in each numbered sentence of an \
encyclopedia entry about "{name}", against the notes it was written from. \
Each note starts with its article's date and weekday in brackets: the day the \
article appeared, not always the day of what it tells.

1. WHEN. For each date in the sentence (a year, a month, a day): do its notes \
date that thing to that time? A date written in the note's own text wins over \
its bracket date. An article's date is right for what the article reports as \
new or current ("said on Tuesday", "is on hunger strike", "this week") and for \
when something was said or reported; a day or year worked out from it ("on \
Monday" in a Monday article is that day; "In September" in an April article \
is the September before) is right. It is wrong for an older event, ruling, \
report or document the article only looks back on: "In April 2018, the \
Senate report found ..." from an April 2018 article about a 2014 report is \
wrong.
2. WHO. Is each act, decision or statement given to the person or body the \
notes give it to, and is it about this entry's subject and not a related \
person or case the notes describe? A judge's ruling given to a lawyer, or \
one detainee's transfer to another, is wrong.

Each sentence comes with a question that code found about its dates; answer \
it first. A date in the sentence that the notes' own text gives as another \
time for the same thing is wrong.

For each sentence: copy its first three words; say in a few words what you \
found, or "none"; then give "when" (right, wrong or no date) and "who" (right \
or wrong). If something is wrong, give the sentence mended with as few words \
changed as possible: the right date or actor from the notes, or the wrong \
date taken out when the notes give none. Leave "fixed" empty otherwise. Judge \
only by the notes shown."""

BATCH = 8
NOTE_CHARS = 1500
MAX_NOTES = 6
MAX_CHANGED = 8  # words a fix may add or remove


class WhenWho(BaseModel):
    model_config = ConfigDict(extra="ignore")
    sentence: int = Field(description="The sentence's number")
    starts_with: str = Field(description="Its first three words, copied")
    problem: str = Field(default="", description="What you found, or none")
    when: str = Field(description="right, wrong or no date")
    who: str = Field(description="right or wrong")
    fixed: str = Field(default="", description="The mended sentence, or empty")


class WhenWhoChecks(BaseModel):
    model_config = ConfigDict(extra="ignore")
    checks: List[WhenWho]


def _dated(note: Any) -> str:
    """ "2018-09-24, a Monday", or the stored date as it is."""
    raw = str(note.date or "")
    try:
        day = date.fromisoformat(raw[:10])
    except ValueError:
        return raw or "no date"
    return f"{raw[:10]}, a {day.strftime('%A')}"


def _messages(
    batch: Sequence[Any], by_n: Dict[int, Any], name: str
) -> List[Dict[str, str]]:
    parts = []
    for i, w in enumerate(batch, 1):
        shown = [by_n[n] for n in w.notes[:MAX_NOTES] if n in by_n]
        lines = "\n".join(f"  - [{_dated(n)}] {n.text[:NOTE_CHARS]}" for n in shown)
        ask = suspect(w, by_n)
        parts.append(f"Sentence {i}: {w.text}\nQuestion: {ask}\nIts notes:\n{lines}")
    return [
        {"role": "system", "content": WHEN_WHO.format(name=name)},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


_YEARS = re.compile(r"\b(?:19|20)\d\d\b")
_THAT_TIME = re.compile(r"^\W*That (?:month|year|day|week)\b")


_MONTHS = (
    "January February March April May June July August September October "
    "November December"
).split()
# "May 2015", "May 12, 2015", and "May and July 2003" as May 2003 and July 2003
_PAIRS = re.compile(
    r"\b(" + "|".join(_MONTHS) + r")\b\.?(?:\s+\d{1,2})?"
    r"((?:,?\s*(?:and|to|or|through|-|–)\s+(?:" + "|".join(_MONTHS) + r")\.?"
    r"(?:\s+\d{1,2})?)*),?\s+((?:19|20)\d\d)\b"
)


def month_years(text: str) -> set:
    """(month, year) pairs ``text`` names."""
    out = set()
    for m in _PAIRS.finditer(text):
        months = [m.group(1)] + re.findall("|".join(_MONTHS), m.group(2) or "")
        out |= {(mo, m.group(3)) for mo in months}
    return out


def _paired_wrong(w: Any, notes_text: str) -> List[str]:
    """ "May 2015" when the notes give May only with other years: round 34's
    "waterboarded him twice, in May 2015 and July 2003" against a note's
    "In June 2015, ... in May and July 2003" (2015 is in the note, for the
    disclosure, so a bare-year check passes it)."""
    noted = month_years(notes_text)
    out = []
    for mo, yr in sorted(month_years(w.text)):
        others = {y for m, y in noted if m == mo}
        if others and yr not in others:
            out.append(f"{mo} {yr}")
    return out


def _noted_years(text: str) -> set:
    """Years ``text`` gives, ranges written out ("1997-99" gives 1997, 1998
    and 1999: round 35 asked about "from 1997 to 1999" for a note's
    "1997-99")."""
    from src.engine.written_checks import _long_forms

    return set(_YEARS.findall(_long_forms(text)))


def _month_not_noted(w: Any, notes_text: str) -> List[str]:
    """ "July 2018" when the notes' text gives other months of 2018 but not
    July: on 23 Nov 27 of 38 false statements found on 30 written pages were
    dates, most an article's month put on something its note dates earlier
    ("In July 2018, Connell told a judge" from a July article whose note
    says May 2018); the year check passes it, since 2018 is in the note."""
    noted = month_years(notes_text)
    out = []
    for mo, yr in sorted(month_years(w.text)):
        if (mo, yr) not in noted and any(y == yr for _, y in noted):
            out.append(f"{mo} {yr}")
    return out


# "In July 2018, ..." or "On July 3, 2018, ...": a sentence opening on its date
_OPENER = re.compile(
    r"^(\W*(?:In|On)\s+)("
    + "|".join(_MONTHS)
    + r")\b\.?(\s+\d{1,2})?,?\s+((?:19|20)\d\d),"
)


def sources_of(w: Any, by_n: Dict[int, Any], orig: Dict[str, Any]) -> List[Any]:
    """The page's own notes behind ``w``'s cited notes: a condensed note
    joins several articles' notes, so its text holds several dates and its
    date is one article's (round 43: on the CIA page, condensed to 63 notes,
    "In August 2017, prosecutors argued" was not re-dated though its note
    says July 2017). Without ``orig``, the cited notes themselves."""
    out: List[Any] = []
    for n in w.notes:
        note = by_n.get(n)
        if note is None:
            continue
        keys = [k for k in (note.keys or [note.key]) if k in orig]
        out += [orig[k] for k in keys] if keys else [note]
    return out


def retime_openers(
    written: List[Any], by_n: Dict[int, Any], report: Dict[str, Any], name: str = "",
    orig: Optional[Dict[str, Any]] = None,
) -> None:  # fmt: skip
    """A sentence that opens on the month of a source note's article, which
    no source note's text gives, while that article's note gives exactly one
    month and year, opens on that one instead. Code, no call: the writer put
    the article's date on what the note itself dates."""
    log: List[List[str]] = report.setdefault("openers_retimed", [])
    for w in written:
        if w.dropped or w.summary:
            continue
        m = _OPENER.match(w.text)
        if not m:
            continue
        cited = sources_of(w, by_n, orig or {})
        mo, yr = m.group(2), m.group(4)
        month_no = f"{_MONTHS.index(mo) + 1:02d}"
        if (mo, yr) in month_years(" ".join(n.text for n in cited)):
            continue
        own = [n for n in cited if str(n.date or "")[:7] == f"{yr}-{month_no}"]
        text = " ".join(n.text for n in own)
        noted = month_years(text)
        if not own or len(noted) != 1:
            continue
        ((nmo, nyr),) = noted
        if not _dates_this(w.text[m.end() :], text, nmo, nyr, name):
            continue  # the note's month dates something else in it
        lead = re.match(r"^\W*", w.text).group(0)
        new = f"{lead}In {nmo} {nyr}," + w.text[m.end() :]
        log.append([w.text, new])
        w.text = new


def _dates_this(rest: str, notes_text: str, mo: str, yr: str, name: str) -> bool:
    """The words next to the note's "May 2018" (8 either side) share two
    content words, the entity's own aside, with the sentence's first eight
    after its date: the note's month dates what the sentence tells, not
    another thing the note names ("testified about the July 2002
    firefight")."""
    from src.engine.written_checks import _content

    own = _content(name)
    head = _content(" ".join(rest.split()[:8])) - own
    for m in _PAIRS.finditer(notes_text):
        if m.group(3) != yr or mo not in m.group(0):
            continue
        before = notes_text[: m.start()].split()[-8:]
        if not before or before[-1].lower() not in ("in", "on"):
            continue  # "the March 2002 raid" names a thing, it dates no act
        after = notes_text[m.end() :].split()[:8]
        if len(head & (_content(" ".join(before + after)) - own)) >= 2:
            return True
    return False


def suspect(w: Any, by_n: Dict[int, Any]) -> str:
    """The question code asks about ``w``'s dates, "" when it has none: a
    year its notes' text never gives while giving others ("May 2015" against
    a note's "May and July 2003"; "In June 2017, he was resettled" from a
    June 2017 article that says January 2016), or a "That month" opener.
    Only these sentences go to the model: asked of every sentence with
    "answer wrong only when sure", the check found 0 of 10 real errors in
    two runs over round 34's pages; asked without it, it dropped 12 true
    sentences."""
    text = " ".join(by_n[n].text for n in w.notes if n in by_n)
    paired = _paired_wrong(w, text)
    if paired:
        return (
            f"The sentence's {' and '.join(paired)}: its notes give that month only "
            "with another year. Which year do the notes give it?"
        )
    months = _month_not_noted(w, text)
    if months:
        given = ", ".join(sorted(f"{m} {y}" for m, y in month_years(text)))
        return (
            f"The sentence's {' and '.join(months)} is in no note's text, which "
            f"gives {given}. Is it the right time for what it dates, or the "
            "article's date put on something older?"
        )
    noted = _noted_years(text)
    odd = sorted(set(_YEARS.findall(w.text)) - noted)
    if odd and noted:
        return (
            f"The sentence's {' and '.join(odd)} is in no note's text, which gives "
            f"{', '.join(sorted(noted))}. Is {' and '.join(odd)} the right time for "
            "what it dates, or the article's date put on something older?"
        )
    if _THAT_TIME.match(w.text):
        return 'It opens with "That ...": which time do its notes give? Name it.'
    return ""


_DATE_WORD = re.compile(
    r"^(?:(?:19|20)\d\d|\d{1,2}|jan\w*|feb\w*|mar\w*|apr\w*|may|june?|july?|"
    r"aug\w*|sept?\w*|oct\w*|nov\w*|dec\w*|monday|tuesday|wednesday|thursday|"
    r"friday|saturday|sunday)$",
    re.I,
)
# Words that say who said or reported a thing: a mend that takes them out
# puts the article's date on the event (round 34: "In August 2012,
# prosecutors said the centre had closed in June 2003" lost "prosecutors said")
_SAYING = re.compile(
    r"^(?:said|says|say|told|stated|reported|according|announced|wrote)$", re.I
)


_SAYS_IN = re.compile(
    r"\b(?:said|says|reported|described|describing|told|wrote|testified|"
    r"announced|stated|according)\b",
    re.I,
)


def _words(text: str) -> List[str]:
    return re.findall(r"[\w’'$%-]+", text)


def usable_fix(
    old: str, fixed: str, sources: Sequence[str], name: str,
    page_names: Sequence[str], derived: Sequence[str] = (),
    wrong: Sequence[str] = (), article_years: Sequence[str] = (),
) -> bool:  # fmt: skip
    """``fixed`` changes at most ``MAX_CHANGED`` words of ``old``, keeps at
    least half of them, adds no name, number or month its notes lack, and
    takes out no saying word and no date word its notes write, the date
    pass added (``derived``) or that is its notes' article year when their
    text gives no other (``article_years``: round 35 took "January 2015" off
    a report made then). A year in ``wrong`` (paired with a month the notes
    give only with another year) may go."""
    from src.engine.written_checks import missing_details

    fixed = fixed.strip()
    if not fixed or fixed == old.strip():
        return False
    a, b = _words(old), _words(fixed)
    ops = difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes()
    changed = sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in ops if op != "equal")
    kept = sum(i2 - i1 for op, i1, i2, _, _ in ops if op == "equal")
    if changed > MAX_CHANGED or kept < len(a) / 2:
        return False
    hay = " ".join(sources).lower()
    keep = {d.lower() for d in derived} | set(article_years)
    free = set(wrong)
    for op, i1, i2, j1, j2 in ops:
        if op == "equal":
            continue
        for x in (x for x in a[i1:i2] if x not in b[j1:j2]):
            if _SAYING.match(x):
                return False
            low = x.lower()
            if low in free:
                continue
            if _DATE_WORD.match(x) and (
                low in keep or re.search(rf"\b{re.escape(low)}\b", hay)
            ):
                return False
    return not missing_details(fixed, sources, name, page_names)


def _name_that_time(w: Any, c: Any, by_n: Dict[int, Any]) -> str:
    """``w`` with "That month," made "In December 2016," when the model's
    answer names that month and year and one of its notes' articles is from
    then ("That year," likewise with a year); "" otherwise. Round 35's
    check named the month right and still answered "right", since "That
    month" is not wrong, only unnamed; so code makes the change."""
    m = _THAT_TIME.match(w.text)
    if not m:
        return ""
    unit = m.group(0).split()[-1].lower()
    said = month_years(f"{c.problem or ''} {c.fixed or ''}")
    dates = {str(by_n[n].date or "")[:7] for n in w.notes if n in by_n}
    text = " ".join(by_n[n].text for n in w.notes if n in by_n)
    if unit == "month":
        # a note whose text dates the act itself ("signed a Nov. 17
        # directive" in a January article) is not dated by its article:
        # round 35 made Trivett's November directive "In January 2018"
        own = {
            mo for mo in _MONTHS
            if re.search(rf"\b(?:{mo}|{mo[:3]}\.|Sept\.)" if mo == "September"
                         else rf"\b(?:{mo}|{mo[:3]}\.)(?!\w)", text)
        }  # fmt: skip
        hits = [
            (mo, yr) for mo, yr in said
            if f"{yr}-{_MONTHS.index(mo) + 1:02d}" in dates and (not own or mo in own)
        ]  # fmt: skip
        if len(set(hits)) != 1:
            return ""
        when = f"{hits[0][0]} {hits[0][1]}"
    elif unit == "year":
        years = set(_YEARS.findall(f"{c.problem or ''} {c.fixed or ''}"))
        hits = [y for y in years if any(d.startswith(y) for d in dates)]
        if len(hits) != 1:
            return ""
        when = hits[0]
    else:
        return ""
    return "In " + when + w.text[m.end() :]


def _first(text: str) -> str:
    from src.engine.written_checks import _first_words

    return _first_words(text, 2)


def when_who(
    written: List[Any], notes: Sequence[Any], name: str,
    ask: Callable[..., Any], report: Dict[str, Any], originals: Sequence[Any] = (),
) -> None:  # fmt: skip
    """Mend kept sentences the model finds misdated or given to the wrong
    actor, when the mend passes ``usable_fix()``; nothing is dropped, and a
    failed call leaves its sentences as they are."""
    from src.engine.written_checks import page_names_of

    by_n = {n.n: n for n in notes}
    names = page_names_of(notes)
    log: List[List[str]] = report.setdefault("when_who", [])
    flagged: List[List[str]] = report.setdefault("when_who_flagged", [])
    # the words the date pass added to each sentence it dated
    added: Dict[str, List[str]] = {}
    for before, after in report.get("own_dates") or []:
        added[after] = [x for x in _words(after) if x not in _words(before)]
    years = re.compile(r"\b(?:19|20)\d\d\b")
    # ``originals``: the page's notes before condensing, by key
    orig = {n.key: n for n in originals}
    retime_openers(written, by_n, report, name, orig)
    if report.get("retell", True):  # dated by a later article's retelling
        from src.engine.written_retell import redate_retold

        redate_retold(written, by_n, name, ask, report, orig)
    todo = [w for w in written if not w.dropped and not w.summary and suspect(w, by_n)]
    report["when_who_asked"] = report.get("when_who_asked", 0) + len(todo)
    for start in range(0, len(todo), BATCH):
        batch = todo[start : start + BATCH]
        try:
            reply = ask(_messages(batch, by_n, name), WhenWhoChecks, 4096)
        except Exception as exc:
            report.setdefault("errors", []).append(f"when_who: {str(exc)[:200]}")
            continue
        report["calls"] = report.get("calls", 0) + 1
        for c in reply.checks or []:
            if not 1 <= c.sentence <= len(batch):
                continue
            w = batch[c.sentence - 1]
            if _first(c.starts_with) != _first(w.text):
                continue  # the answer is about another sentence
            when = str(c.when or "").strip().lower()
            who = str(c.who or "").strip().lower()
            named = _name_that_time(w, c, by_n)
            if named:
                log.append([w.text, named, str(c.problem or "")[:200]])
                w.text = named
                continue
            if when != "wrong" and who != "wrong":
                continue
            problem = str(c.problem or "")[:200]
            sources = [by_n[n].text for n in w.notes if n in by_n]
            fix = str(c.fixed or "").strip()
            # a year the date pass added is kept only when the notes give no
            # other year: round 34 kept its own wrong "May 2015" against a
            # note's "May and July 2003"
            text = " ".join(sources)
            noted = _noted_years(text)
            wrong = {p.split()[-1] for p in _paired_wrong(w, text)}
            derived = [x for x in added.get(w.text, ()) if not (
                years.fullmatch(x) and noted and x not in noted)]  # fmt: skip
            arts = {str(by_n[n].date or "")[:4] for n in w.notes if n in by_n}
            # the article's year is right for what was said or reported then
            says = bool(_SAYS_IN.search(w.text))
            article_years = [y for y in arts if y and (says or noted <= {y})]
            if usable_fix(
                w.text, fix, sources, name, names, derived, wrong, article_years
            ):
                log.append([w.text, fix, problem])
                w.text = fix
            else:
                flagged.append([w.text, fix, problem])
