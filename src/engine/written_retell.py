"""A sentence dated by a later article that retells an act an earlier note on
the page already reported, dated by the earlier report.

Why: on 23 Nov a checker read 30 written pages against their notes and found
27 of 38 false statements were dates, most the article's month put on an
older act. Round 43 changed an opener only when its own notes date the act
otherwise, and on 14 pages that fit 1 sentence of about 100: the false ones
cite a note with no date in its text ("Rishikof struck down Baker's $1,000
fine", from a January 2018 article), while another note on the page, from
22 November 2017, reported the decision. Nothing in the sentence's own notes
shows it; only the page's other notes do.

By the stages rule: code finds each sentence that opens on a cited note's
article month while its notes' text gives no month, and an earlier note on
the page that shares a person and several words with that note; one call
per batch says whether the earlier note reports the very same act and when
it was; code re-dates the opener only to a month the earlier note's text
gives ("In November 2017,") or, when it gives none, to the earlier report's
month as a bound ("By November 2017,"), and cites the earlier note too.

Round 46: on 14 tuning pages (cp) the step was asked 8 times and re-dated
nothing. On Rishikof's "In January 2018, Rishikof struck down Baker's fine
...; in February, his office rejected a charge sheet", it took the one most
alike earlier note (Baker's suspended sentence, a different act) and missed
the 22 November decision; and it searched only from the first cited note of
the opener's month, so the charge sheet's note was never looked from. So it
now looks from every cited note of that month, shows up to ``MAX_EARLIER``
earlier notes, and the model names the one that reports the same act.

Round 47: with three earlier notes offered, the model named one as the same
act on 2 of 2 re-dates in cq and both were false ("By October 2016, Zaki
said ..." from a note on the debates; "By June 2017, a federal judge
imposed a restraining order" from a note on hearing schedules that only
shared Rishikof and Mattis), where with one note both of co's were right.
So one earlier note again (most alike over every cited note of the month),
which must share two words other than names with what the sentence tells,
and a sentence telling what someone said is never re-dated: a statement is
dated by when it was made. Every answer goes under ``retell_answers``.
"""

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field

RETELL = """You check dates in an encyclopedia entry about "{name}". Each \
numbered sentence below opens on the date of the news article its note \
comes from, though the note does not say when the act happened. An earlier \
note on the page may have reported the same act.

For each, say in a few words what act the sentence opens on and what act \
each earlier note tells; then answer:
- same: "yes" only if an earlier note reports the very same act the \
sentence opens on (the same ruling, filing, firing, statement, transfer), \
not a later step in the same story, a similar act, or a repeat of it; else \
"no".
- earlier: for "yes", the number of that earlier note (E1, E2, ...: give 1, \
2, ...).
- when: for "yes", the month and year the act happened by the notes \
("November 2017"), or "unknown"."""

BATCH = 8
MAX_EARLIER = 1


class Retold(BaseModel):
    model_config = ConfigDict(extra="ignore")
    sentence: int = Field(description="The sentence's number")
    acts: str = Field(default="", description="What act each tells, in a few words")
    same: str = Field(description='"yes" or "no"')
    earlier: int = Field(default=0, description="For yes: the earlier note's number")
    when: str = Field(default="", description='"Month YYYY" or "unknown"')


class RetoldChecks(BaseModel):
    model_config = ConfigDict(extra="ignore")
    checks: List[Retold]


def _alike(a: str, b: str, name: str) -> int:
    """Shared content words when the notes share a person and 3 words, or 4
    words and half the shorter one's; else 0."""
    from src.engine.written_checks import _content
    from src.engine.written_plan import _surnames

    wa, wb = _content(a), _content(b)
    shared = len(wa & wb)
    people = _surnames(a, name) & _surnames(b, name)
    if (people and shared >= 3) or (
        shared >= 4 and shared >= 0.5 * min(len(wa), len(wb))
    ):
        return shared
    return 0


def _plain_words(text: str, name: str) -> set:
    """Content words that are not names (capitalised) or the entity's own."""
    from src.engine.written_checks import _content

    caps = {w.lower() for w in re.findall(r"\b[A-Z][\w'’-]*", text)}
    return _content(text) - caps - _content(name)


def _tells_saying(rest: str) -> bool:
    """The sentence after its opener tells what someone said (a saying word
    in its first eight words)."""
    from src.engine.written_when_who import _SAYS_IN

    return bool(_SAYS_IN.search(" ".join(rest.split()[:8])))


def candidates(
    written: Sequence[Any], by_n: Dict[int, Any], name: str,
    orig: Optional[Dict[str, Any]] = None,
) -> List[Tuple[Any, Any, List[Any], List[Any]]]:  # fmt: skip
    """(sentence, opener match, its cited notes of the opener's month, up to
    ``MAX_EARLIER`` earlier notes, most alike first) for each kept sentence
    that opens on a cited note's article month, whose notes' text gives no
    month and year, and that an earlier-dated uncited note is alike with."""
    from src.engine.written_when_who import _MONTHS, _OPENER, month_years, sources_of

    orig = orig or {}
    # each page note, and the shown note that holds it (a citation must be one)
    pool = list(orig.values()) or list(by_n.values())
    shown_of = {k: n.n for n in by_n.values() for k in (n.keys or [n.key])}
    out = []
    for w in written:
        if w.dropped or w.summary:
            continue
        m = _OPENER.match(w.text)
        if not m or _tells_saying(w.text[m.end() :]):
            continue
        month = f"{m.group(4)}-{_MONTHS.index(m.group(2)) + 1:02d}"
        told = _plain_words(w.text[m.end() :], name)
        cited = sources_of(w, by_n, orig)
        own = [n for n in cited if str(n.date or "")[:7] == month]
        if not own or any(month_years(n.text) for n in own):
            continue
        mine = {n.key for n in cited}
        scored: List[Tuple[Tuple[int, int, int], Any]] = []
        for e in pool:
            if e.key in mine or not e.date or str(e.date)[:7] >= month:
                continue
            if shown_of.get(e.key, e.n if not orig else None) is None:
                continue
            score = max(_alike(n.text, e.text, name) for n in own)
            shared = len(told & _plain_words(e.text, name))
            if score and shared >= 2:
                # what the sentence tells first (round 47 offered a note on
                # a suspended sentence, not the one on striking the fine)
                day = -int(str(e.date)[:10].replace("-", "") or 0)
                scored.append(((shared, score, day), e))
        if scored:
            scored.sort(key=lambda x: x[0], reverse=True)
            out.append((w, m, own, [e for _, e in scored[:MAX_EARLIER]]))
    return out


def _messages(
    todo: Sequence[Tuple[Any, Any, List[Any], List[Any]]], name: str
) -> List[Dict[str, str]]:
    parts = []
    for i, (w, _, own, earlier) in enumerate(todo, 1):
        lines = [f"  Its note [{n.date}]: {n.text}" for n in own]
        lines += [f"  E{k} [{e.date}]: {e.text}" for k, e in enumerate(earlier, 1)]
        parts.append(f"Sentence {i}: {w.text}\n" + "\n".join(lines))
    return [
        {"role": "system", "content": RETELL.format(name=name)},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def redate_retold(
    written: List[Any], by_n: Dict[int, Any], name: str, ask: Any,
    report: Dict[str, Any], orig: Optional[Dict[str, Any]] = None,
) -> None:  # fmt: skip
    """Re-date each sentence the model finds retelling an earlier note's
    act; under ``report["retold"]`` as [old, new]."""
    from src.engine.written_when_who import _MONTHS, month_years

    todo = candidates(written, by_n, name, orig)
    shown_of = {k: n.n for n in by_n.values() for k in (n.keys or [n.key])}
    report["retell_asked"] = report.get("retell_asked", 0) + len(todo)
    log: List[List[str]] = report.setdefault("retold", [])
    answers: List[Dict[str, Any]] = report.setdefault("retell_answers", [])
    for start in range(0, len(todo), BATCH):
        batch = todo[start : start + BATCH]
        try:
            reply = ask(_messages(batch, name), RetoldChecks, 4096)
        except Exception as exc:
            report.setdefault("errors", []).append(f"retell: {str(exc)[:200]}")
            continue
        report["calls"] = report.get("calls", 0) + 1
        for c in reply.checks or []:
            if not 1 <= c.sentence <= len(batch):
                continue
            w, m, _, earlier = batch[c.sentence - 1]
            answers.append({
                "sentence": w.text, "earlier": [e.text for e in earlier],
                "acts": str(c.acts or "")[:300], "same": str(c.same),
                "picked": c.earlier, "when": str(c.when or ""),
            })  # fmt: skip
            if not str(c.same).strip().lower().startswith("y") or w.dropped:
                continue
            k = c.earlier or (1 if len(earlier) == 1 else 0)
            if not 1 <= k <= len(earlier):
                continue  # which earlier note is unsaid
            e = earlier[k - 1]
            said = re.search(
                r"\b(" + "|".join(_MONTHS) + r")\s+((?:19|20)\d\d)\b", str(c.when or "")
            )
            mo, yr = (said.group(1), said.group(2)) if said else ("", "")
            e_month = str(e.date)[:7]
            if said and (mo, yr) in month_years(e.text):
                head = f"In {mo} {yr},"  # the earlier note says when
            else:
                # its report's month bounds it: the act was by then
                y, mm = e_month.split("-")
                head = f"By {_MONTHS[int(mm) - 1]} {y},"
            lead = re.match(r"^\W*", w.text).group(0)
            new = f"{lead}{head}" + w.text[m.end() :]
            if new == w.text:
                continue
            log.append([w.text, new])
            w.text = new
            w.notes = list(dict.fromkeys(w.notes + [shown_of.get(e.key, e.n)]))
