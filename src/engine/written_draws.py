"""Writing a planned page more than once and keeping each section's best
draw, and the stages every draft goes through (``finish()``).

Why: over rounds 20 to 23 (21-22 Nov) the best draw of each page came from
a different run every time, and within one round the best and worst of
three draws of one page were far apart (Rishikof, the camp page, Jakarta).
What the model decides per draw (what the fill adds, what it calls a
dispute, how it trims) is chance; what code decides (the identity
sentence, the tracked facts, date order) held. Sections share only the
plan, so the plan, filing and conflict steps run once and the sections are
written ``draws`` times; each section is then taken from the draw whose
section scores best by code (``section_score()``), counting what a careful
reading of those rounds counted against a draw.
"""

import re
from typing import Any, Dict, List, Sequence

from src.engine.written_page import Note, Written


def finish(
    page: Any, shown: Sequence[Note], name: str, what: str, approach: str,
    check: bool, ask: Any, report: Dict[str, Any],
) -> List[Written]:  # fmt: skip
    """The page's sentences after every check, rewrite and fill stage."""
    from src.engine.written_checks import (
        code_check,
        drop_neighbours,
        drop_repeats,
        model_check,
        summary_check,
    )
    from src.engine.written_conflicts import (
        drop_one_sided,
        tell_conflicts,
        unnarrate,
    )
    from src.engine.written_order import date_sort, ensure_identity, keep_identity
    from src.engine.written_page import flatten, give_identity_a_note
    from src.engine.written_review import review_page
    from src.engine.written_revise import fill_unused, recite, revise, unwritten

    written = flatten(
        page, max((n.n for n in shown), default=0), summary=approach == "plan"
    )
    if approach == "plan":
        give_identity_a_note(written, report, shown)
    code_check([w for w in written if not w.summary], shown, name)
    if approach == "plan":
        recite(written, shown, name, (report.get("plan") or {}).get("parts") or [])
    drop_repeats(written)
    if check:
        model_check(written, shown, name, ask, report)
        report["identity_kept"] = keep_identity(
            written, shown, name, (report.get("plan") or {}).get("identity") or ""
        )
    summary_check(written, shown, name)
    if check:
        written = revise(written, shown, name, ask, report)
        summary_check(written, shown, name)
        report["revised_kept"] = sum(1 for w in written if w.revised and not w.dropped)
        if approach == "plan":
            parts = (report.get("plan") or {}).get("parts") or []
            written = fill_unused(written, shown, name, what, parts, ask, report)
            # Filed is not written: on 21 Nov the fill stage wrote three
            # sentences for Zaki's first section and stopped, leaving the
            # artwork ban, the ISIS remark and Slahi unwritten. One more pass
            # for what is still unused, then the rest is reported.
            if len(shown) <= 40:  # on long pages the second pass added repeats
                written = fill_unused(written, shown, name, what, parts, ask, report)
            report["unwritten"] = unwritten(written, shown, name, parts)
            conflicts = (report.get("plan") or {}).get("conflicts") or []
            written = tell_conflicts(written, shown, name, what, conflicts, ask, report)
            written = unnarrate(written, shown, name, conflicts, ask, report)
            drop_one_sided(written, conflicts)
            written = review_page(written, shown, name, what, ask, report)
            plan = report.get("plan") or {}
            report["identity_kept"] = report.get("identity_kept") or keep_identity(
                written, shown, name, plan.get("identity") or ""
            )
            report["identity_added"] = ensure_identity(
                written, shown, name, plan.get("identity") or "",
                plan.get("opening_notes") or [],
            )  # fmt: skip
            report["date_sorted"] = date_sort(written, shown)
            drop_repeats(written)
            report["neighbours_dropped"] = drop_neighbours(written)
            summary_check(written, shown, name)
    return written


# A section or sentence that leans on something before it: "He was held
# ...", "The odor from this behavior ...", "The document also ...", "In the
# agreement, ..." (rounds 21-23)
_LEANS = re.compile(
    r"^\W*(?:He|She|They|It|His|Her|Their|Its|This|These|That|Those|"
    r"The \w+ (?:also|then)|In the (?:agreement|deal|filing|ruling|document)|"
    r"(?:The|An?) \w+ from (?:this|the) behaviou?r)\b"
)
_OPENS_DATED = re.compile(
    r"^\W*(?:In|By|On|As of)\s+[A-Z][a-z]+\.?\s+(?:\d{1,2},\s+)?\d{4}\b"
)


_YEAR = re.compile(r"(?:19|20)\d\d")
# A sentence naming what came before it: "That month, Pohl ordered ...",
# "The document also instructed ...", "At the time, the facility held ..."
# (round 25: four wrong picks had one with nothing before it to lean on)
_THAT_TIME = re.compile(r"^\W*(?:That (?:month|year|day|week)|At the time|Then)\b")
_THE_ALSO = re.compile(r"^\W*The (\w+) (?:also|further|then)\b")
_PRONOUN = re.compile(r"^\W*(?:He|She|They|His|Her|Their)\b")


def _leans_on_nothing(before: Written, w: Written, name: str) -> bool:
    """``w`` leans on something ``before`` does not give: a time it does not
    name, the thing it does not mention, a person it does not name."""
    from src.engine.written_groups import text_when
    from src.engine.written_plan import _surnames

    if _THAT_TIME.match(w.text):
        return text_when(before.text) is None
    m = _THE_ALSO.match(w.text)
    if m:
        return m.group(1).lower() not in before.text.lower()
    if _PRONOUN.match(w.text):
        surname = name.split()[-1].lower() if name.split() else ""
        return not _surnames(before.text, name) and surname not in before.text.lower()
    return False


def section_score(
    sentences: Sequence[Written], planned: Sequence[int], notes: Sequence[Note],
    conflicts: Sequence[Dict[str, Any]], name: str,
) -> int:  # fmt: skip
    """Faults of one section's kept sentences, weighted: lower is better.

    - 3 for each sentence setting accounts against each other that tells no
      planned conflict;
    - 2 for each pair of sentences sharing a person and a number that is not
      a year, or 60% of the shorter one's words (round 24's CIA section gave
      the 183 waterboardings four times and scored 2);
    - 2 when the section opens on a sentence that leans on one before it;
    - 2 for each sentence opening with its article's date when its notes give
      the event another year (round 24 counted every article date, and on
      Zaki's page, where each sentence is a remark to a reporter, picked the
      drafts that had dropped their dates);
    - 1 for each step back in date order;
    - 1 for each planned note it does not cite, 2 more when no kept sentence
      of the section has that note's people or numbers (content lost);
    - 1 for each year its notes give that no kept sentence gives.

    A section with no kept sentence scores 1000."""
    from src.engine.written_checks import (
        NARRATES,
        _content,
        _numbers,
        from_article_dates,
    )
    from src.engine.written_conflicts import _DIFFER, tells
    from src.engine.written_groups import text_when
    from src.engine.written_plan import _surnames

    kept = [w for w in sentences if not w.dropped]
    if not kept:
        return 1000
    by_n = {n.n: n for n in notes}
    score = 0
    for w in kept:
        if (NARRATES.search(w.text) or _DIFFER.search(w.text)) and not any(
            tells(w, c) for c in conflicts
        ):
            score += 3
        cited = [by_n[n] for n in w.notes if n in by_n]
        dated = from_article_dates(w.text, cited) if _OPENS_DATED.match(w.text) else []
        if dated and any(set(_YEAR.findall(n.text)) - set(dated) for n in cited):
            score += 2
    facts = [
        (
            _content(w.text),
            {n for n in _numbers(w.text) if not _YEAR.fullmatch(n)},
            _surnames(w.text, name),
        )
        for w in kept
    ]
    for i in range(len(kept)):
        for j in range(i + 1, len(kept)):
            (wa, na, pa), (wb, nb, pb) = facts[i], facts[j]
            small = min(len(wa), len(wb)) or 1
            if ((pa & pb) and (na & nb)) or len(wa & wb) >= 0.6 * small:
                score += 2
    if _LEANS.match(kept[0].text):
        score += 2
    score += 2 * sum(1 for a, b in zip(kept, kept[1:]) if _leans_on_nothing(a, b, name))
    dates = [d for d in (text_when(w.text) for w in kept) if d]
    score += sum(1 for a, b in zip(dates, dates[1:]) if b < a)
    used = {n for w in kept for n in w.notes}
    said = set().union(*(f[1] | f[2] for f in facts))
    years_said = set().union(*(set(_YEAR.findall(w.text)) for w in kept))
    years_noted: set = set()
    for n in planned:
        if n not in by_n:
            continue
        years_noted |= set(_YEAR.findall(by_n[n].text))
        if n in used:
            continue
        score += 1
        text = by_n[n].text
        marks = {x for x in _numbers(text) if not _YEAR.fullmatch(x)} | _surnames(
            text, name
        )
        if marks and not marks & said:
            score += 2
    score += len(years_noted - years_said)
    return score


def _marks_of(ws: Sequence[Written], name: str) -> set:
    from src.engine.written_checks import _numbers
    from src.engine.written_plan import _surnames

    out: set = set()
    for w in ws:
        if not w.dropped:
            out |= _numbers(w.text) | _surnames(w.text, name)
    return out


def _lost_against(
    split: Sequence[Dict[str, List[Written]]], h: str, name: str
) -> List[int]:
    """2 for each date, number or name of section ``h`` that two or more
    other drafts have and this draft lacks (round 25 picked sections that
    had lost April 3, October 2018 or Butler, which the others kept)."""
    marks = [_marks_of(s.get(h, []), name) for s in split]
    out = []
    for k, mine in enumerate(marks):
        others = [m for j, m in enumerate(marks) if j != k]
        common = {x for x in set().union(*others) if sum(x in m for m in others) >= 2}
        out.append(2 * len(common - mine))
    return out


def _near_repeats(ws: Sequence[Written]) -> int:
    """Pairs of kept sentences sharing 40% of the shorter one's words."""
    from src.engine.written_checks import _content

    words = [_content(w.text) for w in ws if not w.dropped]
    return sum(
        1
        for i in range(len(words))
        for j in range(i + 1, len(words))
        if len(words[i] & words[j]) >= 0.4 * (min(len(words[i]), len(words[j])) or 1)
    )


def _words(ws: Sequence[Written]) -> int:
    return sum(len(w.text.split()) for w in ws if not w.dropped)


def _sections(written: Sequence[Written]) -> Dict[str, List[Written]]:
    out: Dict[str, List[Written]] = {}
    for w in written:
        out.setdefault(w.section, []).append(w)
    return out


def pick_sections(
    drafts: Sequence[List[Written]], parts: Sequence[Dict[str, Any]],
    notes: Sequence[Note], conflicts: Sequence[Dict[str, Any]], name: str,
    report: Dict[str, Any],
) -> List[Written]:  # fmt: skip
    """One page from ``drafts``: the opening of the draft whose sections
    score best in all, then each planned section from its best draft (ties
    go to the earlier draft), then any section the fill made in one draft
    whose notes no chosen section cites."""
    split = [_sections(d) for d in drafts]
    planned = {p.get("heading") or "": p.get("notes") or [] for p in parts}
    scores: Dict[str, List[int]] = {
        h: [section_score(s.get(h, []), ns, notes, conflicts, name) for s in split]
        for h, ns in planned.items()
    }
    for h in planned:  # what the other drafts have and this one lost
        for k, extra in enumerate(_lost_against(split, h, name)):
            scores[h][k] += extra
    totals = [sum(sc[k] for sc in scores.values()) for k in range(len(drafts))]
    main = min(range(len(drafts)), key=lambda k: totals[k])
    out: List[Written] = list(split[main].get("", []))
    chosen: Dict[str, int] = {}
    for i, h in enumerate(planned):
        # Ties go to fewer near-repeats, then the shorter section: in round
        # 25 seven of 13 wrong picks were ties won by the longer draft, which
        # was usually the one saying something twice
        k = min(
            range(len(drafts)),
            key=lambda k: (
                scores[h][k],
                _near_repeats(split[k].get(h, [])),
                _words(split[k].get(h, [])),
            ),
        )
        chosen[h] = k
        for w in split[k].get(h, []):
            w.para += (i + 1) * 1_000_000  # paragraphs of two drafts never merge
            out.append(w)
    cited = {n for w in out if not w.dropped and w.section for n in w.notes}
    extra = 0
    for k, s in enumerate(split):
        for h, ws in s.items():
            if h in planned or h in chosen or not h:
                continue
            ns = {n for w in ws if not w.dropped for n in w.notes}
            if ns and not ns <= cited:
                chosen[h] = k
                extra += 1
                for w in ws:
                    w.para += (len(planned) + extra) * 1_000_000
                out.extend(ws)
                cited |= ns
    # Every draft's kept sentences, so a pick can be checked by hand
    drafts_text = {
        h or "(opening)": [
            [w.text for w in s.get(h, []) if not w.dropped] for s in split
        ]
        for h in [""] + list(planned)
    }
    report["draws"] = {
        "scores": scores, "totals": totals, "opening_from": main,
        "chosen": chosen, "drafts": drafts_text,
    }  # fmt: skip
    return out


def write_drawn(
    plan: Any, shown: Sequence[Note], name: str, what: str, words: int, ask: Any,
    report: Dict[str, Any], check: bool, draws: int, local: bool = False,
) -> List[Written]:  # fmt: skip
    """``draws`` drafts from one plan, finished each, then picked by section
    and checked once more for repeats across the sections it joined."""
    from src.engine.written_checks import drop_neighbours, drop_repeats, summary_check
    from src.engine.written_page import WBody, WPage, WParagraph, notes_block
    from src.engine.written_plan import write_from_plan
    from src.engine.written_revise import unwritten
    from src.utils.llm import set_model_params

    drafts: List[List[Written]] = []
    for k in range(draws):
        # Local Qwen at temperature 0 wrote the same three drafts (round 24),
        # so its later drafts' sections are sampled; cloud drafts differ anyway
        if local and k:
            set_model_params({"temperature": 0.7, "seed": k})
        try:
            page = write_from_plan(
                plan, shown, name, what, words, ask, report, notes_block,
                WPage, WBody, WParagraph,
            )  # fmt: skip
        finally:
            if local and k:
                set_model_params(None)
        if page is not None:
            drafts.append(finish(page, shown, name, what, "plan", check, ask, report))
    if not drafts:
        return []
    meta = report.get("plan") or {}
    parts = meta.get("parts") or []
    written = pick_sections(
        drafts, parts, shown, meta.get("conflicts") or [], name, report
    )
    drop_repeats(written)
    drop_neighbours(written)
    report["draws"]["across"] = drop_across(written, name)
    summary_check(written, shown, name)
    report["unwritten"] = unwritten(written, shown, name, parts)
    return written


def drop_across(written: List[Written], name: str) -> int:
    """Drop a kept sentence when an earlier kept sentence of another section
    (or the opening) has every number and person it names and 60% of its
    words; the number dropped. Joining sections from different drafts let
    one fact be told in two (round 24: Rishikof's firing dispute, Karachi's
    Pearl confession in the opening and the body)."""
    from src.engine.written_checks import _content, _numbers
    from src.engine.written_plan import _surnames

    kept = [w for w in written if not w.dropped and not w.summary]
    dropped = 0
    for j, b in enumerate(kept):
        if not b.section:
            continue
        wb, mb = _content(b.text), _numbers(b.text) | _surnames(b.text, name)
        for a in kept[:j]:
            if a.dropped or a.section == b.section:
                continue
            if mb <= _numbers(a.text) | _surnames(a.text, name) and len(
                wb & _content(a.text)
            ) >= 0.6 * max(len(wb), 1):
                b.dropped = "told in another section"
                dropped += 1
                break
    return dropped
