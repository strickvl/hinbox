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


def section_score(
    sentences: Sequence[Written], planned: Sequence[int], notes: Sequence[Note],
    conflicts: Sequence[Dict[str, Any]], name: str,
) -> int:  # fmt: skip
    """Faults of one section's kept sentences, weighted: lower is better.
    3 for each sentence setting accounts against each other that tells no
    planned conflict; 2 for each pair sharing a year and a person or another
    number; 2 when the section opens on a sentence that leans on one before
    it; 2 for each sentence opening with its article's date on an event its
    notes date otherwise; 1 for each step back in date order; 1 for each
    planned note it does not use. A section with no kept sentence scores
    1000."""
    from src.engine.written_checks import NARRATES, _numbers, from_article_dates
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
        if _OPENS_DATED.match(w.text) and from_article_dates(w.text, cited):
            score += 2
    marks = [
        (
            {n for n in _numbers(w.text) if re.fullmatch(r"(?:19|20)\d\d", n)},
            {n for n in _numbers(w.text) if not re.fullmatch(r"(?:19|20)\d\d", n)},
            _surnames(w.text, name),
        )
        for w in kept
    ]
    for i in range(len(kept)):
        for j in range(i + 1, len(kept)):
            (ya, na, pa), (yb, nb, pb) = marks[i], marks[j]
            if ya & yb and ((pa & pb) or (na & nb)):
                score += 2
    if _LEANS.match(kept[0].text):
        score += 2
    dates = [d for d in (text_when(w.text) for w in kept) if d]
    score += sum(1 for a, b in zip(dates, dates[1:]) if b < a)
    used = {n for w in kept for n in w.notes}
    score += sum(1 for n in planned if n not in used)
    return score


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
    totals = [sum(sc[k] for sc in scores.values()) for k in range(len(drafts))]
    main = min(range(len(drafts)), key=lambda k: totals[k])
    out: List[Written] = list(split[main].get("", []))
    chosen: Dict[str, int] = {}
    for i, h in enumerate(planned):
        k = min(range(len(drafts)), key=lambda k: scores[h][k])
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
    report["draws"] = {
        "scores": scores, "totals": totals, "opening_from": main,
        "chosen": chosen,
    }  # fmt: skip
    return out


def write_drawn(
    plan: Any, shown: Sequence[Note], name: str, what: str, words: int, ask: Any,
    report: Dict[str, Any], check: bool, draws: int,
) -> List[Written]:  # fmt: skip
    """``draws`` drafts from one plan, finished each, then picked by section
    and checked once more for repeats across the sections it joined."""
    from src.engine.written_checks import drop_neighbours, drop_repeats, summary_check
    from src.engine.written_page import WBody, WPage, WParagraph, notes_block
    from src.engine.written_plan import write_from_plan
    from src.engine.written_revise import unwritten

    drafts: List[List[Written]] = []
    for _ in range(draws):
        page = write_from_plan(
            plan, shown, name, what, words, ask, report, notes_block,
            WPage, WBody, WParagraph,
        )  # fmt: skip
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
    summary_check(written, shown, name)
    report["unwritten"] = unwritten(written, shown, name, parts)
    return written
