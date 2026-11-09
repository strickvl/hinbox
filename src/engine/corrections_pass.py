"""Carry out a historian's corrections in each run (step 2 of the design).

The reader (``src/frontend/``) only adds lines to the corrections file
(``src/utils/corrections.py``) and draws them on top of the page. This module
makes them hold in the stored profiles, so the tables, the grounding check,
the merge check and every export see corrected text:

- **After merging, before grounding** (``run_corrections_pass()``, called from
  ``main()``): each struck sentence still in its profile is cut out, and the
  historian's own sentence, if they wrote one, goes in its place with its
  citation. A sentence that restates a struck one (every word of it in the
  struck sentence, ``is_repeat()``) is cut too. A struck sentence whose
  replacement has gone missing is put back. A withdrawn strike that a run had
  carried out is undone where it can be (the historian's sentence swapped back
  for the original). Every correction gets a state, written to a report beside
  the corrections file: applied (first carried out now), held (nothing tried to
  undo it), re-applied (something did, and the correction won), undone,
  waiting (a kind no run carries out yet) or needs a look (with why).
- **During a profile update** (``guard_corrections()``, called by
  ``edit_profile()`` inside ``correction_scope()``): an ``add`` sentence that
  restates a struck one is cut from the edit, and a ``replace`` aimed at a
  historian's sentence is dropped, so the edit model can't reword it.

Checker verdicts the historian overruled are not written anywhere: the
checker still judges the sentence each run, and the reader shows the
historian's verdict over it. The pass only reports whether the sentence is
still there with the same words (held) or not (needs a look).

Nothing here calls a model, so a ``--local`` run stays offline.
"""

import contextvars
import json
import os
import re
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

from src.engine.edit_guards import is_repeat
from src.utils.claim_spans import claim_end, parse_claims
from src.utils.corrections import (
    DONE,
    NEEDS_RUN,
    REPORT_LATEST,
    CorrectionSet,
    _report_dir,
    last_states,
    load_corrections,
    md_literal,
    md_plain,
    resolve_ref,
)
from src.utils.corrections import sentence_key as _raw_key
from src.utils.entity_keys import KEY_FUNCS, NAME_FIELDS

VERSION_TRIGGER = "corrections:"  # a profile version made by this pass


def sentence_key(text: str) -> str:
    """``sentence_key()`` of the words as written: a historian's sentence is
    stored with Markdown escaped (``md_literal()``), so both sides of every
    comparison drop the escapes first."""
    return _raw_key(md_plain(text))


_ACTIVE: Optional[CorrectionSet] = None
_CURRENT: contextvars.ContextVar = contextvars.ContextVar("corrections", default=None)
# Strikes the edit guard had to defend during this run: id -> why
_GUARDED: Dict[str, str] = {}
_LOCK = threading.Lock()


@dataclass
class PageRules:
    """One page's active corrections, by what they protect."""

    struck: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # key -> line
    written: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # key -> line
    verdicts: List[Dict[str, Any]] = field(default_factory=list)
    waiting: List[Dict[str, Any]] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.struck or self.verdicts or self.waiting)


def _rules(lines: List[Dict[str, Any]]) -> PageRules:
    rules = PageRules()
    for r in lines:
        if r["kind"] == "strike":
            rules.struck[sentence_key(r["sentence"])] = r
            if r.get("replacement"):
                rules.written[sentence_key(r["replacement"])] = r
        elif r["kind"] == "verdict":
            rules.verdicts.append(r)
        elif r["kind"] in NEEDS_RUN:
            rules.waiting.append(r)
    return rules


def _linked(entity: Dict[str, Any]) -> List[str]:
    return [
        a["article_id"]
        for a in entity.get("articles") or []
        if isinstance(a, dict) and a.get("article_id")
    ]


def _candidates(table: str, pages: Dict[Any, Dict[str, Any]]) -> Iterator[Tuple]:
    key, name = KEY_FUNCS[table], NAME_FIELDS[table]
    for pk, ent in pages.items():
        yield key(ent), ent.get(name) or "", _linked(ent), pk


# ── during the run: the edit guard ───────────────────────────


def configure_corrections(cs: Optional[CorrectionSet]) -> None:
    """Load the corrections the edit guard checks during this run."""
    global _ACTIVE
    _ACTIVE = cs if cs and cs.active else None
    with _LOCK:
        _GUARDED.clear()


def _names(ref: Any, table: str, entity: Dict[str, Any]) -> bool:
    if not isinstance(ref, dict) or ref.get("table") != table:
        return False
    if ref.get("key") == KEY_FUNCS[table](entity):
        return True
    return bool(ref.get("name")) and ref.get("name") == entity.get(NAME_FIELDS[table])


@contextmanager
def correction_scope(table: str, entity: Dict[str, Any]) -> Iterator[None]:
    """While a profile update runs, the guard knows this page's strikes. A
    page is matched by key or by name; two pages sharing a name both get the
    other's strikes guarded, which can only keep a struck sentence out."""
    lines = [
        r
        for r in (_ACTIVE.active if _ACTIVE else [])
        if _names(r.get("entity"), table, entity)
    ]
    token = _CURRENT.set(_rules(lines) if lines else None)
    try:
        yield
    finally:
        _CURRENT.reset(token)


def current_rules() -> Optional[PageRules]:
    return _CURRENT.get()


def _restated(text: str, rules: PageRules) -> Optional[Dict[str, Any]]:
    """The strike this claim says again: the same words, or the same words
    with filler round them ("The article states that …"), each way round
    (``is_repeat()`` both ways). One way only is not enough: "He was born in
    Taiz in 1975" has every word in the struck "He was born in Taiz in 1975
    and studied engineering in Karachi", and may be the true half of it."""
    key = sentence_key(text)
    if key in rules.struck:
        return rules.struck[key]
    if key in rules.written:
        return None
    for rec in rules.struck.values():
        if is_repeat(text, [rec["sentence"]]) and is_repeat(rec["sentence"], [text]):
            return rec
    return None


def _cut_restated(text: str, rules: PageRules) -> Tuple[str, List[Tuple[str, str]]]:
    """``text`` without the cited claims that restate a struck sentence, and
    (correction id, claim) for each one cut."""
    claims, markers = parse_claims(text)
    cuts: List[Tuple[Any, str]] = []
    for c in claims:
        rec = _restated(c.text, rules)
        if rec:
            cuts.append((c, rec["id"]))
    for c, _ in reversed(cuts):
        text = _cut_claim(text, markers, c)
    return text, [(i, c.text) for c, i in cuts]


def _note(cid: str, why: str) -> None:
    with _LOCK:
        _GUARDED.setdefault(cid, why)


def guard_corrections(text: str, edits: List[Any]) -> Tuple[List[Any], Dict[str, Any]]:
    """The edits with a struck sentence's restatements cut out, and with any
    ``replace`` of a historian's sentence dropped. Uses the page's rules from
    ``correction_scope()``; outside one, the edits pass unchanged."""
    from src.engine.profile_edits import _find

    rules = current_rules()
    report: Dict[str, Any] = {"restated": [], "kept_yours": [], "moved_after_yours": []}
    if not rules or not rules.struck:
        return edits, report
    # Where each of the historian's own sentences sits in the text
    claims, markers = parse_claims(text)
    yours = [
        (rng, rules.written[sentence_key(c.text)], c)
        for c in claims
        if sentence_key(c.text) in rules.written
        for rng in _claim_ranges(text, markers, c)
    ]
    kept = []
    for edit in edits:
        span = _find(text, edit.anchor.strip()) if edit.anchor.strip() else None
        # Any overlap: an anchor on one sentence of a two-sentence
        # replacement would otherwise rewrite half of it
        hit = next(
            ((r, c) for (a, b), r, c in yours if span and span[0] < b and a < span[1]),
            None,
        )
        if hit and edit.op == "replace":
            report["kept_yours"].append(edit.anchor[:200])
            _note(hit[0]["id"], "the edit model tried to reword your sentence")
            continue
        if hit and span[1] < hit[1].end:
            # An add inside their sentence would split it: it goes after it
            edit = edit.model_copy(update={"anchor": text[hit[1].start : hit[1].end]})
            report["moved_after_yours"].append(edit.text[:200])
        new_text, cut = _cut_restated(edit.text, rules)
        for cid, claim in cut:
            report["restated"].append(claim[:200])
            _note(cid, f"an update wrote it again: “{claim[:160]}”")
        if cut and not re.search(r"\w", re.sub(r"\^\[[^\]]*\]", "", new_text)):
            continue
        kept.append(edit.model_copy(update={"text": new_text}) if cut else edit)
    return kept, report


# ── after merging: the pass over every page ──────────────────


_EMPTY_LINE = re.compile(r"[ \t]*(?:[-*+]|\d+[.)])?[ \t]*")
_HEADING_LINE = re.compile(r"[ \t]*#")
_LIST_MARK = re.compile(r"[ \t]*(?:[-*+]|\d+[.)])[ \t]+|[ \t]+")


def _remove(text: str, start: int, end: int, put: str = "") -> str:
    """``text`` with ``[start, end)`` replaced by ``put``, without leaving a
    double space, an empty list item or a run of blank lines there. Only the
    text at the cut changes, never anything before the line it is on, so
    cuts found in one pass can be made back to front."""
    before, after = text[:start], text[end:]
    if put:
        return before + put + after
    if before.endswith((" ", "\t")) and after[:1] in (" ", "\t", "\n", ""):
        before = before.rstrip(" \t")
    elif after[:1] in (" ", "\t") and (not before or before.endswith("\n")):
        after = after.lstrip(" \t")
    line_start = before.rfind("\n") + 1
    line_end = after.find("\n")
    line_end = len(after) if line_end < 0 else line_end
    if _EMPTY_LINE.fullmatch(before[line_start:] + after[:line_end]):
        before, after = before[:line_start], after[line_end + 1 :]
        # the blank lines either side of a paragraph that is gone: keep one
        if before.endswith("\n\n") and after.startswith("\n"):
            after = after.lstrip("\n")
        elif not before:
            after = after.lstrip("\n")
    return before + after


def _claim_ranges(text: str, markers: List[Any], c: Any) -> List[Tuple[int, int]]:
    """Where a claim's words are, to cut them all. The claim runs from the
    previous claim's last marker to its own last marker, as grounding reads
    it; ``c.start`` is only where its last block starts (the reader keeps its
    highlight inside one paragraph), so an uncited paragraph before it is a
    range of its own. Heading lines in between are not the claim's, and stay."""
    prev = [m.end for m in markers if m.claim == c.index - 1]
    lead = max(prev) if prev else 0
    ranges: List[Tuple[int, int]] = []
    pos = lead
    for line in text[lead : c.start].splitlines(keepends=True):
        body = line.rstrip("\n")
        if body.strip() and not _HEADING_LINE.match(body):
            skip = _LIST_MARK.match(body) if pos > lead or not prev else None
            first = pos + (skip.end() if skip else 0)
            if pos == lead and prev:  # the rest of the line after the marker
                first = pos + len(body) - len(body.lstrip(" \t.,;:!?)]"))
            if re.search(r"\w", text[first : pos + len(body)]):
                ranges.append((first, pos + len(body)))
        pos += len(line)
    return ranges + [(c.start, claim_end(markers, c.index))]


def _cut_claim(text: str, markers: List[Any], c: Any, put: str = "") -> str:
    """``text`` without claim ``c``'s words, and ``put`` where its last block was."""
    ranges = _claim_ranges(text, markers, c)
    for n, (start, end) in enumerate(reversed(ranges)):
        text = _remove(text, start, end, put if n == 0 else "")
    return text


def _yours(rec: Dict[str, Any]) -> str:
    words = md_literal(" ".join(rec["replacement"].split()))
    return f"{words}^[{rec['replacement_cites']}]"


def _original(rec: Dict[str, Any]) -> str:
    cites = rec.get("cites") or []
    return rec["sentence"].strip() + (f"^[{', '.join(cites)}]" if cites else "")


def _swap_back(text: str, rec: Dict[str, Any]) -> Tuple[str, bool]:
    """``text`` with an earlier strike's replacement swapped back for the
    original sentence, if the replacement is there and the original isn't."""
    claims, markers = parse_claims(text)
    by_key = {sentence_key(c.text): c for c in claims}
    rep = by_key.get(sentence_key(rec.get("replacement") or ""))
    if rep is None or sentence_key(rec["sentence"]) in by_key:
        return text, False
    return _cut_claim(text, markers, rep, _original(rec)), True


def apply_to_text(
    text: str,
    rules: PageRules,
    done_before: Set[str],
    withdrawn: List[Dict[str, Any]],
    superseded: Sequence[Dict[str, Any]] = (),
) -> Tuple[str, Dict[str, Tuple[str, str]], List[Dict[str, str]]]:
    """Carry one page's corrections out in its profile text. Returns the new
    text, (state, why) per correction id, and claims that may say a struck
    thing again in other words (sharing a year and a name with it).

    ``superseded`` are older strikes a run carried out that a newer strike
    of the same sentence replaced: their sentence is first swapped back for
    the original, so the newer strike is carried out on it like any other
    (2000 → 2001 by the old one, then 2001 → original → 2002)."""
    states: Dict[str, Tuple[str, str]] = {}
    for old in superseded:
        if sentence_key(old.get("replacement") or "") not in rules.written:
            text, _ = _swap_back(text, old)
    claims, markers = parse_claims(text)
    keys = {sentence_key(c.text) for c in claims}
    present = {k for k in rules.written if k in keys}
    edits: List[Tuple[Any, str]] = []
    for c in claims:
        key = sentence_key(c.text)
        rec = _restated(c.text, rules)
        if not rec:
            continue
        put = _yours(rec) if rec.get("replacement") else ""
        rep_key = sentence_key(rec.get("replacement") or "")
        if put and rep_key in present:
            put = ""  # their sentence is already in the profile
        elif put:
            present.add(rep_key)
        edits.append((c, put))
        if rec["id"] not in done_before:
            states.setdefault(rec["id"], ("applied", ""))
        elif key == sentence_key(rec["sentence"]):
            states[rec["id"]] = ("re-applied", "the struck sentence was back")
        else:
            states[rec["id"]] = ("re-applied", f"said again: “{c.text[:160]}”")
    for c, put in reversed(edits):  # back to front, so offsets hold
        text = _cut_claim(text, markers, c, put)
    for rec in {r["id"]: r for r in rules.struck.values()}.values():
        cid, rep = rec["id"], sentence_key(rec.get("replacement") or "")
        if cid in states:
            continue
        if rep and rep not in present and cid in done_before:
            # its own paragraph: after a heading left empty it would
            # otherwise become part of the heading line
            text = text.rstrip() + "\n\n" + _yours(rec) + "\n"
            present.add(rep)
            states[cid] = (
                "re-applied",
                "your sentence was missing; put back at the end of the profile",
            )
        elif cid in done_before or (rep and rep in present):
            states[cid] = ("held", "")
        else:
            states[cid] = (
                "needs a look",
                "the struck sentence is not in the profile (reworded by an earlier run?)",
            )
    claims, markers = parse_claims(text)
    by_key = {sentence_key(c.text): c for c in claims}
    for rec in withdrawn:
        text, swapped = _swap_back(text, rec)
        if swapped:
            claims, markers = parse_claims(text)
            by_key = {sentence_key(c.text): c for c in claims}
            states[rec["id"]] = (
                "undone",
                "your sentence swapped back for the original",
            )
        elif sentence_key(rec["sentence"]) not in by_key:
            states[rec["id"]] = (
                "needs a look",
                "withdrawn after a run cut the sentence; it is not put back "
                "(its words are in the correction line)",
            )
        else:
            states[rec["id"]] = ("undone", "")
    for rec in rules.verdicts:
        c = by_key.get(sentence_key(rec["sentence"]))
        if c is not None and rec["article_id"] in c.articles:
            states[rec["id"]] = ("held", "")
        else:
            states[rec["id"]] = (
                "needs a look",
                "the sentence's words changed, so your verdict no longer applies",
            )
    for rec in rules.waiting:
        states[rec["id"]] = ("waiting", "runs don't carry out this kind yet")
    return text, states, _maybe_again(claims, rules)


_YEAR = re.compile(r"\b(?:1[89]|20)\d\d\b")
_NAME = re.compile(r"\b[A-Z][a-z]{2,}\b")


def _maybe_again(claims: List[Any], rules: PageRules) -> List[Dict[str, str]]:
    """Claims sharing a year and a capitalised word (past the first) with a
    struck sentence: possibly the same claim in other words, for a person to
    check. Nothing is changed for them."""
    out = []
    for rec in {r["id"]: r for r in rules.struck.values()}.values():
        years = set(_YEAR.findall(rec["sentence"]))
        names = set(_NAME.findall(rec["sentence"][1:]))
        for c in claims:
            if sentence_key(c.text) in rules.written:
                continue
            if years & set(_YEAR.findall(c.text)) and names & set(
                _NAME.findall(c.text[1:])
            ):
                out.append({"id": rec["id"], "sentence": c.text[:300]})
    return out


def _new_version(entity: Dict[str, Any], ids: List[str]) -> None:
    """Record the corrected text as a profile version, so the page's history
    shows what the corrections changed."""
    from src.engine.profiles import VersionedProfile

    pv = entity.get("profile_versions")
    if not pv:
        return
    versioned = VersionedProfile(**pv)
    versioned.add_version(
        entity["profile"], trigger_article_id=VERSION_TRIGGER + ",".join(ids)
    )
    entity["profile_versions"] = versioned.model_dump()


def run_corrections_pass(
    entities: Dict[str, Dict[Any, Dict[str, Any]]],
    path: str,
    output_dir: str = "",
    write: bool = True,
) -> Optional[Dict[str, Any]]:
    """Carry out the corrections file's strikes in every profile and return
    the report (None with no file). With ``write``, the report is written
    beside the file at once; a run passes False and writes it with
    ``write_report()`` once the tables are saved, so the report never says a
    strike was carried out in tables that were not written."""
    if not os.path.exists(path):
        return None
    cs = load_corrections(path)
    before = last_states(path)
    done_before = {i for i, s in before.items() if s in DONE}
    by_id = cs.by_id()
    rows: List[Dict[str, Any]] = []
    maybe: List[Dict[str, str]] = []
    pages: Dict[Tuple[str, Any], List[Dict[str, Any]]] = {}
    withdrawn: Dict[Tuple[str, Any], List[Dict[str, Any]]] = {}

    def where(rec: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
        ref = rec.get("entity") or {}
        table = ref.get("table")
        if table not in entities:
            return None
        pk = resolve_ref(ref, _candidates(table, entities[table]))
        return None if pk is None else (table, pk)

    for rec in cs.active:
        if rec["kind"] in ("withdraw",):
            continue
        page = where(rec)
        if page is None:
            rows.append(_row(rec, "needs a look", "the page can't be found"))
            continue
        pages.setdefault(page, []).append(rec)
    # A withdrawn strike that a run carried out: put the original back
    for cid in cs.withdrawn:
        rec = by_id.get(cid)
        if not rec or rec["kind"] != "strike" or before.get(cid) not in DONE:
            continue
        page = where(rec)
        if page is not None:
            withdrawn.setdefault(page, []).append(rec)
            pages.setdefault(page, [])
    # An older strike a run carried out, replaced by a newer strike of the
    # same sentence: its sentence goes before the newer one is carried out
    superseded: Dict[Tuple[str, Any], List[Dict[str, Any]]] = {}
    for cid in cs.replaced:
        rec = by_id.get(cid)
        if not rec or rec["kind"] != "strike" or not rec.get("replacement"):
            continue
        if before.get(cid) not in DONE:
            continue
        page = where(rec)
        if page is not None and page in pages:
            superseded.setdefault(page, []).append(rec)
    changed = 0
    for (table, pk), lines in pages.items():
        entity = entities[table][pk]
        profile = entity.get("profile") or {}
        rules = _rules(lines)
        text, states, again = apply_to_text(
            profile.get("text") or "",
            rules,
            done_before,
            withdrawn.get((table, pk), []),
            superseded.get((table, pk), []),
        )
        if text != (profile.get("text") or ""):
            entity["profile"] = {**profile, "text": text}
            _new_version(
                entity,
                sorted(
                    i
                    for i, (s, _) in states.items()
                    if s in ("applied", "re-applied", "undone")
                ),
            )
            changed += 1
        for cid, (state, why) in states.items():
            guarded = _GUARDED.get(cid)
            if guarded and state in ("applied", "held"):
                state, why = "re-applied", guarded
            rows.append(_row(by_id[cid], state, why))
        maybe += again
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r["state"]] = counts.get(r["state"], 0) + 1
    report = {
        "run": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "output_dir": output_dir,
        "counts": counts,
        "profiles_changed": changed,
        "rows": rows,
        "maybe_again": maybe,
        "problems": cs.problems,
        "contradictions": cs.contradictions,
    }
    if write:
        write_report(path, report)
    return report


def _row(rec: Dict[str, Any], state: str, why: str) -> Dict[str, Any]:
    ent = rec.get("entity") or {}
    return {
        "id": rec["id"],
        "kind": rec["kind"],
        "page": f"{ent.get('table')}/{ent.get('name') or ent.get('key')}",
        "state": state,
        "why": why,
    }


def write_report(path: str, report: Dict[str, Any]) -> None:
    folder = _report_dir(path)
    stamp = report["run"].replace(":", "").replace("-", "")
    body = json.dumps(report, ensure_ascii=False, indent=1)
    for name in (f"report-{stamp}.json", REPORT_LATEST):
        tmp = os.path.join(folder, name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(body)
        os.replace(tmp, os.path.join(folder, name))


def describe_report(report: Dict[str, Any]) -> str:
    """One line for the run log: "Corrections: 41 held, 3 re-applied, …"."""
    order = ("applied", "held", "re-applied", "undone", "waiting", "needs a look")
    counts = report.get("counts") or {}
    parts = [f"{counts[s]} {s}" for s in order if counts.get(s)]
    line = "Corrections: " + (", ".join(parts) or "none active")
    if report.get("maybe_again"):
        line += f"; {len(report['maybe_again'])} sentences may say a struck thing again"
    if report.get("problems"):
        line += f"; {len(report['problems'])} unreadable lines"
    if report.get("contradictions"):
        line += f"; {len(report['contradictions'])} contradictions"
    return line
