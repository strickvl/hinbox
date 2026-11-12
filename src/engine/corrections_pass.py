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
from src.logging_config import log
from src.utils.claim_spans import claim_end, parse_claims
from src.utils.corrections import (
    DONE,
    MERGE_KINDS,
    NEEDS_RUN,
    REPORT_LATEST,
    CorrectionSet,
    _report_dir,
    drop_list_mark,
    last_report,
    last_states,
    latest_name,
    load_corrections,
    md_literal,
    resolve_ref,
    stored_key,
)
from src.utils.corrections import sentence_key as _raw_key
from src.utils.entity_keys import KEY_FUNCS, NAME_FIELDS

VERSION_TRIGGER = "corrections:"  # a profile version made by this pass


def sentence_key(text: str) -> str:
    """``sentence_key()`` of a claim read from the stored text (or a struck
    sentence copied from it): a list item's own marker is dropped, then the
    escapes of a historian's sentence (stored with ``md_literal()``), so
    "- He arrived in 2001." in a list matches the words they wrote."""
    return stored_key(text)


def words_key(words: str) -> str:
    """``sentence_key()`` of a historian's sentence as they wrote it."""
    return _raw_key(words)


_ACTIVE: Optional[CorrectionSet] = None
_CURRENT: contextvars.ContextVar = contextvars.ContextVar("corrections", default=None)
# Strikes the edit guard had to defend during this run: id -> why
_GUARDED: Dict[str, str] = {}
# Everything the edit guard did during this run, for the run report
_GUARD_LOG: List[Dict[str, Any]] = []
_PAGE: contextvars.ContextVar = contextvars.ContextVar("corrections_page", default=None)
_LOCK = threading.Lock()


@dataclass
class PageRules:
    """One page's active corrections, by what they protect."""

    struck: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # key -> line
    written: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # key -> line
    # every strike whose sentence has these words (two strikes may both be
    # replaced by "Correct.", each citing its own source)
    sharing: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
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
                rules.written[words_key(r["replacement"])] = r
                rules.sharing.setdefault(words_key(r["replacement"]), []).append(r)
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


def active_corrections() -> Optional[CorrectionSet]:
    """The corrections loaded for this run (None with none active)."""
    return _ACTIVE


def configure_corrections(cs: Optional[CorrectionSet]) -> None:
    """Load the corrections the edit guard and the merge rules check during
    this run."""
    global _ACTIVE
    _ACTIVE = cs if cs and cs.active else None
    with _LOCK:
        _GUARDED.clear()
        _GUARD_LOG.clear()


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
    page = _PAGE.set((table, entity.get(NAME_FIELDS[table]) or ""))
    try:
        yield
    finally:
        _CURRENT.reset(token)
        _PAGE.reset(page)


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


def _note(cid: str, why: str, action: str = "", text: str = "") -> None:
    """Record that the guard stepped in for correction ``cid``: its state
    becomes re-applied (``why``), and the run report lists what it did."""
    table, name = _PAGE.get() or ("", "")
    with _LOCK:
        if why:
            _GUARDED.setdefault(cid, why)
        _GUARD_LOG.append(
            {"correction": cid, "table": table, "page": name, "did": action,
             "text": text[:200]}
        )  # fmt: skip
    log(f"Corrections guard on {name or '?'} ({cid}): {action}", level="info")


def guard_log() -> List[Dict[str, Any]]:
    """What the edit guard did during this run (see ``configure_corrections``)."""
    with _LOCK:
        return list(_GUARD_LOG)


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
            _note(
                hit[0]["id"],
                "the edit model tried to reword your sentence",
                "dropped an edit rewording your sentence",
                edit.text,
            )
            continue
        if hit and span[1] < hit[1].end:
            # An add inside their sentence would split it: it goes after it
            edit = edit.model_copy(update={"anchor": text[hit[1].start : hit[1].end]})
            report["moved_after_yours"].append(edit.text[:200])
            _note(
                hit[0]["id"], "", "moved an addition to after your sentence", edit.text
            )
        new_text, cut = _cut_restated(edit.text, rules)
        for cid, claim in cut:
            report["restated"].append(claim[:200])
            _note(
                cid,
                f"an update wrote it again: “{claim[:160]}”",
                "cut a sentence saying a struck one again",
                claim,
            )
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


# a sentence's own full stop, when the model put it after the citation
# ("born in Leeds^[a1]. She moved"): it goes with the sentence
_STOP_AFTER = re.compile(r"[.!?]+(?=\s|$)")
_STOP_IN = re.compile(r"[.!?]+(?=[\"'”’)]*\s*$)")
_TRAILING_MARKERS = re.compile(r"(?:\^\[[^\]]*\])+$")


def _ends_a_sentence(words: str) -> bool:
    return bool(_STOP_IN.search(words))


def _cut_claim(text: str, markers: List[Any], c: Any, put: str = "") -> str:
    """``text`` without claim ``c``'s words, and ``put`` where its last block
    was. The sentence's full stop goes with it, before or after its citation,
    and ``put`` gets one if its own words have none: "Leeds^[a1]." becomes
    "Hull.^[a1]" for "Hull.", and "Hull^[a1]." for "Hull"."""
    ranges = _claim_ranges(text, markers, c)
    start, end = ranges[-1]
    stop = ""
    if m := _STOP_AFTER.match(text, end):
        stop, end = m.group(), m.end()
    elif m := _STOP_IN.search(text[start:end].split("^[")[0]):
        stop = m.group()  # inside: kept by the cut, lent to ``put``
    ranges[-1] = (start, end)
    if put and stop and not _ends_a_sentence(_TRAILING_MARKERS.sub("", put)):
        put += stop
    for n, (start, end) in enumerate(reversed(ranges)):
        text = _remove(text, start, end, put if n == 0 else "")
    return text


def _yours(rec: Dict[str, Any]) -> str:
    words = md_literal(" ".join(rec["replacement"].split()))
    return f"{words}^[{rec['replacement_cites']}]"


def _original(rec: Dict[str, Any]) -> str:
    cites = rec.get("cites") or []
    # the list marker the reader copied stays in the text when a claim is cut
    words = drop_list_mark(rec["sentence"]).strip()
    return words + (f"^[{', '.join(cites)}]" if cites else "")


def _swap_back(text: str, rec: Dict[str, Any]) -> Tuple[str, bool]:
    """``text`` with an earlier strike's replacement swapped back for the
    original sentence, if the replacement is there and the original isn't."""
    claims, markers = parse_claims(text)
    by_key = {sentence_key(c.text): c for c in claims}
    rep = by_key.get(words_key(rec.get("replacement") or ""))
    cite = rec.get("replacement_cites")
    if rep is None:
        return text, False
    if words_key(rec.get("replacement") or "") == sentence_key(rec["sentence"]):
        # same words, new source: undone by the citation, not the words
        if rep.articles != [cite]:
            return text, False
        return _cut_claim(text, markers, rep, _original(rec)), True
    if sentence_key(rec["sentence"]) in by_key:
        return text, False
    if any(a != cite for a in rep.articles):
        # their words were in the profile already, citing other sources
        # ("Correct.^[a]^[b]"): that claim stays, without their citation,
        # and the original goes back in front of it
        # (the pass writes their citation as a marker of its own, wherever
        # it sits among the claim's markers: "^[c]^[a]^[b]" loses "^[a]")
        mine = [m for m in markers if m.claim == rep.index and m.articles == [cite]]
        if cite and mine:
            text = text[: mine[-1].start] + text[mine[-1].end :]
        return text[: rep.start] + _original(rec) + " " + text[rep.start :], True
    return _cut_claim(text, markers, rep, _original(rec)), True


def _is_theirs(c: Any, rules: PageRules) -> bool:
    """A claim that is the historian's own sentence with only their source."""
    key = sentence_key(c.text)
    rec = rules.written.get(key)
    struck = rules.struck.get(key)
    if struck is not None and (rec is None or struck["id"] != rec["id"]):
        return False  # they struck their own sentence again later
    cites = {r.get("replacement_cites") for r in rules.sharing.get(key, [])}
    return bool(rec) and bool(c.articles) and set(c.articles) <= cites


def _add_your_sources(text: str, rules: PageRules) -> str:
    """Each historian's sentence in the text cites the source they chose: one
    whose words were already in the profile ("Correct.^[a]", they cite b)
    gets their citation added after its own ("Correct.^[a]^[b]")."""
    claims, markers = parse_claims(text)
    adds = []
    for c in claims:
        missing = []
        for rec in rules.sharing.get(sentence_key(c.text), []):
            cite = rec.get("replacement_cites")
            if cite and cite not in c.articles and cite not in missing:
                missing.append(cite)
        if missing:
            add = "".join(f"^[{cite}]" for cite in missing)
            adds.append((claim_end(markers, c.index), add))
    for pos, add in sorted(adds, reverse=True):
        text = text[:pos] + add + text[pos:]
    return text


def _chain(rec: Dict[str, Any], rules: PageRules) -> List[Dict[str, Any]]:
    """``rec`` and each active strike of the sentence before it in the list
    (2000 → 2001, then 2001 → 2002: both)."""
    out = [rec]
    while rec.get("replacement"):
        nxt = rules.struck.get(words_key(rec["replacement"]))
        if nxt is None or any(r["id"] == nxt["id"] for r in out):
            break
        out.append(nxt)
        rec = nxt
    return out


def _last_in_chain(rec: Dict[str, Any], rules: PageRules) -> Dict[str, Any]:
    """The strike whose sentence goes in for ``rec``: when the historian later
    struck their own sentence, the later strike's sentence (2000 → 2001, then
    2001 → 2002 writes 2002)."""
    return _chain(rec, rules)[-1]


def apply_to_text(
    text: str,
    rules: PageRules,
    done_before: Set[str],
    withdrawn: List[Dict[str, Any]],
    superseded: Sequence[Dict[str, Any]] = (),
    gone: Set[str] = frozenset(),
) -> Tuple[str, Dict[str, Tuple[str, str]], List[Dict[str, str]]]:
    """Carry one page's corrections out in its profile text. Returns the new
    text, (state, why) per correction id, and claims that may say a struck
    thing again in other words (sharing a year and a name with it).

    ``superseded`` are older strikes a run carried out that a newer strike
    of the same sentence replaced: their sentence is first swapped back for
    the original, so the newer strike is carried out on it like any other
    (2000 → 2001 by the old one, then 2001 → original → 2002)."""
    states: Dict[str, Tuple[str, str]] = {}
    # First put back what withdrawn and replaced strikes changed, so the
    # strikes still active are carried out on the original text in this
    # same run (a withdrawn 2000 → 2002 brings back the older 2000 → 2001)
    # A chain (2000 → 2001, then 2001 → 2002) withdrawn whole unwinds from
    # its last link: each round swaps back what it can, until none moves
    waiting_back = list(withdrawn)
    while waiting_back:
        left = []
        for rec in waiting_back:
            text, swapped = _swap_back(text, rec)
            if swapped:
                states[rec["id"]] = (
                    "undone",
                    "your sentence swapped back for the original",
                )
            else:
                left.append(rec)
        if len(left) == len(waiting_back):
            break
        waiting_back = left
    for rec in waiting_back:
        if sentence_key(rec["sentence"]) not in {
            sentence_key(c.text) for c in parse_claims(text)[0]
        }:
            states[rec["id"]] = (
                "needs a look",
                "withdrawn after a run cut the sentence; it is not put back "
                "(its words are in the correction line)",
            )
        else:
            states[rec["id"]] = ("undone", "")
    for old in superseded:
        newer = rules.written.get(words_key(old.get("replacement") or ""))
        # same words but other sources: swapped too, so the new citation goes in
        if not newer or newer.get("replacement_cites") != old.get("replacement_cites"):
            text, swapped = _swap_back(text, old)
            chain = _chain(old, rules)
            if not swapped and len(chain) > 1:
                # its sentence was struck again (2000 → 2001 → 2002): the
                # last one in the chain goes back to the original
                last = {
                    **chain[-1],
                    "sentence": old["sentence"],
                    "cites": old.get("cites"),
                }
                text, swapped = _swap_back(text, last)
                for link in chain[1:] if swapped else []:
                    states[link["id"]] = (
                        "held",
                        f"the sentence it struck went with {old['id']}, "
                        "which a newer correction replaced",
                    )
    claims, markers = parse_claims(text)
    # Which claims go: a struck one, unless it is the historian's own
    # sentence citing their source (same words, new source: "Correct.^[b]")
    to_cut = [
        (c, r)
        for c in claims
        if not _is_theirs(c, rules) and (r := _restated(c.text, rules))
    ]
    cut = {c.index for c, _ in to_cut}
    present = {
        k
        for c in claims
        if c.index not in cut and (k := sentence_key(c.text)) in rules.written
    }
    edits: List[Tuple[Any, str]] = []
    for c, rec in to_cut:
        key = sentence_key(c.text)
        last = _last_in_chain(rec, rules)
        put = _yours(last) if last.get("replacement") else ""
        rep_key = words_key(last.get("replacement") or "")
        if last is not rec and last["id"] not in done_before:
            states.setdefault(last["id"], ("applied", ""))
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
        cid, rep = rec["id"], words_key(rec.get("replacement") or "")
        if cid in states:
            continue
        later = rules.struck.get(rep) if rep else None
        if sentence_key(rec["sentence"]) in gone and (not rep or rep not in present):
            # it struck a sentence of theirs that a newer correction replaced
            states[cid] = (
                "held",
                "the sentence it struck was yours, and a newer correction replaced it",
            )
        elif later is not None and later["id"] != cid:
            # their own sentence, struck by a later correction of theirs
            states[cid] = ("held", f"your sentence was struck again by {later['id']}")
        elif rep and rep not in present and cid in done_before:
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
    # last, so a sentence put back above also gets every strike's source
    text = _add_your_sources(text, rules)
    claims, markers = parse_claims(text)
    by_key = {sentence_key(c.text): c for c in claims}
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
    early: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Carry out the corrections file's strikes in every profile and return
    the report (None with no file). ``early`` is what the start of the run
    did with splits and joins (``carry_out_merges()``); without it they are
    reported as waiting for a run. With ``write``, the report is written
    beside the file at once; a run passes False and writes it with
    ``write_report()`` once the tables are saved, so the report never says a
    strike was carried out in tables that were not written."""
    if not os.path.exists(path):
        return None
    cs = load_corrections(path)
    before = last_states(path, output_dir)
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

    early = early or {}
    for rec in cs.active:
        if rec["kind"] in MERGE_KINDS:  # carried out before merging
            state, why = (early.get("states") or {}).get(
                rec["id"], ("waiting", "a pipeline run carries it out")
            )
            rows.append(_row(rec, state, why))
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
    gone: Dict[Tuple[str, Any], Set[str]] = {}
    for cid in cs.replaced:
        rec = by_id.get(cid)
        if not rec or rec["kind"] != "strike" or not rec.get("replacement"):
            continue
        page = where(rec)
        if page is not None:
            gone.setdefault(page, set()).add(words_key(rec["replacement"]))
        if before.get(cid) not in DONE:
            continue
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
            gone.get((table, pk), set()),
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
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    _since(rows, last_report(path, output_dir), now)
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r["state"]] = counts.get(r["state"], 0) + 1
    report = {
        "run": now,
        "output_dir": output_dir,
        "counts": counts,
        "profiles_changed": changed,
        "rows": rows,
        "maybe_again": maybe,
        "split_mixed": early.get("mixed") or [],
        "guarded": guard_log(),
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


def _since(rows: List[Dict[str, Any]], last: Dict[str, Any], now: str) -> None:
    """Give each row ``since``: when a run last changed the tables for it.
    A held strike keeps the time of the run that carried it out, so the
    reader can say which run did it, not only which ran last."""
    old = {r.get("id"): r for r in last.get("rows") or [] if isinstance(r, dict)}
    for r in rows:
        prev = old.get(r["id"]) or {}
        if r["state"] == "held" and prev.get("state") in DONE:
            r["since"] = prev.get("since") or last.get("run") or now
        else:
            r["since"] = now


def write_report(path: str, report: Dict[str, Any]) -> None:
    folder = _report_dir(path)
    stamp = report["run"].replace(":", "").replace("-", "")
    body = json.dumps(report, ensure_ascii=False, indent=1)
    names = {f"report-{stamp}.json", REPORT_LATEST}
    names.add(latest_name(report.get("output_dir") or ""))
    for name in sorted(names):
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
    if report.get("split_mixed"):
        line += (
            f"; {len(report['split_mixed'])} sentences cite sources a split moved "
            "and others (kept, to check)"
        )
    if report.get("guarded"):
        line += f"; the edit guard stepped in {len(report['guarded'])} times"
    if report.get("problems"):
        line += f"; {len(report['problems'])} unreadable lines"
    if report.get("contradictions"):
        line += f"; {len(report['contradictions'])} contradictions"
    return line
