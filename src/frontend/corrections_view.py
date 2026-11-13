"""What the correction pages show, and how a submitted form becomes a line in
the corrections file (``src/utils/corrections.py``).

The reader never edits the entity tables. A correction is appended to the
file, and every page draws the file on top of the stored profile: a struck
sentence is shown struck through, the historian's own sentence follows it,
and an overruled verdict replaces the checker's (which stays visible beside
it). Splits and joins are carried out by the next pipeline run
(``src/engine/corrections_merges.py``), and each says what that run did;
headings, names and tables are recorded and listed as waiting.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

from src.utils.corrections import (
    DONE,
    KIND_LABELS,
    MERGE_KINDS,
    NEEDS_RUN,
    VERDICTS,
    CorrectionSet,
    append_correction,
    correction_sources,
    entity_ref,
    last_report,
    latest_name,
    load_corrections,
    resolve_correction,
    resolve_ref,
    sentence_key,
)

from .data_access import SPECS, TABLES, Entity, Store
from .profile_render import (
    _HEADING,
    VERDICT_LABELS,
    Overlay,
    RenderedProfile,
    plain_claim,
    render_profile,
)


def load(store: Store) -> CorrectionSet:
    return load_corrections(store.corrections_path)


def for_page(store: Store, cs: CorrectionSet, ent: Entity) -> List[Dict[str, Any]]:
    """Active corrections about this page (a join from either side), found the
    way the pipeline finds them (``find_page()``), so a page whose key changed
    since (an organization retyped, say) keeps its corrections, and so does
    the page a join or split moved them to."""
    mine = {a["article_id"] for a in ent.articles}

    def here(ref: Any) -> bool:
        if not isinstance(ref, dict) or ref.get("table") != ent.spec.table:
            return False
        return ref.get("key") == ent.key or find_entity(store, ref) is ent

    def about(r: Dict[str, Any]) -> bool:
        if r["kind"] in MERGE_KINDS:
            return here(r.get("entity")) or here(r.get("other"))
        ref = r.get("entity")
        if not isinstance(ref, dict) or ref.get("table") != ent.spec.table:
            return False
        srcs = correction_sources(r)
        if ref.get("key") == ent.key and (not srcs or srcs & mine):
            return True  # the usual case, without a lookup
        return find_page(store, cs, r) is ent

    return [r for r in cs.active if about(r)]


def overlay_for(store: Store, cs: CorrectionSet, ent: Entity) -> Overlay:
    """The struck sentences and overruled verdicts on one page."""
    ov = Overlay()
    for r in for_page(store, cs, ent):
        if r["kind"] == "strike":
            ov.struck[sentence_key(r["sentence"])] = r
        elif r["kind"] == "verdict":
            ov.verdicts[(sentence_key(r["sentence"]), r["article_id"])] = r
    return ov


def render_current(
    store: Store, ent: Entity, cs: CorrectionSet, link: str = "/sources/{article}"
) -> RenderedProfile:
    """The current profile with its corrections drawn on top."""
    return render_profile(
        ent.profile.get("text") or "",
        ent.row.get("profile_grounding"),
        link=link,
        overlay=overlay_for(store, cs, ent),
    )


def find_entity(store: Store, ref: Any) -> Optional[Entity]:
    """The page a correction names (``resolve_ref()``: by key, else by name,
    else by shared sources), as the pipeline's corrections pass finds it."""
    if not isinstance(ref, dict):
        return None
    ent = store.get(str(ref.get("table")), str(ref.get("key")))
    if ent is not None:
        return ent
    pages = store.entities.get(str(ref.get("table")), {}).values()
    return resolve_ref(
        ref, ((e.key, e.name, [a["article_id"] for a in e.articles], e) for e in pages)
    )


def find_page(store: Store, cs: CorrectionSet, rec: Dict[str, Any]) -> Optional[Entity]:
    """The page a correction is about now (``resolve_correction()``): the one
    it names, else the page a join put it into or a split moved its sources
    to, as the pipeline's corrections pass finds it."""
    ref = rec.get("entity")
    if not isinstance(ref, dict):
        return None
    pages = store.entities.get(str(ref.get("table")), {}).values()
    return resolve_correction(
        rec,
        cs,
        ((e.key, e.name, [a["article_id"] for a in e.articles], e) for e in pages),
    )


def _source_title(store: Store, aid: str) -> str:
    src = store.source(aid)
    return f"{src.title} ({src.date})" if src.date else src.title


def describe(store: Store, r: Dict[str, Any]) -> str:
    """One line saying what the correction says."""
    k = r["kind"]
    if k == "strike":
        out = f"“{r['sentence']}”"
        return out + (f" → “{r['replacement']}”" if r.get("replacement") else "")
    if k == "verdict":
        checker = VERDICT_LABELS.get(
            r.get("checker_verdict", ""), r.get("checker_verdict") or "no verdict"
        )
        return (f"“{r['sentence']}” against {_source_title(store, r['article_id'])}: "
                f"{VERDICT_LABELS[r['verdict']].lower()} (the checker said {checker.lower()})")  # fmt: skip
    if k == "heading":
        return f"“{r['sentence']}” belongs under “{r['heading']}”"
    if k == "rename":
        return f"call it “{r['name']}”"
    if k == "wrong_table":
        return f"belongs with {SPECS[r['to_table']].label.lower()}"
    if k == "not_entity":
        return "not a person, organization, place or event"
    if k == "split":
        titles = "; ".join(_source_title(store, a) for a in r["article_ids"])
        n = len(r["article_ids"])
        return f"{n} source{'s' if n != 1 else ''} about someone else: {titles}"
    if k == "join":
        return f"the same as “{r['other'].get('name') or r['other']['key']}”"
    return ""


def status(
    store: Store,
    cs: CorrectionSet,
    r: Dict[str, Any],
    cache: Dict[Tuple[str, str], RenderedProfile],
) -> Tuple[str, str]:
    """(tone, words): "ok" shown on the page, "wait" needs a pipeline run,
    "look" needs a person, "off" withdrawn or replaced."""
    state = cs.state(r)
    if state == "withdrawn":
        w = cs.withdrawn[r["id"]]
        return "off", f"Withdrawn by {w['id']}" + (
            f": {w['reason']}" if w.get("reason") else ""
        )
    if state == "replaced":
        return "off", f"Replaced by {cs.replaced[r['id']]}"
    if r["kind"] in MERGE_KINDS:
        return _merge_status(store, r)
    ent = find_page(store, cs, r)
    if ent is None:
        return (
            "look",
            "Needs a look: the page can't be found (renamed or merged since?)",
        )
    if r["kind"] == "join" and find_entity(store, r.get("other")) is None:
        return "look", "Needs a look: the other page can't be found"
    if r["kind"] in NEEDS_RUN:
        return (
            "wait",
            "Saved. Pipeline runs don't carry out this kind yet; it waits in the file.",
        )
    key = (ent.spec.table, ent.key)
    if key not in cache:
        cache[key] = render_current(store, ent, cs)
    gone = any(u["id"] == r["id"] for u in cache[key].unmatched)
    run = run_rows(store).get(r["id"])
    when = _when(str((run or {}).get("run", "")))
    done_at = _when(str((run or {}).get("since") or (run or {}).get("run", "")))
    if run and run["state"] == "needs a look":
        return "look", f"Needs a look after the run of {when}: {run.get('why', '')}"
    if r["kind"] == "strike" and run and run["state"] in DONE:
        again = f" again ({run['why']})" if run["state"] == "re-applied" else ""
        return "ok", f"Carried out in the tables{again} by the run of {done_at}"
    if gone:
        return "look", "Needs a look: the sentence is no longer on the page"
    if r["kind"] == "strike":
        return "ok", "Shown on the page; the next run carries it out in the tables"
    return "ok", "Shown on the page"


def _merge_status(store: Store, r: Dict[str, Any]) -> Tuple[str, str]:
    """A split or join: what the last run did with it. After a join the
    smaller page is gone, so the run's report says, not the tables."""
    run = run_rows(store).get(r["id"])
    if not run:
        if find_entity(store, r.get("entity")) is None or (
            r["kind"] == "join" and find_entity(store, r.get("other")) is None
        ):
            return "look", "Needs a look: a page it names can't be found"
        return "wait", "Saved. The next pipeline run carries it out."
    when = _when(str(run.get("run", "")))
    done_at = _when(str(run.get("since") or run.get("run", "")))
    why = f": {run['why']}" if run.get("why") else ""
    if run["state"] in DONE:
        return "ok", f"Carried out by the run of {done_at}{why}"
    if run["state"] == "waiting":
        return "wait", f"Not carried out by the run of {when}{why}"
    return "look", f"Needs a look after the run of {when}{why}"


def _when(stamp: str) -> str:
    """ "12 Nov 2026, 17:21 UTC" from a report's "2026-11-12T17:21:28Z"."""
    from .views import fmt_date  # views imports this module

    day = fmt_date(stamp[:10])
    if not day:
        return "the last run"
    return f"{day}, {stamp[11:16]} UTC" if len(stamp) >= 16 else day


def run_rows(store: Store) -> Dict[str, Dict[str, Any]]:
    """The last pipeline run's state for each correction (its report beside
    the corrections file), each row with the run's time under ``run``."""
    report = _last_report(store.corrections_path, store.output_dir)
    return {
        r["id"]: {**r, "run": report.get("run", "")}
        for r in report.get("rows") or []
        if isinstance(r, dict) and r.get("id")
    }


def _last_report(path: str, output_dir: str = "") -> Dict[str, Any]:
    """The last run's report for the tables this reader shows."""
    if not path:
        return {}
    where = os.path.join(
        os.path.dirname(os.path.abspath(path)), latest_name(output_dir)
    )
    try:
        stamp = os.path.getmtime(where)
    except OSError:
        return {}
    if _REPORT_CACHE.get("key") != (where, stamp):
        _REPORT_CACHE.update(key=(where, stamp), report=last_report(path, output_dir))
    return _REPORT_CACHE["report"]


_REPORT_CACHE: Dict[str, Any] = {}


def row(
    store: Store,
    cs: CorrectionSet,
    r: Dict[str, Any],
    cache: Dict[Tuple[str, str], RenderedProfile],
) -> Dict[str, Any]:
    tone, words = status(store, cs, r, cache)
    ent = (
        find_entity(store, r.get("entity"))
        if r["kind"] in MERGE_KINDS
        else find_page(store, cs, r)
    )
    return {"rec": r, "label": KIND_LABELS[r["kind"]], "what": describe(store, r),
            "tone": tone, "status": words, "ent": ent,
            "page_name": ent.name if ent else (r.get("entity") or {}).get("name", "")}  # fmt: skip


def entity_box(
    store: Store, cs: CorrectionSet, ent: Entity, rp: RenderedProfile
) -> Dict[str, Any]:
    """The page's own corrections, for the box above the profile."""
    cache = {(ent.spec.table, ent.key): rp}
    rows = [row(store, cs, r, cache) for r in for_page(store, cs, ent)]
    return {"rows": rows, "look": sum(1 for r in rows if r["tone"] == "look")}


def corrections_page(store: Store, cs: CorrectionSet, kind: str = "") -> Dict[str, Any]:
    cache: Dict[Tuple[str, str], RenderedProfile] = {}
    lines = [r for r in cs.lines if r["kind"] != "withdraw"]
    rows = [
        row(store, cs, r, cache)
        for r in reversed(lines)
        if not kind or r["kind"] == kind
    ]
    counts: Dict[str, int] = {}
    for r in lines:
        counts[r["kind"]] = counts.get(r["kind"], 0) + 1
    return {"rows": rows, "kind": kind, "counts": counts, "labels": KIND_LABELS,
            "problems": cs.problems, "contradictions": cs.contradictions,
            "path": store.corrections_path,
            "look": sum(1 for r in rows if r["tone"] == "look")}  # fmt: skip


def home_summary(store: Store) -> Dict[str, int]:
    cs = load(store)
    if not cs.lines:
        return {}
    cache: Dict[Tuple[str, str], RenderedProfile] = {}
    look = sum(1 for r in cs.active if status(store, cs, r, cache)[0] == "look")
    return {
        "active": len(cs.active),
        "look": look + len(cs.problems) + len(cs.contradictions),
    }


def _sources(store: Store, ent: Entity) -> List[Dict[str, Any]]:
    arts = sorted(ent.articles, key=lambda a: store.source(a["article_id"]).date)
    return [
        {"id": a["article_id"], "source": store.source(a["article_id"])} for a in arts
    ]


def _heading_of(rp: RenderedProfile, start: int) -> str:
    title = ""
    for m in _HEADING.finditer(rp.text):
        if m.start() > start:
            break
        title = m.group(2).strip()
    return title


def sentence_form(
    store: Store, cs: CorrectionSet, ent: Entity, index: int, article: str
) -> Optional[Dict[str, Any]]:
    """The form for one sentence: strike it, overrule the checker, move it."""
    rp = render_current(store, ent, cs)
    if not 0 <= index < len(rp.claims):
        return None
    c = rp.claims[index]
    here = _heading_of(rp, c.start)
    return {
        "ent": ent,
        "claim": c,
        "text": plain_claim(c.text),
        "index": index,
        "article": article
        if article in c.articles
        else (c.articles[0] if c.articles else ""),
        "cites": [
            {
                "id": a,
                "source": store.source(a),
                "n": rp.order.index(a) + 1 if a in rp.order else None,
                "verdict": c.verdicts.get(a, "unchecked")
                if rp.checked or a in c.checker
                else "unchecked",
                "checker": c.checker.get(a),
                "by": c.by.get(a),
            }
            for a in c.articles
        ],  # fmt: skip
        "sources": _sources(store, ent),
        "heading": here,
        "headings": [s.title for s in rp.sections if s.title != here],
        "verdicts": [(v, VERDICT_LABELS[v]) for v in VERDICTS],
        "by": c.by.get(""),
    }


def page_form(store: Store, cs: CorrectionSet, ent: Entity) -> Dict[str, Any]:
    """The form for the page: rename, wrong table, split, join, and a list of
    its sentences to correct one by one (works without JavaScript)."""
    rp = render_current(store, ent, cs)
    others = [
        e for e in store.entities.get(ent.spec.table, {}).values() if e.key != ent.key
    ]
    others.sort(key=lambda e: (-len(e.articles), e.name))
    return {
        "ent": ent,
        "sources": _sources(store, ent),
        "tables": [t for t in TABLES if t.table != ent.spec.table],
        "others": [
            {
                "key": e.key,
                "label": f"{e.name}{' · ' + e.type if e.type else ''} · {len(e.articles)} articles",
            }
            for e in others[:3000]
        ],  # fmt: skip
        "claims": [
            {"index": c.index, "text": plain_claim(c.text), "correction": c.correction}
            for c in rp.claims
        ],
        "box": entity_box(store, cs, ent, rp),
    }


def _one(form: Dict[str, List[str]], name: str) -> str:
    return (form.get(name) or [""])[0].strip()


def build_record(
    store: Store, cs: CorrectionSet, form: Dict[str, List[str]]
) -> Tuple[Dict[str, Any], Entity, str]:
    """A posted form as a correction (not yet saved), the page it is about,
    and the anchor to return to. Raises ``ValueError`` with a message for
    the person."""
    ent = store.get(_one(form, "t"), _one(form, "k"))
    if ent is None:
        raise ValueError("That page isn't in the tables any more; reload it.")
    kind = _one(form, "kind")
    ref = entity_ref(
        ent.spec.table, ent.key, ent.name, [a["article_id"] for a in ent.articles]
    )
    rec: Dict[str, Any] = {"kind": kind, "entity": ref, "reason": _one(form, "reason")}
    anchor = ""
    if kind in ("strike", "verdict", "heading"):
        rp = render_current(store, ent, cs)
        try:
            c = rp.claims[int(_one(form, "c"))]
        except (ValueError, IndexError):
            raise ValueError(
                "That sentence isn't on the page any more; reload it."
            ) from None
        text = plain_claim(c.text)
        if text != " ".join(_one(form, "sentence").split()):
            raise ValueError(
                "The page changed since you opened this form; reload it and try again."
            )
        if c.correction == "mine":
            raise ValueError(
                "This is your own sentence; withdraw its correction to change it."
            )
        rec["sentence"], anchor = text, f"#claim-{c.index}"
        if kind == "strike":
            rec["cites"] = list(c.articles)
            rep = " ".join(_one(form, "replacement").split())
            if rep:
                cite = _one(form, "replacement_cites")
                if cite not in ref["article_ids"]:
                    raise ValueError(
                        "Pick one of this page's sources for your sentence to cite."
                    )
                rec["replacement"], rec["replacement_cites"] = rep, cite
        elif kind == "verdict":
            aid = _one(form, "article_id")
            if aid not in c.articles:
                raise ValueError("Pick one of the sources this sentence cites.")
            rec["article_id"], rec["verdict"] = aid, _one(form, "verdict")
            rec["checker_verdict"] = c.checker.get(aid) or c.verdicts.get(
                aid, "unchecked"
            )
            rec["passage"] = _one(form, "passage")
        else:
            rec["heading"] = _one(form, "heading")
            if rec["heading"] not in [s.title for s in rp.sections]:
                raise ValueError("Pick one of the page's headings.")
    elif kind == "rename":
        rec["name"] = " ".join(_one(form, "name").split())
        if rec["name"] == ent.name:
            raise ValueError("That is already the page's name.")
    elif kind == "wrong_table":
        to = _one(form, "to_table")
        if to == "none":
            rec["kind"] = "not_entity"
        else:
            rec["to_table"] = to
    elif kind == "split":
        rec["article_ids"] = sorted(set(form.get("article_ids") or []))
        if not rec["article_ids"]:
            raise ValueError("Tick the sources that are about someone else.")
        rec["name"] = " ".join(_one(form, "name").split())  # the new page's
    elif kind == "join":
        value = _one(form, "other")
        table = store.entities.get(ent.spec.table, {})
        other = table.get(value)
        if other is None:
            named = [e for e in table.values() if e.name.lower() == value.lower()]
            if len(named) != 1:
                raise ValueError(
                    f"No single {ent.spec.singular} page is called “{value}”; pick one from the list."
                )
            other = named[0]
        rec["other"] = entity_ref(
            other.spec.table,
            other.key,
            other.name,
            [a["article_id"] for a in other.articles],
        )
    else:
        raise ValueError("Unknown kind of correction.")
    return rec, ent, anchor


def save(
    store: Store, cs: CorrectionSet, form: Dict[str, List[str]]
) -> Tuple[Dict[str, Any], Entity, str]:
    rec, ent, anchor = build_record(store, cs, form)
    return append_correction(store.corrections_path, rec), ent, anchor


def withdraw(
    store: Store, cs: CorrectionSet, form: Dict[str, List[str]]
) -> Dict[str, Any]:
    cid = _one(form, "id")
    rec = cs.by_id().get(cid)
    if rec is None or rec["kind"] == "withdraw":
        raise ValueError(f"There is no correction {cid}.")
    if cs.state(rec) != "active":
        raise ValueError(f"{cid} is already {cs.state(rec)}.")
    return append_correction(
        store.corrections_path,
        {"kind": "withdraw", "withdraws": cid, "reason": _one(form, "reason")},
    )
