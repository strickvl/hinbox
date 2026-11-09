"""Build what each page shows from the store: plain dicts and lists that the
Jinja templates lay out. No HTML here except the rendered profile body and
highlighted source paragraphs, which are escaped before markup is added."""

import html
import re
import threading
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from markupsafe import Markup

from . import corrections_view as cv
from .data_access import SPECS, TABLES, Entity, Source, Store, pretty_type, to_date
from .profile_render import (
    SEVERITY,
    VERDICT_LABELS,
    Claim,
    RenderedProfile,
    attach_verdicts,
    parse_claims,
    plain_claim,
    render_profile,
    shift_headings,
)

# Hand test of grounding judges on 64 labelled claims, 42 of them wrong or
# partly wrong (CLAUDE.md, "Configuration"; README "Choosing a grounding
# judge"): how many of those 42 each judge still called "supported".
# Keep this in step with new measurements.
CHECKER_MISSES: List[Tuple[Tuple[str, ...], int]] = [
    (("haiku",), 2),
    (("luna",), 4),
    (("qwen", "high"), 11),
    (("qwen",), 19),
]


def checker_note(model: str) -> Dict[str, str]:
    """The checker's display name and what its hand test found."""
    name = re.sub(r"^(ollama|anthropic|openai|gemini)/", "", model or "").strip()
    low = (model or "").lower()
    note = "This checker has not been hand-tested yet, so treat unmarked sentences with care."
    for words, misses in CHECKER_MISSES:
        if all(w in low for w in words):
            note = (
                f"In a hand test it passed {misses} of 42 wrong or partly wrong claims "
                "as supported, so an unmarked sentence is likely, not certain, to be right."
            )
            break
    return {"name": name or "an unknown checker", "note": note}


def fmt_date(iso: str) -> str:
    """ "2024-07-31" -> "31 Jul 2024"."""
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", iso or "")
    if not m:
        return iso or ""
    months = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
    return f"{int(m.group(3))} {months[int(m.group(2)) - 1]} {m.group(1)}"


def _issue_counts(claims: List[Claim]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for c in claims:
        if c.verdicts and c.verdict != "supported":
            out[c.verdict] = out.get(c.verdict, 0) + 1
    return out


_ISSUES: Dict[int, Dict[str, Dict[str, int]]] = {}
_ISSUES_LOCK = threading.Lock()


def issues_by_entity(store: Store) -> Dict[str, Dict[str, int]]:
    """Per entity URL, the checker's exception counts. Computed once per
    store (parsing every profile takes about a second)."""
    with _ISSUES_LOCK:
        cached = _ISSUES.get(id(store))
        if cached is not None:
            return cached
        out: Dict[str, Dict[str, int]] = {}
        for ent in store.all():
            claims, _ = parse_claims(shift_headings(ent.profile.get("text") or ""))
            if attach_verdicts(claims, ent.row.get("profile_grounding")):
                counts = _issue_counts(claims)
                if counts:
                    out[ent.url] = counts
        _ISSUES.clear()
        _ISSUES[id(store)] = out
        return out


def entity_summary(store: Store, ent: Entity) -> Dict[str, Any]:
    """One row of a list: what a reader needs to pick the right page."""
    dates = sorted(
        d for d in (to_date(a.get("article_published_date")) for a in ent.articles) if d
    )
    return {
        "name": ent.name,
        "url": ent.url,
        "type": ent.type,
        "table": ent.spec.table,
        "singular": ent.spec.singular,
        "kind": store.kind_of(ent),
        "n_articles": len(ent.articles),
        "first": dates[0][:4] if dates else "",
        "last": dates[-1][:4] if dates else "",
        "last_date": dates[-1] if dates else "",
        "other_names": ent.other_names[:3],
        "issues": issues_by_entity(store).get(ent.url, {}),
    }


def _versions(ent: Entity) -> List[Dict[str, Any]]:
    pv = ent.row.get("profile_versions") or {}
    return sorted(pv.get("versions") or [], key=lambda v: v.get("version_number", 0))


def _history(store: Store, ent: Entity) -> List[Dict[str, Any]]:
    rows = []
    prev: set = set()
    for v in _versions(ent):
        text = (v.get("profile_data") or {}).get("text") or ""
        claims, _ = parse_claims(text)
        now = {plain_claim(c.text) for c in claims}
        added, removed = len(now - prev), len(prev - now)
        trigger = v.get("trigger_article_id")
        src = store.sources.get(trigger) if trigger else None
        rows.append(
            {
                "n": v.get("version_number"),
                "date": fmt_date(to_date(v.get("created_at"))),
                "source": src,
                "added": added,
                "removed": removed,
                "chars": len(text),
            }
        )
        prev = now
    return list(reversed(rows))


def _event_dates(ent: Entity) -> str:
    if ent.spec.table != "events":
        return ""
    start, end = to_date(ent.row.get("start_date")), to_date(ent.row.get("end_date"))
    text = fmt_date(start)
    if end and end != start:
        text = f"{text} – {fmt_date(end)}"
    if text and ent.row.get("is_fuzzy_date"):
        text += " (approximate)"
    return text


def _version_text(
    ent: Entity, version: Optional[int]
) -> Tuple[str, Optional[Dict[str, Any]], int]:
    """The profile text to show, the old version's record (None for the
    current one) and the current version number."""
    versions = _versions(ent)
    current = (ent.row.get("profile_versions") or {}).get("current_version") or len(
        versions
    )
    if version and versions and version != current:
        old = next((v for v in versions if v.get("version_number") == version), None)
        if old:
            return (old.get("profile_data") or {}).get("text") or "", old, current
    return ent.profile.get("text") or "", None, current


def entity_page(
    store: Store, ent: Entity, version: Optional[int], domain_query: str = ""
) -> Dict[str, Any]:
    """``domain_query`` ("?domain=x" or "") is added to citation links, so a
    citation followed without the script stays in the same domain."""
    versions = _versions(ent)
    text, old, current = _version_text(ent, version)
    report = ent.row.get("profile_grounding") or {}
    cs = cv.load(store)
    rp: RenderedProfile = render_profile(
        text,
        report,
        link="/sources/{article}" + domain_query,
        show_verdicts=old is None,
        overlay=cv.overlay_for(store, cs, ent) if old is None else None,
    )
    counts = rp.counts()
    issues = [
        {"verdict": k, "label": VERDICT_LABELS.get(k, k).lower(), "n": counts[k]}
        for k in sorted(counts, key=lambda k: -SEVERITY.get(k, 1))
        if k != "supported"
    ]
    summary = entity_summary(store, ent)
    cites_by_source: Counter = Counter(a for c in rp.claims for a in c.articles)
    flags_by_source: Dict[str, Counter] = {}
    for c in rp.claims:
        for aid, v in c.verdicts.items():
            if v != "supported":
                flags_by_source.setdefault(aid, Counter())[v] += 1

    arts = sorted(ent.articles, key=lambda a: to_date(a.get("article_published_date")))
    own = {a["article_id"] for a in arts}
    timeline = []
    for a in arts:
        src = store.source(a["article_id"])
        events = [
            e
            for e in store.by_source.get(a["article_id"], [])
            if e.spec.table == "events" and e.key != ent.key
        ][:3]
        timeline.append({"source": src, "date": src.date, "events": events})

    shared: Counter = Counter()
    for aid in own:
        for other in store.by_source.get(aid, []):
            if other.url != ent.url:
                shared[other.url] += 1
    lookup = {e.url: e for aid in own for e in store.by_source.get(aid, [])}
    appears = [
        {"ent": lookup[u], "n": n, "kind": store.kind_of(lookup[u])}
        for u, n in sorted(shared.items(), key=lambda kv: (-kv[1], lookup[kv[0]].name))[
            :15
        ]
    ]

    ordered = rp.order + sorted(
        (aid for aid in own if aid not in rp.order),
        key=lambda aid: store.source(aid).date,
    )
    sources = [
        {
            "n": rp.order.index(aid) + 1 if aid in rp.order else None,
            "source": store.source(aid),
            "cites": cites_by_source.get(aid, 0),
            "flags": dict(flags_by_source.get(aid, {})),
        }
        for aid in ordered
    ]
    disputes = [
        {
            "claim": c,
            "text": plain_claim(c.text),
            "verdict": c.verdict if rp.checked else "",
        }
        for c in rp.claims
        if c.is_dispute
    ]
    return {
        "ent": ent,
        "summary": summary,
        "kind": store.kind_of(ent),
        "event_dates": _event_dates(ent),
        "description": ent.row.get("description") if ent.spec.table == "events" else "",
        "body": Markup(rp.html),
        "rendered": rp,
        "sections": rp.sections,
        "checked": rp.checked,
        "issues": issues,
        "n_claims": len(rp.claims),
        "checker": checker_note(str(report.get("model") or "")),
        "timeline": timeline,
        "appears": appears,
        "sources": sources,
        "disputes": disputes,
        "history": _history(store, ent) if len(versions) > 1 else [],
        "version": old.get("version_number") if old else None,
        "current_version": current,
        "old_date": fmt_date(to_date(old.get("created_at"))) if old else "",
        "first_source": store.source(arts[0]["article_id"]) if arts else None,
        "cite_data": cite_data(rp),
        "corrections": cv.entity_box(store, cs, ent, rp) if old is None else None,
    }


def cite_data(rp: RenderedProfile) -> Dict[str, Any]:
    """What the static copy's panel needs about each cited sentence (the
    server's ``/panel`` works it out per click): its words and, per source,
    the checker's verdict and reason. Empty verdicts when nothing was checked
    or the page shows an older version."""
    return {
        str(c.index): {
            "q": plain_claim(c.text),
            "v": dict(c.verdicts) if rp.checked else {},
            "r": dict(c.reasons) if rp.checked else {},
            # A historian's corrections, so the static panel can say whose
            # verdict it shows: x = struck/mine/verdict, c = the checker's
            # overruled verdicts, i = correction id per source ("" = sentence)
            "x": c.correction,
            "c": dict(c.checker),
            "i": {k: r["id"] for k, r in c.by.items()},
        }
        for c in rp.claims
    }


def _highlight(paragraph: str, needle: str) -> Markup:
    """A paragraph, escaped, with ``needle`` marked where it appears word for
    word (a quoted stub sentence does; most written sentences don't)."""
    esc = html.escape(paragraph)
    if needle and len(needle) >= 25:
        n = html.escape(needle)
        i = esc.find(n)
        if i >= 0:
            return Markup(f"{esc[:i]}<mark>{n}</mark>{esc[i + len(n) :]}")
    return Markup(esc)


def named_in(store: Store, article_id: str) -> List[Dict[str, Any]]:
    """The entities linked to a source, grouped by table."""
    groups = []
    linked = store.by_source.get(article_id, [])
    for spec in TABLES:
        ents = sorted(
            (e for e in linked if e.spec.table == spec.table), key=lambda e: e.name
        )
        if ents:
            groups.append({"spec": spec, "entities": ents})
    return groups


def source_panel(
    store: Store,
    ent: Entity,
    article_id: str,
    claim_index: Optional[int],
    version: Optional[int] = None,
) -> Dict[str, Any]:
    """The panel for one citation. ``claim_index`` counts claims in the
    version the reader is looking at; an older version gets no verdict,
    since the stored report describes the current text only."""
    src = store.source(article_id)
    text, old, _ = _version_text(ent, version)
    cs = cv.load(store)
    rp = render_profile(
        text,
        ent.row.get("profile_grounding"),
        show_verdicts=old is None,
        overlay=cv.overlay_for(store, cs, ent) if old is None else None,
    )
    claim = (
        rp.claims[claim_index]
        if claim_index is not None and 0 <= claim_index < len(rp.claims)
        else None
    )
    verdict = reason = ""
    quote = ""
    if claim:
        quote = plain_claim(claim.text)
        if rp.checked or article_id in claim.checker:
            verdict = claim.verdicts.get(article_id, "unchecked")
            reason = claim.reasons.get(article_id, "")
    needle = quote.rstrip(".,;: ")
    paras = [_highlight(p, needle) for p in src.paragraphs()]
    return {
        "source": src,
        "n": rp.order.index(article_id) + 1 if article_id in rp.order else None,
        "claim": quote,
        "verdict": verdict,
        "verdict_label": VERDICT_LABELS.get(verdict, verdict),
        "reason": reason,
        "paragraphs": paras,
        "highlighted": any("<mark>" in p for p in paras),
        "old_version": old is not None,
        "named": named_in(store, article_id),
        "ent": ent,
        "index": claim.index if claim else None,
        "correction": claim.correction if claim else "",
        "mine": (claim.by.get(article_id) or claim.by.get("")) if claim else None,
        "checker_said": claim.checker.get(article_id, "") if claim else "",
    }


def source_page(store: Store, article_id: str) -> Dict[str, Any]:
    src = store.source(article_id)
    citing = []
    for ent in store.by_source.get(article_id, []):
        claims, _ = parse_claims(shift_headings(ent.profile.get("text") or ""))
        checked = attach_verdicts(claims, ent.row.get("profile_grounding"))
        rows = [
            {
                "text": plain_claim(c.text),
                "verdict": c.verdicts.get(article_id, "unchecked") if checked else "",
                "index": c.index,
            }
            for c in claims
            if article_id in c.articles
        ]
        if rows:
            citing.append({"ent": ent, "claims": rows})
    citing.sort(key=lambda r: (TABLES.index(r["ent"].spec), r["ent"].name))
    return {
        "source": src,
        "paragraphs": [Markup(html.escape(p)) for p in src.paragraphs()],
        "named": named_in(store, article_id),
        "citing": citing,
        "n_citing": sum(len(r["claims"]) for r in citing),
    }


def home_page(store: Store) -> Dict[str, Any]:
    issues = issues_by_entity(store)
    by_url = {e.url: e for e in store.all()}
    attention = sorted(
        issues.items(),
        key=lambda kv: (-kv[1].get("not_supported", 0), -kv[1].get("partial", 0)),
    )
    attention = [
        {"summary": entity_summary(store, by_url[u]), "issues": c}
        for u, c in attention
        if c.get("not_supported")
    ][:10]
    tables = []
    for spec in TABLES:
        ents = list(store.entities.get(spec.table, {}).values())
        top = sorted(ents, key=lambda e: -len(e.articles))[:6]
        tables.append(
            {
                "spec": spec,
                "count": len(ents),
                "full": sum(1 for e in ents if store.kind_of(e) == "full"),
                "top": [entity_summary(store, e) for e in top],
            }
        )
    totals: Counter = Counter()
    for c in issues.values():
        totals.update(c)
    latest = sorted(
        (s for s in store.sources.values() if s.id in store.by_source),
        key=lambda s: s.date,
        reverse=True,
    )[:8]
    return {
        "tables": tables,
        "attention": attention,
        "totals": dict(totals),
        "n_sources": sum(1 for s in store.sources if s in store.by_source),
        "latest": latest,
        "queue": store.review_queue_count(),
        "corrections": cv.home_summary(store),
    }


def _matches(q: str, *fields: str) -> bool:
    return any(q in (f or "").lower() for f in fields)


def search(store: Store, q: str, limit: int = 40) -> Dict[str, Any]:
    q = q.strip().lower()
    groups = []
    if q:
        for spec in TABLES:
            hits = [
                e
                for e in store.entities.get(spec.table, {}).values()
                if _matches(q, e.name, *e.other_names)
            ]
            hits.sort(
                key=lambda e: (not e.name.lower().startswith(q), -len(e.articles))
            )
            if hits:
                groups.append(
                    {
                        "spec": spec,
                        "total": len(hits),
                        "rows": [entity_summary(store, e) for e in hits[:limit]],
                    }
                )
        srcs = [
            s
            for s in store.sources.values()
            if s.id in store.by_source and q in s.title.lower()
        ]
        srcs.sort(key=lambda s: s.date, reverse=True)
    else:
        srcs = []
    return {"q": q, "groups": groups, "sources": srcs[:limit], "n_sources": len(srcs)}


SORTS = {"articles": "Most articles", "name": "Name", "recent": "Most recent"}
KINDS = {"full": "Full profiles", "stub": "Stubs", "link-only": "Link-only"}


def list_page(store: Store, table: str, params: Dict[str, str]) -> Dict[str, Any]:
    spec = SPECS[table]
    ents = list(store.entities.get(table, {}).values())
    types = Counter(e.type for e in ents if e.type)
    q = (params.get("q") or "").strip().lower()
    typ = params.get("type") or ""
    kind = params.get("kind") or ""
    flagged = params.get("flagged") == "1"
    sort = params.get("sort") if params.get("sort") in SORTS else "articles"
    rows = [entity_summary(store, e) for e in ents]
    rows = [
        r
        for r in rows
        if (not q or _matches(q, r["name"], *r["other_names"]))
        and (not typ or r["type"] == typ)
        and (not kind or r["kind"] == kind)
        and (
            not flagged
            or r["issues"].get("not_supported")
            or r["issues"].get("partial")
        )
    ]
    if sort == "name":
        rows.sort(key=lambda r: r["name"].lower())
    elif sort == "recent":
        rows.sort(key=lambda r: r["last_date"], reverse=True)
    else:
        rows.sort(key=lambda r: (-r["n_articles"], r["name"].lower()))
    return {
        "spec": spec,
        "rows": rows,
        "total": len(ents),
        "types": sorted(types.items(), key=lambda kv: -kv[1]),
        "kinds": KINDS,
        "sorts": SORTS,
        "f": {
            "q": params.get("q") or "",
            "type": typ,
            "kind": kind,
            "flagged": flagged,
            "sort": sort,
        },
    }


def sources_page(store: Store, params: Dict[str, str]) -> Dict[str, Any]:
    q = (params.get("q") or "").strip().lower()
    srcs = [s for s in store.sources.values() if s.id in store.by_source]
    if q:
        srcs = [s for s in srcs if q in s.title.lower()]
    srcs.sort(key=lambda s: s.date, reverse=True)
    return {
        "rows": [
            {"source": s, "n_entities": len(store.by_source.get(s.id, []))}
            for s in srcs
        ],
        "q": params.get("q") or "",
    }


__all__ = [
    "Source",
    "checker_note",
    "entity_page",
    "fmt_date",
    "home_page",
    "list_page",
    "pretty_type",
    "search",
    "source_page",
    "source_panel",
    "sources_page",
]
