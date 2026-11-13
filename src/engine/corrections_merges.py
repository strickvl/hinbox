"""Carry out a historian's splits and joins (step 3 of the corrections design).

A **join** says two pages are one entity. At the start of a run the page with
fewer sources is fed into the one with more, article by article, with the
pipeline's own profile update (``join_pair()`` in ``src/engine/kind_twins.py``,
which also joins pages a run saved twice); a country, region or outlet is only
linked. The smaller page's name becomes another name of the kept page. During
merging, a mention spelled like the page that is gone goes straight to the
kept page, with no match check (``join_target()``).

A **split** says some of a page's sources are about someone else. At the start
of a run those sources leave the page: each sentence citing only them is cut,
a sentence citing them and other sources keeps only the other citations (and
is listed in the report for a person to check, since its words may come from
the sources that left), and a new page is built from the ticked sources the way
a run builds any page: a stub from the oldest, then a profile update for each
further one. During merging, a mention from a ticked source is never merged
into a page holding the page's other sources, and the other way round
(``split_block()``): the candidate is passed over before any model call. On a
fresh run into an empty folder that rule alone keeps them apart.

Splits and joins that contradict each other (``CorrectionSet.contradictions``)
are not carried out; each is reported as needing a look. Withdrawing a split
or join does not undo it; the merge rules simply stop applying.
"""

import copy
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from src.engine.alternative_names import add_alternative_name
from src.engine.corrections_pass import (
    _candidates,
    _cut_claim,
    _linked,
    _new_version,
    active_corrections,
    joined_refs,
)
from src.engine.entity_rows import new_entity_row
from src.engine.kind_twins import _rank, alt_field, join_pair
from src.engine.profile_stubs import first_profile, update_or_fill
from src.engine.profiles import create_profile
from src.logging_config import log
from src.utils.claim_spans import parse_claims
from src.utils.corrections import MERGE_KINDS, CorrectionSet, resolve_ref
from src.utils.entity_keys import NAME_FIELDS
from src.utils.profiles import extract_profile_text
from src.utils.task_models import task_model

Pages = Dict[Any, Dict[str, Any]]


def _usable(cs: Optional[CorrectionSet], kind: str, table: str) -> List[Dict]:
    if not cs or cs.contradictions:
        return []
    return [
        r
        for r in cs.active
        if r["kind"] == kind and (r.get("entity") or {}).get("table") == table
    ]


def _name(table: str, row: Dict[str, Any]) -> str:
    return row.get(NAME_FIELDS[table]) or ""


def split_names(rec: Dict[str, Any]) -> Set[str]:
    """The names a split is about: the page's, and the new page's."""
    return {n for n in ((rec.get("entity") or {}).get("name"), rec.get("name")) if n}


def _family(rec: Dict[str, Any], rules: List[Dict]) -> Set[str]:
    """The names of every split of the same page as ``rec``. Two splits of
    "Ann Lee" (into "Ann Lee (reporter)", then one of those sources into
    "Ann Lee (trader)") are about all three pages: the second has to find its
    sources on the page the first one made. So are a split of "Ann Lee" and a
    later split of the page it made."""
    names, keys = split_names(rec), {str((rec.get("entity") or {}).get("key"))}
    left = list(rules)
    while True:  # a split of a page that a split made belongs too
        more = [
            r
            for r in left
            if split_names(r) & names or str(r["entity"].get("key")) in keys
        ]
        if not more:
            break
        for r in more:
            names |= split_names(r)
            keys.add(str(r["entity"].get("key")))
        left = [r for r in left if r not in more]
    return names


def split_family(ref: Any, table: str) -> Set[str]:
    """Names of the pages the usable splits made of this page (and the page's
    own), empty when no split is about it. A strike made on "Ann Lee" may be
    about "Ann Lee (reporter)" once its sources moved there."""
    rules = _usable(active_corrections(), "split", table)
    if not isinstance(ref, dict) or ref.get("table") != table or not rules:
        return set()
    names = _family({"entity": ref}, rules)
    return names if len(names) > 1 else set()


def _find(table: str, ref: Any, pages: Pages) -> Any:
    """The page a join's side names, else the page other joins put it into:
    with "A is B" and "A is C" both waiting, "A" is gone after the first, and
    the second joins B and C."""
    for r in (ref, *joined_refs(ref, active_corrections())):
        pk = resolve_ref(r, _candidates(table, pages))
        if pk is not None:
            return pk
    return None


def _base(name: str) -> str:
    return name.split(" (")[0]  # "Ann Lee (2)" is a page of "Ann Lee"


def _sides(rec: Dict[str, Any]) -> Tuple[Set[str], Set[str]]:
    ticked = set(rec["article_ids"])
    return ticked, set(rec["entity"].get("article_ids") or []) - ticked


# ── during merging ───────────────────────────────────────────


def split_block(
    table: str, article_id: str, incoming: str, candidate: Dict[str, Any]
) -> Optional[str]:
    """The split that forbids merging this mention into ``candidate``, if any.

    Example: a historian split "Ann Lee" because two of her twelve sources are
    about another Ann Lee. A mention of "Ann Lee" from one of those two is
    never merged into a page holding any of the other ten, and the other way
    round. Only pages and mentions under the split's names count, so the
    places those articles mention still merge as usual.
    """
    rules = _usable(active_corrections(), "split", table)
    if not rules:
        return None
    arts = set(_linked(candidate))
    for r in rules:
        names = _family(r, rules)
        if incoming not in names and _name(table, candidate) not in names:
            continue
        ticked, rest = _sides(r)
        if (article_id in ticked and arts & rest) or (
            article_id in rest and arts & ticked
        ):
            return r["id"]
    return None


def split_create_key(table: str, article_id: str, key: Any, pages: Pages) -> Any:
    """A free key for a mention that a split kept off the page holding ``key``
    (same name): the split's name for the ticked side, the page's own for the
    rest, numbered when taken ("Ann Lee (2)"). None when no split applies."""
    incoming = key[0] if isinstance(key, tuple) else key
    page = pages.get(key)
    if page is None or not split_block(table, article_id, incoming, page):
        return None
    for r in _usable(active_corrections(), "split", table):
        ticked, rest = _sides(r)
        if incoming in split_names(r) and article_id in ticked | rest:
            name = r["entity"].get("name") or incoming
            if article_id in ticked:
                name = r.get("name") or name
            return _free_key(table, name, key, pages)
    return None


def split_home(
    table: str, article_id: str, incoming: str, pages: Pages
) -> Optional[Tuple]:
    """(key, correction id) of the page on this mention's own side of a split.

    The historian said which side each of the split's sources is on, so a
    mention under the split's name from one of them goes to the one page that
    holds sources of that side and none of the other, with no match check.
    Example: "Ann Lee" from a ticked article goes to "Ann Lee (reporter)",
    which holds the other ticked article; without this the match check could
    turn it away and a third page, "Ann Lee (reporter) (2)", was created.
    With several splits of one page, the home holds no source from the other
    side of any of them.
    """
    rules = _usable(active_corrections(), "split", table)
    apply = [
        r
        for r in rules
        if incoming in _family(r, rules) and article_id in set().union(*_sides(r))
    ]
    if not apply:
        return None
    bases = {_base(n) for r in apply for n in _family(r, rules)}
    homes = []
    for pk, e in pages.items():
        if _base(_name(table, e)) not in bases:
            continue
        arts, own = set(_linked(e)), False
        for r in apply:
            ticked, rest = _sides(r)
            mine, other = (ticked, rest) if article_id in ticked else (rest, ticked)
            if arts & other:
                break
            own = own or bool(arts & mine)
        else:
            if own:
                homes.append(pk)
    return (homes[0], apply[-1]["id"]) if len(homes) == 1 else None


def forced_target(
    table: str, article_id: str, incoming: str, pages: Pages
) -> Optional[Tuple]:
    """Where a split or join sends this mention without a match check."""
    return split_home(table, article_id, incoming, pages) or join_target(
        table, incoming, pages
    )


def join_target(table: str, incoming: str, pages: Pages) -> Optional[Tuple]:
    """(key, correction id) of the page a join put this mention's name into.

    Only for a name no page carries any more (the joined-in page's): "J. Lee"
    after a historian joined "J. Lee" into "Jane Lee" goes to "Jane Lee"
    without a match check. A mention named like the kept page goes the usual
    way, so a third, different Jane Lee can still be told apart.
    """
    rules = [
        r
        for r in _usable(active_corrections(), "join", table)
        if incoming in (r["entity"].get("name"), r["other"].get("name"))
    ]
    if not rules or any(_name(table, p) == incoming for p in pages.values()):
        return None
    for r in rules:
        found = {
            pk
            for ref in (r["entity"], r["other"])
            if (pk := _find(table, ref, pages)) is not None
        }
        if len(found) == 1:
            return found.pop(), r["id"]
    return None


# ── at the start of a run ────────────────────────────────────


def _key(table: str, name: str, like: Dict[str, Any]) -> Any:
    if table == "people":
        return name
    if table == "events":
        return (name, like.get("start_date", ""))
    return (name, like.get("type", ""))


def _free_key(table: str, name: str, like: Any, pages: Pages) -> Any:
    row = like if isinstance(like, dict) else _row_like(table, like)
    key, n = _key(table, name, row), 2
    while key in pages:
        key, n = _key(table, f"{name} ({n})", row), n + 1
    return key


def _row_like(table: str, key: Any) -> Dict[str, Any]:
    if not isinstance(key, tuple):
        return {}
    field = "start_date" if table == "events" else "type"
    return {field: key[1] if len(key) > 1 else ""}


def drop_sources(text: str, gone: Set[str]) -> Tuple[str, List[str], List[str]]:
    """``text`` without the articles in ``gone``: (text, sentences cut, sentences
    kept with only their other citations). A cut takes the sentence's words and
    its markers; a kept sentence loses only the markers' gone ids."""
    claims, markers = parse_claims(text)
    mixed = {
        c.index for c in claims if set(c.articles) & gone and set(c.articles) - gone
    }
    kept = [c.text for c in claims if c.index in mixed]
    for m in reversed(markers):
        if m.claim in mixed:
            ids = [a for a in m.articles if a not in gone]
            if ids != m.articles:
                put = f"^[{', '.join(ids)}]" if ids else ""
                text = text[: m.start] + put + text[m.end :]
    claims, markers = parse_claims(text)
    cut = [c for c in claims if c.articles and set(c.articles) <= gone]
    for c in reversed(cut):
        text = _cut_claim(text, markers, c)
    return text, [c.text for c in cut], kept


def _by_date(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(entries, key=lambda a: str(a.get("article_published_date") or ""))


def _build_page(
    table: str,
    like: Dict[str, Any],
    key: Any,
    entries: List[Dict[str, Any]],
    texts: Dict[str, str],
    model_type: str,
    domain: str,
    update: Callable,
    said: str = "",
) -> Tuple[Optional[Dict[str, Any]], str]:
    """A page built from ``entries`` as a run builds one: a stub (or with stubs
    off, a written profile) from the oldest article, then one profile update
    per further article. (row, "") or (None, why not)."""
    entries = _by_date(entries)
    first = entries[0]
    text = texts.get(first["article_id"])
    if not text:
        return None, f"no text for article {first['article_id']}"
    alt = alt_field(table)
    entity = {
        "type": like.get("type", ""),
        **{f: like.get(f, "") for f in ("event_type", "end_date") if table == "events"},
    }  # fmt: skip
    # The stub looks for the name the articles use (``said``: "Ann Lee", not
    # the page's "Ann Lee (reporter)")
    profile, versioned, history = first_profile(
        table,
        _key(table, said or _name(table, like), like),
        entity,
        text,
        first["article_id"],
        model_type,
        domain,
        alternative_field=alt,
        model=task_model("profile", model_type),
        create=create_profile,
    )
    profile = extract_profile_text(profile)
    if not profile or not profile.get("text"):
        return None, f"no profile could be written from {first['article_id']}"
    row = new_entity_row(
        table,
        alt,
        entity,
        key,
        profile,
        versioned,
        [],
        history,
        first["article_id"],
        first.get("article_title", ""),
        first.get("article_url", ""),
        first.get("article_published_date"),
        like.get("extraction_timestamp", ""),
    )
    if len(entries) > 1:
        row, report = join_pair(
            table, row, {"articles": entries[1:]}, texts, model_type, domain, update
        )
        if report.get("failed"):
            return None, report["failed"]
    return row, ""


def _embed_rows(rows: List[Dict[str, Any]], domain: str) -> None:
    from src.engine.merge_evidence import batch_embed_texts
    from src.engine.profile_view import embedding_text
    from src.utils.embeddings.similarity import get_embedding_manager

    texts = [embedding_text(r["profile"]["text"]) for r in rows]
    vecs, model, dim, fp = batch_embed_texts(
        get_embedding_manager(domain=domain), texts, max(len(texts), 1), "embed.profile"
    )
    for row, vec in zip(rows, vecs):
        row.update(
            profile_embedding=vec,
            profile_embedding_model=model,
            profile_embedding_dim=dim or len(vec or []),
            profile_embedding_fingerprint=fp,
        )


class _Run:
    def __init__(self, entities, texts, model_type, domain, done, update, embed):
        self.entities, self.texts = entities, texts
        self.model_type, self.domain, self.done = model_type, domain, done
        self.update, self.embed = update, embed
        self.mixed: List[Dict[str, str]] = []
        self.changed = 0  # pages rewritten, for the report's profiles_changed

    def state(self, rec: Dict[str, Any]) -> str:
        return "re-applied" if rec["id"] in self.done else "applied"

    def join(self, rec: Dict[str, Any]) -> Tuple[str, str]:
        table = rec["entity"]["table"]
        pages = self.entities.get(table, {})
        a = _find(table, rec["entity"], pages)
        b = _find(table, rec["other"], pages)
        if a is None and b is None:
            return "needs a look", "neither page can be found"
        if a is None or b is None or a == b:
            found = pages[a if a is not None else b]
            gone = rec["entity"] if a is None else rec["other"]
            if a == b or set(gone.get("article_ids") or []) <= set(_linked(found)):
                return "held", ""
            return "needs a look", f"“{gone.get('name')}” can't be found"
        keep_k, give_k = sorted((a, b), key=lambda k: _rank(pages[k]), reverse=True)
        keeper, donor = pages[keep_k], pages[give_k]
        work, report = join_pair(
            table, keeper, donor, self.texts, self.model_type, self.domain, self.update
        )
        if report.get("failed"):
            # Keep what went in, so the next run asks only for the rest
            if report["updated"]:
                self.embed([work], self.domain)
                pages[keep_k] = work
                self.changed += 1
            return (
                "waiting",
                f"not joined yet: {report['updated']} articles fed in, the rest "
                f"failed ({report['failed']}); the next run tries those again",
            )
        add_alternative_name(table, alt_field(table), work, give_k, donor)
        for alias in donor.get("aliases") or []:
            if alias not in (work.setdefault("aliases", [])):
                work["aliases"].append(alias)
        self.embed([work], self.domain)
        pages[keep_k] = work
        del pages[give_k]
        self.changed += 1
        why = (
            f"“{_name(table, donor)}” joined into “{_name(table, work)}”, "
            f"{report['updated']} articles fed in"
        )
        log(f"Corrections: {why} ({rec['id']})", level="success")
        return self.state(rec), why

    def split(self, rec: Dict[str, Any]) -> Tuple[str, str]:
        table = rec["entity"]["table"]
        pages = self.entities.get(table, {})
        ticked, rest = _sides(rec)
        names = _family(rec, _usable(active_corrections(), "split", table))
        bases = {_base(n) for n in names}
        hit = resolve_ref(rec["entity"], _candidates(table, pages))
        both = [
            pk
            for pk, e in pages.items()
            if (pk == hit or _base(_name(table, e)) in bases)
            and set(_linked(e)) & ticked
            and set(_linked(e)) & rest
        ]
        if not both:
            if hit is None and not any(
                _name(table, e) in names for e in pages.values()
            ):
                return "needs a look", "the page can't be found"
            return "held", "" if rec[
                "id"
            ] in self.done else "the sources are already on separate pages"
        whys = []
        for pk in both:
            why = self._split_page(rec, table, pages, pk, ticked)
            if why.startswith("not split"):
                return "waiting", why
            whys.append(why)
        return self.state(rec), "; ".join(whys)

    def _split_page(
        self, rec, table: str, pages: Pages, pk: Any, ticked: Set[str]
    ) -> str:
        page = pages[pk]
        moved = [a for a in page.get("articles") or [] if a.get("article_id") in ticked]
        kept = [
            a for a in page.get("articles") or [] if a.get("article_id") not in ticked
        ]
        name = _name(table, page)
        # The name the articles use: the split's page, not a page an earlier
        # split of it named ("Ann Lee", not "Ann Lee (reporter)")
        said = rec["entity"].get("name") or name
        new_key = _free_key(table, rec.get("name") or said, page, pages)
        args = (self.texts, self.model_type, self.domain, self.update)
        new, why = _build_page(table, page, new_key, moved, *args, said=said)
        if new is None:
            return f"not split yet ({why}); the next run tries again"
        profile = page.get("profile") or {}
        text, cut, mixed = drop_sources(profile.get("text") or "", ticked)
        if parse_claims(text)[0]:
            rest = copy.deepcopy(page)
            rest["articles"] = kept
            srcs = [s for s in profile.get("sources") or [] if s not in ticked]
            rest["profile"] = {**rest["profile"], "text": text, "sources": srcs}
            _new_version(rest, [rec["id"]])
        else:  # nothing cited is left: write the page again from its own sources
            rest, why = _build_page(table, page, pk, kept, *args, said=said)
            if rest is None:
                return f"not split yet ({why}); the next run tries again"
            for f in ("aliases", alt_field(table)):
                rest[f] = list(page.get(f) or [])
        self.embed([rest, new], self.domain)
        pages[pk], pages[new_key] = rest, new
        self.changed += 2
        shown = new_key[0] if isinstance(new_key, tuple) else new_key
        for s in mixed:
            self.mixed.append(
                {"id": rec["id"], "page": f"{table}/{name}", "sentence": s[:300]}
            )
        why = (
            f"{len(moved)} sources moved to “{shown}”; {len(cut)} sentences citing "
            f"only them cut from “{name}”"
        )
        if mixed:
            why += f"; {len(mixed)} sentences cite both and kept only the other sources"
        log(f"Corrections: {why} ({rec['id']})", level="success")
        return why


def carry_out_merges(
    entities: Dict[str, Pages],
    texts: Dict[str, str],
    model_type: str,
    domain: str,
    done_before: Set[str],
    *,
    update: Callable = update_or_fill,
    embed: Callable = _embed_rows,
) -> Dict[str, Any]:
    """Carry out every active split and join in the loaded tables, before
    merging. Returns ``{"states": {id: (state, why)}, "mixed": [...],
    "changed": pages rewritten}`` for the
    run's corrections report (``run_corrections_pass(early=...)``)."""
    cs = active_corrections()
    out: Dict[str, Any] = {"states": {}, "mixed": [], "changed": 0}
    if not cs:
        return out
    recs = [r for r in cs.active if r["kind"] in MERGE_KINDS]
    if cs.contradictions:
        why = "contradicts another split or join: " + "; ".join(cs.contradictions)
        out["states"] = {r["id"]: ("needs a look", why) for r in recs}
        return out
    run = _Run(entities, texts, model_type, domain, done_before, update, embed)
    for r in recs:
        try:
            out["states"][r["id"]] = (
                run.join(r) if r["kind"] == "join" else run.split(r)
            )
        except Exception as e:  # one failed step must not stop the run
            log(f"Corrections: {r['id']} failed: {e}", level="error")
            out["states"][r["id"]] = (
                "waiting",
                f"failed ({type(e).__name__}); the next run tries again",
            )
    out["mixed"], out["changed"] = run.mixed, run.changed
    return out
