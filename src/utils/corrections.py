"""A historian's corrections, kept in one append-only JSONL file.

Each line is one correction: a sentence struck (and maybe rewritten), a
checker verdict overruled, a sentence that belongs under another heading, a
page renamed or filed in the wrong table, a page that isn't an entity, sources
that belong to someone else (a split), or two pages that are one entity (a
join). Withdrawing a correction adds a ``withdraw`` line; nothing is ever
deleted or rewritten, so the file is a full record of what was corrected, by
whom, why, and when it was taken back.

The file sits outside every run folder (``corrections.path`` in the domain's
``config.yaml``, default ``data/<domain>/corrections/corrections.jsonl``;
``HINBOX_CORRECTIONS_PATH`` overrides both), so a fresh run into a new output
folder still finds it. It quotes sentences written from articles, so it is
article-derived data and must stay out of the public repository.

The reader (``src/frontend/``) appends to it and draws the corrections on
top of the page; the pipeline will read it at the start of each run.
"""

import json
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from src.utils.quality_controls import CITATION_RE

ENV_PATH = "HINBOX_CORRECTIONS_PATH"
ENV_REVIEWER = "HINBOX_REVIEWER"

# What each kind needs beyond the common fields (kind, table, entity)
KINDS: Dict[str, Tuple[str, ...]] = {
    "strike": ("sentence",),
    "verdict": ("sentence", "article_id", "verdict"),
    "heading": ("sentence", "heading"),
    "rename": ("name",),
    "wrong_table": ("to_table",),
    "not_entity": (),
    "split": ("article_ids",),
    "join": ("other",),
    "withdraw": ("withdraws",),
}
KIND_LABELS = {
    "strike": "Sentence struck",
    "verdict": "Checker overruled",
    "heading": "Wrong heading",
    "rename": "Renamed",
    "wrong_table": "Wrong table",
    "not_entity": "Not an entity",
    "split": "Sources belong to someone else",
    "join": "Same as another page",
    "withdraw": "Withdrawn",
}
VERDICTS = ("supported", "partial", "not_supported")
TABLES = ("people", "organizations", "locations", "events")
# Kinds that only a pipeline run can carry out (the reader just records them)
NEEDS_RUN = ("heading", "rename", "wrong_table", "not_entity", "split", "join")
MAX_TEXT = 2000

_ID = re.compile(r"^c-(\d+)$")
_LOCK = threading.Lock()


def corrections_path(domain: str, config: Optional[Dict[str, Any]] = None) -> str:
    """Where the domain's corrections file is."""
    override = os.getenv(ENV_PATH, "").strip()
    if override:
        return override
    configured = ((config or {}).get("corrections") or {}).get("path")
    if configured:
        return str(configured)
    return os.path.join("data", domain, "corrections", "corrections.jsonl")


def sentence_key(text: str) -> str:
    """A sentence as corrections compare it: no citation markers, single
    spaces. The same text whatever heading or line breaks surround it."""
    return " ".join(CITATION_RE.sub("", text or "").split()).strip()


def entity_ref(
    table: str, key: str, name: str, article_ids: Iterable[str]
) -> Dict[str, Any]:
    """How a correction names a page. The key is today's; the name and the
    linked articles let a later run find the same entity under a new key."""
    return {
        "table": table,
        "key": key,
        "name": name,
        "article_ids": sorted(set(article_ids)),
    }


def _bad_text(value: Any, what: str, required: bool = True) -> Optional[str]:
    if not isinstance(value, str) or (required and not value.strip()):
        return f"{what} is missing" if required else f"{what} must be text"
    if len(value) > MAX_TEXT:
        return f"{what} is longer than {MAX_TEXT} characters"
    return None


def _bad_entity(ref: Any, what: str) -> List[str]:
    if not isinstance(ref, dict):
        return [f"{what} is missing"]
    errs = []
    if ref.get("table") not in TABLES:
        errs.append(f"{what} has no known table")
    if not ref.get("key") or not isinstance(ref.get("key"), str):
        errs.append(f"{what} has no key")
    if not isinstance(ref.get("name", ""), str):
        errs.append(f"{what} has a name that isn't text")
    ids = ref.get("article_ids")
    if not isinstance(ids, list) or not all(isinstance(a, str) for a in ids):
        errs.append(f"{what} has no article list")
    return errs


def validate(rec: Dict[str, Any]) -> List[str]:
    """What is wrong with a correction (empty when it can be saved)."""
    kind = rec.get("kind")
    if not isinstance(kind, str) or kind not in KINDS:
        return [f"unknown kind {kind!r}"]
    errs = [f"{f} is missing" for f in KINDS[kind] if rec.get(f) in (None, "", [])]
    if errs:
        return errs
    if kind == "withdraw":
        ok = isinstance(rec["withdraws"], str) and _ID.match(rec["withdraws"])
        return [] if ok else ["withdraws must be a correction id"]
    for f in ("verdict", "article_id", "to_table", "replacement_cites"):
        if f in rec and not isinstance(rec[f], str):
            errs.append(f"{f} must be text")
    for f in ("article_ids", "cites"):
        ids = rec.get(f, [])
        if not isinstance(ids, list) or not all(isinstance(a, str) for a in ids):
            errs.append(f"{f} must be a list of article ids")
    if errs:
        return errs
    errs += _bad_entity(rec.get("entity"), "the page")
    for f in ("sentence", "heading", "name", "reason", "passage", "replacement"):
        if f in rec and (msg := _bad_text(rec[f], f, required=f in KINDS[kind])):
            errs.append(msg)
    if kind == "verdict" and rec["verdict"] not in VERDICTS:
        errs.append(f"verdict must be one of {', '.join(VERDICTS)}")
    if kind == "strike" and rec.get("replacement"):
        rep = rec["replacement"]
        if "\n" in rep or CITATION_RE.search(rep) or "^[" in rep:
            errs.append("the replacement must be one line without citation markers")
        if not rec.get("replacement_cites"):
            errs.append("the replacement needs a source to cite")
    if kind == "wrong_table" and rec["to_table"] not in TABLES:
        errs.append("to_table must be an entity table")
    if kind == "wrong_table" and rec["to_table"] == (rec.get("entity") or {}).get(
        "table"
    ):
        errs.append("the page is already in that table")
    if kind == "split":
        ids = set(rec["article_ids"])
        own = set((rec.get("entity") or {}).get("article_ids") or [])
        if not ids <= own:
            errs.append("a split may only tick the page's own sources")
        elif ids == own:
            errs.append("a split must leave at least one source on the page")
    if kind == "join":
        errs += _bad_entity(rec["other"], "the other page")
        a, b = rec.get("entity") or {}, rec["other"] or {}
        if (a.get("table"), a.get("key")) == (b.get("table"), b.get("key")):
            errs.append("a page can't be joined to itself")
        elif a.get("table") != b.get("table"):
            errs.append("only pages in the same table can be joined")
    return errs


@dataclass
class CorrectionSet:
    """Everything in the file, sorted out."""

    lines: List[Dict[str, Any]] = field(default_factory=list)  # in file order
    active: List[Dict[str, Any]] = field(default_factory=list)
    withdrawn: Dict[str, Dict[str, Any]] = field(
        default_factory=dict
    )  # id -> withdraw line
    replaced: Dict[str, str] = field(default_factory=dict)  # id -> id of the newer one
    problems: List[str] = field(default_factory=list)  # unreadable or invalid lines
    contradictions: List[str] = field(default_factory=list)

    def by_id(self) -> Dict[str, Dict[str, Any]]:
        return {r["id"]: r for r in self.lines if r.get("id")}

    def state(self, rec: Dict[str, Any]) -> str:
        """ "withdrawn", "replaced" or "active"."""
        if rec.get("id") in self.withdrawn:
            return "withdrawn"
        if rec.get("id") in self.replaced:
            return "replaced"
        return "active"

    def for_entity(self, table: str, key: str) -> List[Dict[str, Any]]:
        """Active corrections pointing at this page (a join from either side)."""

        def hits(ref: Any) -> bool:
            return (
                isinstance(ref, dict)
                and ref.get("table") == table
                and ref.get("key") == key
            )

        return [r for r in self.active if hits(r.get("entity")) or hits(r.get("other"))]


def _target(rec: Dict[str, Any]) -> Optional[Tuple[str, ...]]:
    """What a correction is about. A newer correction of the same thing
    replaces the older one (it is kept and shown as replaced); splits and
    joins never replace each other."""
    ent = rec.get("entity") or {}
    page = (str(ent.get("table")), str(ent.get("key")))
    kind = rec["kind"]
    if kind in ("strike", "heading"):
        return (kind, *page, sentence_key(rec["sentence"]))
    if kind == "verdict":
        return (kind, *page, sentence_key(rec["sentence"]), str(rec["article_id"]))
    if kind in ("rename", "wrong_table", "not_entity"):
        return ("page", *page) if kind != "rename" else (kind, *page)
    return None


def _contradictions(active: List[Dict[str, Any]]) -> List[str]:
    """Splits and joins that can't all hold. Each page's linked articles are
    its mentions; a split cuts its ticked articles from the rest, a join puts
    two pages together. If a join chain puts a mention back beside one it was
    split from, the lines conflict and a person has to choose."""
    parent: Dict[Any, Any] = {}

    def find(x: Any) -> Any:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: Any, b: Any) -> None:
        parent[find(a)] = find(b)

    splits: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    pages: Dict[Tuple[str, str], Set[str]] = {}
    for r in active:
        for ref in (r.get("entity"), r.get("other")):
            if r["kind"] in ("split", "join") and isinstance(ref, dict):
                page = (ref["table"], ref["key"])
                pages.setdefault(page, set()).update(ref.get("article_ids") or [])
        if r["kind"] == "split":
            e = r["entity"]
            splits.setdefault((e["table"], e["key"]), []).append(r)
    # Within a page, articles with the same split memberships stay together
    for page, arts in pages.items():
        groups: Dict[Tuple[bool, ...], List[str]] = {}
        for aid in sorted(arts):
            sig = tuple(aid in set(s["article_ids"]) for s in splits.get(page, []))
            groups.setdefault(sig, []).append(aid)
        for members in groups.values():
            for aid in members[1:]:
                union((page, members[0]), (page, aid))
    for r in active:
        if r["kind"] == "join":
            a, b = r["entity"], r["other"]
            pa, pb = (a["table"], a["key"]), (b["table"], b["key"])
            for x in sorted(pages.get(pa, ())):
                for y in sorted(pages.get(pb, ())):
                    union((pa, x), (pb, y))
    out = []
    for page, rs in splits.items():
        for s in rs:
            ticked = set(s["article_ids"])
            rest = pages[page] - ticked
            if any(find((page, x)) == find((page, y)) for x in ticked for y in rest):
                joins = [r["id"] for r in active if r["kind"] == "join"]
                out.append(
                    f"{s['id']} splits sources off {s['entity'].get('name') or page[1]}, "
                    f"but joins ({', '.join(joins)}) put them back together"
                )
    return out


def load_corrections(path: str) -> CorrectionSet:
    """Read the file. A line that can't be read or doesn't make sense is
    listed under ``problems``, never dropped quietly."""
    cs = CorrectionSet()
    try:
        with open(path, encoding="utf-8") as f:
            raw = f.read().splitlines()
    except FileNotFoundError:
        return cs
    for n, line in enumerate(raw, 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            cs.problems.append(f"line {n}: not JSON ({e.msg})")
            continue
        if not isinstance(rec, dict) or not _ID.match(str(rec.get("id", ""))):
            cs.problems.append(f"line {n}: no correction id")
            continue
        try:
            errs = validate(rec)
        except Exception as e:  # a hand-edited line of an unexpected shape
            errs = [f"can't be read ({type(e).__name__})"]
        if errs:
            cs.problems.append(f"line {n} ({rec['id']}): {'; '.join(errs)}")
            continue
        cs.lines.append(rec)
    ids = {r["id"] for r in cs.lines}
    for r in cs.lines:
        if r["kind"] == "withdraw":
            if r["withdraws"] in ids:
                cs.withdrawn[r["withdraws"]] = r
            else:
                cs.problems.append(
                    f"{r['id']} withdraws {r['withdraws']}, which isn't in the file"
                )
    latest: Dict[Tuple[str, ...], str] = {}
    for r in cs.lines:
        if r["kind"] == "withdraw" or r["id"] in cs.withdrawn:
            continue
        t = _target(r)
        if t is not None:
            if t in latest:
                cs.replaced[latest[t]] = r["id"]
            latest[t] = r["id"]
    cs.active = [
        r for r in cs.lines if r["kind"] != "withdraw" and cs.state(r) == "active"
    ]
    cs.contradictions = _contradictions(cs.active)
    return cs


def _next_id(path: str) -> str:
    top = 0
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    m = _ID.match(str(json.loads(line).get("id", "")))
                except (json.JSONDecodeError, AttributeError):
                    continue
                if m:
                    top = max(top, int(m.group(1)))
    except FileNotFoundError:
        pass
    return f"c-{top + 1:04d}"


def _newline_needed(path: str) -> str:
    """ "\n" when the file's last line has no line end (a hand edit), so a
    new line never runs on into it and spoils both."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            if f.tell() == 0:
                return ""
            f.seek(-1, os.SEEK_END)
            return "" if f.read(1) == b"\n" else "\n"
    except FileNotFoundError:
        return ""


def append_correction(path: str, rec: Dict[str, Any]) -> Dict[str, Any]:
    """Give the correction an id, a time and the reviewer's name, check it,
    and add it as one line at the end of the file (written to disk before
    returning). Raises ``ValueError`` listing what is wrong."""
    rec = {k: v for k, v in rec.items() if v not in (None, "", [])}
    errs = validate(rec)
    if errs:
        raise ValueError("; ".join(errs))
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with _LOCK:
        out = {
            "id": _next_id(path),
            "made": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            **(
                {"by": os.environ[ENV_REVIEWER].strip()}
                if os.getenv(ENV_REVIEWER, "").strip()
                else {}
            ),
            **rec,
        }
        line = json.dumps(out, ensure_ascii=False, sort_keys=False)
        with open(path, "a", encoding="utf-8") as f:
            f.write(_newline_needed(path) + line + "\n")
            f.flush()
            os.fsync(f.fileno())
    return out
