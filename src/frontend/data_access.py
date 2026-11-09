"""Load a domain's entity tables and sources, and index them for the pages.

``load_store(domain)`` reads the four entity Parquet tables from the domain's
output folder and the source articles from its data path, and keeps them in
memory until a file changes (checked by modification time on each request,
so a finished pipeline run shows up on the next page load).

Entity tables are listed once in ``TABLES``; navigation, search and the
"appears with" lists loop over it, so a fifth table (court cases, say) is one
new ``TableSpec``.
"""

import hashlib
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import pyarrow.parquet as pq

from src.config_loader import DomainConfig
from src.constants import HASH_TRUNCATE_LENGTH


def _hashed_key(name_field: str, other_field: str) -> Callable[[Dict[str, Any]], str]:
    def key(row: Dict[str, Any]) -> str:
        name = row.get(name_field) or ""
        other = row.get(other_field) or ""
        h = hashlib.md5(f"{name}##{other}".encode()).hexdigest()[:HASH_TRUNCATE_LENGTH]
        return f"{name} ({h})"

    return key


def make_person_key(person: Dict[str, Any]) -> str:
    return person.get("name", "") or ""


make_event_key = _hashed_key("title", "start_date")
make_location_key = _hashed_key("name", "type")
make_org_key = _hashed_key("name", "type")


@dataclass(frozen=True)
class TableSpec:
    """How one entity table is named, keyed and displayed."""

    table: str  # Parquet file stem and URL prefix
    label: str  # "People"
    singular: str  # "person"
    name_field: str
    type_field: str
    alt_field: str  # other names
    key: Callable[[Dict[str, Any]], str]


TABLES: List[TableSpec] = [
    TableSpec("people", "People", "person", "name", "type", "alternative_names", make_person_key),
    TableSpec("organizations", "Organizations", "organization", "name", "type", "alternative_names", make_org_key),
    TableSpec("locations", "Places", "place", "name", "type", "alternative_names", make_location_key),
    TableSpec("events", "Events", "event", "title", "event_type", "alternative_titles", make_event_key),
]  # fmt: skip
SPECS: Dict[str, TableSpec] = {t.table: t for t in TABLES}


def to_date(value: Any) -> str:
    """Any stored date (epoch seconds as int or str, ISO text, datetime) as
    ``YYYY-MM-DD``; "" when unknown."""
    if value is None or value == "":
        return ""
    if isinstance(value, datetime):
        return value.date().isoformat()
    text = str(value).strip()
    if text.lstrip("-").isdigit():
        try:
            return datetime.fromtimestamp(int(text), tz=timezone.utc).date().isoformat()
        except (OverflowError, OSError, ValueError):
            return ""
    return text[:10] if len(text) >= 10 and text[4] == "-" else text


def pretty_type(value: Any) -> str:
    """ "EventType.DETENTION" -> "Detention"; "military" -> "Military"."""
    text = str(value or "").split(".")[-1].replace("_", " ").strip()
    return text[:1].upper() + text[1:].lower() if text else ""


@dataclass
class Source:
    """A source document. Today every source is a news article; ``kind``
    and ``locator`` leave room for books, reports and translations."""

    id: str
    title: str
    url: str = ""
    date: str = ""
    outlet: str = ""
    kind: str = "News article"
    text: str = ""

    def paragraphs(self) -> List[str]:
        parts = [p.strip() for p in self.text.replace("\r", "").split("\n")]
        return [p for p in parts if p]


@dataclass
class Entity:
    spec: TableSpec
    key: str
    row: Dict[str, Any]

    @property
    def name(self) -> str:
        return str(self.row.get(self.spec.name_field) or "Untitled")

    @property
    def type(self) -> str:
        return pretty_type(self.row.get(self.spec.type_field))

    @property
    def raw_type(self) -> str:
        return str(self.row.get(self.spec.type_field) or "").split(".")[-1].lower()

    @property
    def profile(self) -> Dict[str, Any]:
        return self.row.get("profile") or {}

    @property
    def articles(self) -> List[Dict[str, Any]]:
        return [a for a in (self.row.get("articles") or []) if a.get("article_id")]

    @property
    def other_names(self) -> List[str]:
        names: List[str] = []
        for n in list(self.row.get(self.spec.alt_field) or []) + list(
            self.row.get("aliases") or []
        ):
            # Alternative names are stored as {"name"/"title": ..., "type": ...}
            if isinstance(n, dict):
                n = n.get("name") or n.get("title") or ""
            n = str(n).strip()
            if n and n != self.name and n not in names:
                names.append(n)
        return names

    @property
    def url(self) -> str:
        from urllib.parse import quote

        return f"/{self.spec.table}/{quote(self.key, safe='')}"


@dataclass
class Store:
    """Everything the pages read for one domain."""

    domain: str
    output_dir: str
    entities: Dict[str, Dict[str, Entity]] = field(default_factory=dict)
    sources: Dict[str, Source] = field(default_factory=dict)
    # source id -> the entities linked to it
    by_source: Dict[str, List[Entity]] = field(default_factory=dict)
    link_only: Dict[str, List[str]] = field(default_factory=dict)
    texts_loaded: bool = False

    def get(self, table: str, key: str) -> Optional[Entity]:
        return self.entities.get(table, {}).get(key)

    def all(self) -> List[Entity]:
        return [e for t in TABLES for e in self.entities.get(t.table, {}).values()]

    def source(self, article_id: str) -> Source:
        return self.sources.get(article_id) or Source(article_id, "Unknown source")

    def kind_of(self, entity: Entity) -> str:
        """ "stub", "link-only" or "full"."""
        p = entity.profile
        # A hub (country, region, news outlet) is frozen after its first
        # article, often while still a stub, and never gets more text
        hub_types = self.link_only.get(entity.spec.table, [])
        if p.get("linked_only") or (entity.raw_type and entity.raw_type in hub_types):
            return "link-only"
        if p.get("stub"):
            return "stub"
        return "full"

    def review_queue_count(self) -> int:
        path = os.path.join(self.output_dir, "merge_review_queue.jsonl")
        try:
            with open(path, encoding="utf-8") as f:
                return sum(1 for line in f if line.strip())
        except OSError:
            return 0


def _read(path: str, columns: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    table = pq.read_table(path)
    if columns:
        table = table.select([c for c in columns if c in table.column_names])
    return table.to_pylist()


def _paths(domain: str) -> Tuple[str, str, Dict[str, Any]]:
    config = DomainConfig(domain)
    cfg = config.load_config()
    output = config.get_output_dir()
    data = os.getenv("HINBOX_ARTICLES_PATH", "").strip() or config.get_data_path()
    return output, data, cfg


_DROP = (
    "profile_embedding",
    "search_embedding",
    "reflection_history",
)


def build_store(domain: str) -> Store:
    output, data_path, cfg = _paths(domain)
    store = Store(domain, output)
    if os.getenv("HINBOX_HUB_LINK_ONLY", "on").strip().lower() != "off":
        listed = (cfg.get("profiles") or {}).get("link_only") or {}
        store.link_only = {
            k: [str(x).lower() for x in v or []] for k, v in listed.items()
        }
    for spec in TABLES:
        rows = _read(os.path.join(output, f"{spec.table}.parquet"))
        index: Dict[str, Entity] = {}
        for row in rows:
            for col in _DROP:
                row.pop(col, None)
            ent = Entity(spec, spec.key(row), row)
            index[ent.key] = ent
            for a in ent.articles:
                aid = a["article_id"]
                store.by_source.setdefault(aid, []).append(ent)
                if aid not in store.sources:
                    store.sources[aid] = Source(
                        aid,
                        a.get("article_title") or "Untitled",
                        a.get("article_url") or "",
                        to_date(a.get("article_published_date")),
                    )
        store.entities[spec.table] = index
    cols = ["id", "title", "url", "published_date", "content", "author", "source"]
    for row in _read(data_path, cols):
        aid = str(row.get("id") or "")
        if not aid:
            continue
        old = store.sources.get(aid)
        store.sources[aid] = Source(
            aid,
            row.get("title") or (old.title if old else "Untitled"),
            row.get("url") or (old.url if old else ""),
            to_date(row.get("published_date")) or (old.date if old else ""),
            text=row.get("content") or "",
        )
        store.texts_loaded = True
    for src in store.sources.values():
        src.outlet = _outlet(src.url)
    return store


def _outlet(url: str) -> str:
    from urllib.parse import urlparse

    host = urlparse(url).netloc.lower().removeprefix("www.")
    known = {"miamiherald.com": "Miami Herald", "nytimes.com": "New York Times"}
    return known.get(host, host)


_CACHE: Dict[str, Tuple[Tuple[float, ...], Store]] = {}
_LOCK = threading.Lock()


def _stamp(domain: str) -> Tuple[float, ...]:
    output, data_path, _ = _paths(domain)
    files = [os.path.join(output, f"{t.table}.parquet") for t in TABLES] + [data_path]
    return tuple(os.path.getmtime(f) if os.path.exists(f) else 0.0 for f in files) + (
        hash(output),
    )


def load_store(domain: str = "guantanamo") -> Store:
    """The domain's store, rebuilt when any of its files changed."""
    stamp = _stamp(domain)
    with _LOCK:
        hit = _CACHE.get(domain)
        if hit and hit[0] == stamp:
            return hit[1]
        store = build_store(domain)
        _CACHE[domain] = (stamp, store)
        return store


def available_domains() -> List[str]:
    try:
        return DomainConfig.get_available_domains()
    except Exception:
        return []
