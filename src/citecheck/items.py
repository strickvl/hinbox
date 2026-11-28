"""Cited sentences of a finished run, article splits and the test sample.

A "claim" here is what the citation check judges: one profile sentence (or
run of sentences) and one article it cites, cut from the text exactly as
``_extract_cited_claims()`` cuts it. A sentence citing two articles is two
claims.

Articles, not claims, are split. If one article gave claims to both the
training data and the test set, a tuned model could pass the test by
remembering that article, and the score would say nothing about a new
archive. So every claim from an article goes where the article goes:
``article_split()`` hashes the article id into train (75%), tune (10%) or
test (15%), and the same id always lands in the same place.
"""

import json
import os
import random
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.utils.cache_utils import sha256_text
from src.utils.corrections import sentence_key
from src.utils.grounding import GROUNDING_MAX_CLAIM_CHARS, _extract_cited_claims

SPLIT_BOUNDS: Tuple[Tuple[str, int], ...] = (("test", 15), ("tune", 25), ("train", 100))
# Verdicts that make a stored claim "doubted": the half of the test set drawn
# from these holds most of the wrong claims, so models can be told apart
DOUBTED = ("partial", "not_supported", "unclear")
# At most this many test claims per article and per entity, so a few long
# pages or articles don't fill the sample
PER_ARTICLE = 3
PER_ENTITY = 3


def article_split(article_id: str) -> str:
    """'train', 'tune' or 'test' for an article, fixed by its id."""
    bucket = int(sha256_text(f"citecheck:{article_id}")[:8], 16) % 100
    return next(name for name, bound in SPLIT_BOUNDS if bucket < bound)


def item_id(entity: str, article_id: str, claim: str) -> str:
    return sha256_text(f"{entity}|{article_id}|{sentence_key(claim)}")[:16]


def _name(key: Any, data: Dict[str, Any]) -> str:
    name = data.get("name") or data.get("title")
    if not name:
        name = key[0] if isinstance(key, tuple) else key
    return str(name)


def _level(value: Any) -> Optional[str]:
    value = getattr(value, "value", value)
    return str(value) if value else None


def claim_rows(entities: Dict[str, Dict]) -> List[Dict[str, Any]]:
    """Every cited claim in a run's entity tables (``load_existing_entities()``),
    with the verdict the run's judge stored for it, if any.

    Link-only hub pages are skipped: their text is frozen and says little
    about the hub itself."""
    rows: List[Dict[str, Any]] = []
    for table, entries in entities.items():
        for key, data in entries.items():
            profile = data.get("profile") or {}
            if profile.get("linked_only"):
                continue
            text = profile.get("text") or ""
            name = _name(key, data)
            report = data.get("profile_grounding") or {}
            stored = {
                (v.get("article_id"), sentence_key(v.get("claim") or "")): v
                for v in report.get("verifications") or []
            }
            seen = set()
            for c in _extract_cited_claims(text):
                claim = c["claim"][:GROUNDING_MAX_CLAIM_CHARS]
                if claim == "(no claim text)" or len(claim) < 20:
                    continue
                entity = f"{table}/{name}"
                iid = item_id(entity, c["article_id"], claim)
                if iid in seen:
                    continue
                seen.add(iid)
                v = stored.get((c["article_id"], sentence_key(claim))) or {}
                rows.append(
                    {
                        "item_id": iid,
                        "entity_table": table,
                        "entity_name": name,
                        "article_id": c["article_id"],
                        "claim": claim,
                        "split": article_split(c["article_id"]),
                        "stored_verdict": _level(v.get("support_level")),
                        "stored_judge": report.get("model") or "",
                    }
                )
    return rows


def _spread_pick(
    pool: List[Dict[str, Any]],
    n: int,
    rng: random.Random,
    per_article: Dict[str, int],
    per_entity: Dict[str, int],
    taken: set,
) -> List[Dict[str, Any]]:
    """Up to ``n`` claims from ``pool`` in random order, skipping any whose
    article or entity already gave its share."""
    pool = pool[:]
    rng.shuffle(pool)
    out = []
    for row in pool:
        if len(out) >= n:
            break
        entity = f"{row['entity_table']}/{row['entity_name']}"
        if row["item_id"] in taken:
            continue
        if per_article[row["article_id"]] >= PER_ARTICLE:
            continue
        if per_entity[entity] >= PER_ENTITY:
            continue
        out.append(row)
        taken.add(row["item_id"])
        per_article[row["article_id"]] += 1
        per_entity[entity] += 1
    return out


def sample_test_set(
    rows: List[Dict[str, Any]],
    n: int = 400,
    seed: int = 20261126,
    doubted_share: float = 0.5,
    have_text: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    """The claims a person labels, from test articles only.

    Half (``1 - doubted_share``) are drawn first at random from all test
    claims, so a real-world error rate can be stated for that half alone;
    the rest from the remaining claims the run's judge doubted (partial, not
    supported, unclear), so the set holds enough wrong claims to tell
    checkers apart. Each item keeps
    its ``stratum``; the stored verdict is not written out, so the labeller
    can't see it. ``have_text`` (article ids with text) drops claims whose
    article is missing."""
    rng = random.Random(seed)
    ok = set(have_text) if have_text is not None else None
    test = [
        r
        for r in rows
        if r["split"] == "test" and (ok is None or r["article_id"] in ok)
    ]
    doubted = [r for r in test if r.get("stored_verdict") in DOUBTED]
    per_article: Dict[str, int] = defaultdict(int)
    per_entity: Dict[str, int] = defaultdict(int)
    taken: set = set()
    want_doubted = int(round(n * doubted_share))
    # The random half first, from every test claim, so it stays a fair
    # sample (doubted claims in it at their real share); the extra doubted
    # claims are then drawn from what is left
    picked = [
        dict(r, stratum="random")
        for r in _spread_pick(
            test, n - want_doubted, rng, per_article, per_entity, taken
        )
    ]
    picked += [
        dict(r, stratum="doubted")
        for r in _spread_pick(
            doubted, n - len(picked), rng, per_article, per_entity, taken
        )
    ]
    rng.shuffle(picked)
    keep = ("item_id", "entity_table", "entity_name", "article_id", "claim", "split")
    return [{**{k: r[k] for k in keep}, "stratum": r["stratum"]} for r in picked]


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


LABELS = ("yes", "no", "unsure")


def latest_labels(path: str) -> Dict[str, Dict[str, Any]]:
    """item id -> the newest label row for it (a later click replaces an
    earlier one; the file itself is append-only)."""
    out: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("label") in LABELS:
            out[row["item_id"]] = row
    return out


def label_row(item: str, label: str, note: str = "", who: str = "") -> Dict[str, Any]:
    if label not in LABELS:
        raise ValueError(f"label must be one of {LABELS}, got {label!r}")
    return {
        "item_id": item,
        "label": label,
        "note": note,
        "labeller": who,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def article_texts(path: str) -> Dict[str, str]:
    """Article id -> text from the domain's articles Parquet."""
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["id", "content"])
    return {
        str(r["id"]): r["content"]
        for r in table.to_pylist()
        if r.get("id") and r.get("content")
    }


def article_meta(path: str) -> Dict[str, Dict[str, Any]]:
    """Article id -> ``{"title", "date"}`` from the articles Parquet (the
    columns it has; nothing for a file without either)."""
    import pyarrow.parquet as pq

    have = set(pq.read_schema(path).names)
    cols = [c for c in ("title", "published_date") if c in have]
    if "id" not in have or not cols:
        return {}
    table = pq.read_table(path, columns=["id", *cols])
    return {
        str(r["id"]): {"title": r.get("title"), "date": r.get("published_date")}
        for r in table.to_pylist()
        if r.get("id")
    }


def _day(value: Any) -> str:
    """YYYY-MM-DD from Unix seconds (as the articles file stores dates),
    milliseconds, ISO text or a date; anything else as it came, cut to 10
    characters."""
    from src.engine.later_updates import to_date

    text = str(value).strip()
    day = to_date(int(text) if text.isdigit() else value)
    return day.isoformat() if day else text[:10]


def article_head(meta: Dict[str, Any]) -> str:
    """The lines put above an article's text by ``prompts --with-head`` and
    ``score --with-head``: its headline and publication date (the day only),
    as the labellers saw them beside each sentence. Empty when neither is
    known."""
    lines = []
    if meta.get("title"):
        lines.append(f"Headline: {str(meta['title']).strip()}")
    if meta.get("date") not in (None, ""):
        lines.append(f"Published: {_day(meta['date'])}")
    return "\n".join(lines) + "\n\n" if lines else ""


def article_heads(path: str) -> Dict[str, str]:
    """Article id -> ``article_head()`` from the articles Parquet."""
    return {aid: article_head(m) for aid, m in article_meta(path).items()}
