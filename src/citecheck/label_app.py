"""A local page for yes/no labels on cited claims.

``just citecheck label --items <file> --labels <file> --articles <parquet>``
serves it on http://127.0.0.1:5078. Labelling runs in sprints: the start
screen offers 10, 20 or 40 sentences (20 by default), a bar along the top
fills as you go, and a short summary ends the sprint (how many, how long,
how far toward the first goal of ``GOAL`` labels).

Each screen shows one claim, the entity whose page it comes from, the
article's date and headline, and the whole article with the two sentences
that share the most words with the claim marked. A year, number or name in
the claim that appears nowhere in the article gets a dotted underline
(``claim_pieces()``). Both are word matching only, no model, so neither says
anything about any checker's verdict. On a wide screen the claim, date and
buttons stay in a left column while the article scrolls on the right.

Sentences from one article come one after another, so the article is read
once (``label_order()``); the articles themselves come in a fixed random
order, so stopping part way still leaves a fair sample.

Keys: ``y`` yes, ``n`` no, ``u`` can't tell, ``b`` back, Enter on the start
and end screens. Each click is appended to the labels file; clicking again
on an item replaces the earlier label.

The question is narrow on purpose: "Ignoring tone and wording, does the
article state everything in this sentence?" Loaded words or a sentence on
the wrong page are real faults, but not ones a citation check can see.

Now and then a cat peeks up from the bottom corner with a word of
encouragement (``cat_for()``). ``HINBOX_LABEL_CATS`` names them, e.g.
``Tom:ginger,Kit:tortie`` (kinds: ``ginger``, ``tortie``); unset, they are
just "the ginger cat" and "the tortoiseshell cat".

Audit mode: items that carry a ``level`` (teacher labels) show that verdict
and ask whether it is right, so the same page measures how often the
teachers are wrong. The page binds to 127.0.0.1 and accepts posts only from
itself.
"""

import os
import random
import re
import time
import unicodedata
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlencode

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from starlette.routing import Route

from src.citecheck.items import append_jsonl, label_row, latest_labels
from src.citecheck.label_page import PAGE
from src.engine.profile_stubs import split_sentences

QUESTION = (
    "Ignoring tone and wording, does the article state everything in this sentence?"
)
AUDIT_QUESTION = "Is the checkers' verdict right for this sentence?"
GOAL = 150
SIZES = (10, 20, 40)
DEFAULT_SIZE = 20
SPRINT_KEYS = ("s", "n", "b")
_WORD = re.compile(r"[a-z0-9]+")
_TOKEN = re.compile(r"\w[\w'’.,-]*\w|\w")
_STOP = set(
    "the a an and or of to in on at for by with from as is was were are be been "
    "that this it its his her their he she they said says had has have not".split()
)
CAT_KINDS = {"ginger": "the ginger cat", "tortie": "the tortoiseshell cat"}
CAT_LINES = (
    "{name} approves.",
    "{name} has come to see how it's going.",
    "{name} thinks that one was tricky.",
    "{name} would like a word about dinner.",
    "{name} is keeping you company.",
    "{name} says: carry on.",
)


def _content_words(text: str) -> set:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 2}


def marked_pieces(article: str, claim: str, top: int = 2) -> List[Tuple[str, bool]]:
    """The article as (text, marked) pieces, the ``top`` sentences sharing
    the most content words with the claim marked. Every character of the
    article is kept, in order."""
    want = _content_words(claim)
    sentences = split_sentences(article)
    scored = sorted(
        ((len(want & _content_words(s)), i) for i, s in enumerate(sentences)),
        reverse=True,
    )
    chosen = [sentences[i] for score, i in scored[:top] if score >= 2]
    spans = []
    for s in chosen:
        pattern = r"\s+".join(re.escape(w) for w in s.split())
        found = re.search(pattern, article)
        if found:
            spans.append(found.span())
    spans.sort()
    pieces, at = [], 0
    for start, end in spans:
        if start < at:
            continue
        pieces += [(article[at:start], False), (article[start:end], True)]
        at = end
    pieces.append((article[at:], False))
    return [(text, mark) for text, mark in pieces if text]


def _norm(token: str) -> str:
    """A word for matching: no accents, straight quotes, lower case, no
    possessive ending."""
    plain = "".join(
        c for c in unicodedata.normalize("NFKD", token) if not unicodedata.combining(c)
    )
    plain = plain.replace("’", "'").lower().rstrip(".")
    return plain[:-2] if plain.endswith("'s") else plain


def _checked(token: str) -> bool:
    """Only years, numbers and capitalised words (names) are looked for."""
    return any(c.isdigit() for c in token) or token[:1].isupper()


def claim_pieces(claim: str, article: str) -> List[Tuple[str, bool]]:
    """The claim as (text, flagged) pieces: a year, number or capitalised
    word that appears nowhere in the article is flagged. A hyphenated word
    counts as found when each of its parts is ("al-Nashiri" / "al Nashiri").
    Every character of the claim is kept, in order."""
    words = {_norm(t) for t in _TOKEN.findall(article)}
    words |= {p for w in list(words) for p in re.split(r"[-']", w) if p}
    pieces, at = [], 0
    for m in _TOKEN.finditer(claim):
        token = m.group(0)
        if not _checked(token):
            continue
        norm = _norm(token)
        parts = [p for p in re.split(r"[-']", norm) if p]
        if norm in words or (parts and all(p in words for p in parts)):
            continue
        pieces += [(claim[at : m.start()], False), (token, True)]
        at = m.end()
    pieces.append((claim[at:], False))
    return [(text, flag) for text, flag in pieces if text]


def label_order(items: List[Dict[str, Any]], seed: int = 5078) -> List[int]:
    """Item positions with each article's sentences together (in their
    file order) and the articles in a fixed random order."""
    groups: Dict[str, List[int]] = {}
    for i, item in enumerate(items):
        groups.setdefault(str(item.get("article_id")), []).append(i)
    articles = sorted(groups)
    random.Random(seed).shuffle(articles)
    return [i for a in articles for i in groups[a]]


def cats(spec: Optional[str] = None) -> List[Tuple[str, str]]:
    """(name, kind) for each cat in ``HINBOX_LABEL_CATS`` ("Name:kind,...");
    unnamed cats when it is unset or unreadable."""
    spec = os.getenv("HINBOX_LABEL_CATS", "") if spec is None else spec
    out = []
    for part in spec.split(","):
        name, _, kind = part.strip().partition(":")
        kind = kind.strip().lower()
        if name.strip() and kind in CAT_KINDS:
            out.append((name.strip(), kind))
    return out or [(label.capitalize(), kind) for kind, label in CAT_KINDS.items()]


def cat_for(
    sprint_start: int, done: int, size: int, pets: List[Tuple[str, str]]
) -> Optional[Dict[str, str]]:
    """A cat to show after the ``done``-th answer of a sprint, or None: about
    one answer in five, never two in a row, always at the halfway point."""
    if done <= 0 or done >= size:
        return None
    rng = random.Random(f"{sprint_start}:{done}")
    halfway = done == size // 2
    before = done - 1 == size // 2 or (
        done + 1 != size // 2
        and random.Random(f"{sprint_start}:{done - 1}").random() < 0.2
    )
    if not halfway and (before or rng.random() >= 0.2):
        return None
    name, kind = rng.choice(pets)
    line = (
        "Halfway there. {name} stretches." if halfway else rng.choice(CAT_LINES)
    ).format(name=name)
    return {"kind": kind, "line": line[0].upper() + line[1:]}


NAP_LINES = ("{name} is impressed.", "{name} fell asleep watching. Well done.")


def nap_for(sprint_start: int, pets: List[Tuple[str, str]]) -> Dict[str, str]:
    """The cat on the end-of-sprint screen."""
    rng = random.Random(f"nap:{sprint_start}")
    name, kind = rng.choice(pets)
    return {"kind": kind, "line": rng.choice(NAP_LINES).format(name=name)}


def _date(value: Any) -> str:
    if not value:
        return ""
    try:
        from datetime import datetime

        stamp = (
            value
            if hasattr(value, "strftime")
            else datetime.fromisoformat(str(value)[:19])
        )
        return f"{stamp.day} {stamp.strftime('%B %Y')}"
    except (TypeError, ValueError):
        return str(value)[:10]


def _minutes(seconds: float) -> str:
    """12 s, 4 min 05 s, 1 h 05 min; a break of more than a day is "over a day"."""
    total = max(0, int(seconds))
    if total >= 86400:
        return "over a day"
    h, rest = divmod(total, 3600)
    m, s = divmod(rest, 60)
    if h:
        return f"{h} h {m:02d} min"
    return f"{m} min {s:02d} s" if m else f"{s} s"


def make_app(
    items: List[Dict[str, Any]],
    labels_path: str,
    texts: Dict[str, str],
    labeller: str = "",
    meta: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Starlette:
    order = label_order(items)
    place = {i: p for p, i in enumerate(order)}
    pets = cats()
    meta = meta or {}

    def next_unlabelled(after: int = -1) -> Optional[int]:
        labels = latest_labels(labels_path)
        start = place.get(after, -1) + 1
        ring = order[start:] + order[:start]
        return next((i for i in ring if items[i]["item_id"] not in labels), None)

    def sprint_of(params: Dict[str, str]) -> Dict[str, int]:
        if not all(params.get(k, "").isdigit() for k in SPRINT_KEYS):
            return {}
        return {k: int(params[k]) for k in SPRINT_KEYS}

    def counts() -> Dict[str, Any]:
        labels = latest_labels(labels_path)
        done = sum(1 for i in items if i["item_id"] in labels)
        return {
            "done": done,
            "total": len(items),
            "left": done < len(items),
            "goal": GOAL,
            "goal_pct": min(100, round(100 * done / GOAL)),
        }

    def url(path: str, **params: Any) -> str:
        clean = {k: v for k, v in params.items() if v is not None}
        return path + ("?" + urlencode(clean) if clean else "")

    def _last(raw: str, sprint: Dict[str, int]) -> Optional[str]:
        """Link back to the sprint's last answer, to fix a misclick."""
        if not raw.isdigit() or int(raw) >= len(items):
            return None
        return url("/", i=int(raw), **{k: sprint[k] for k in SPRINT_KEYS})

    async def start(request: Request) -> RedirectResponse:
        raw = request.query_params.get("n", "")
        size = int(raw) if raw.isdigit() and 0 < int(raw) <= 400 else DEFAULT_SIZE
        return RedirectResponse(
            url("/", s=int(time.time()), n=size, b=counts()["done"]), status_code=303
        )

    async def page(request: Request):
        params = dict(request.query_params)
        sprint = sprint_of(params)
        base = counts()
        raw = params.get("i", "")
        if not raw.isdigit() and not sprint:
            return HTMLResponse(
                PAGE.render(
                    screen="start", sizes=SIZES, default_size=DEFAULT_SIZE, **base
                )
            )
        info = None
        if sprint:
            got = max(0, base["done"] - sprint["b"])
            info = {
                "size": sprint["n"],
                "done": got,
                "pct": min(100, round(100 * got / max(1, sprint["n"]))),
            }
            # The summary only on the redirect after the last answer: a link
            # naming an item (back, or "change the last answer") still opens it.
            if not raw.isdigit() and (got >= sprint["n"] or not base["left"]):
                spent = max(0, time.time() - sprint["s"])
                info["minutes"] = _minutes(spent)
                info["per_item"] = _minutes(spent / got) if got else "–"
                return HTMLResponse(
                    PAGE.render(
                        screen="finish",
                        sprint=info,
                        last=_last(params.get("last", ""), sprint),
                        nap=nap_for(sprint["s"], pets),
                        **base,
                    )
                )
        index = (
            int(raw) if raw.isdigit() and int(raw) < len(items) else next_unlabelled()
        )
        if index is None:  # nothing left to label
            return RedirectResponse("/", status_code=303)
        item = items[index]
        labels = latest_labels(labels_path)
        current = labels.get(item["item_id"], {})
        keep = {k: sprint[k] for k in SPRINT_KEYS} if sprint else {}
        p = place[index]
        back = url("/", i=order[p - 1], **keep) if p > 0 else None
        prev = items[order[p - 1]] if p > 0 else None
        article = texts.get(item["article_id"], "")
        pieces = claim_pieces(item["claim"], article)
        row = meta.get(str(item["article_id"]), {})
        cat = (
            cat_for(sprint["s"], info["done"], sprint["n"], pets)
            if sprint and params.get("ok")
            else None
        )
        html = PAGE.render(
            screen="item",
            item=item,
            index=index,
            sprint=info,
            keep=keep,
            back=back,
            saved=bool(params.get("ok")),
            cat=cat,
            same_article=bool(prev and prev["article_id"] == item["article_id"]),
            meta={"date": _date(row.get("date")), "title": row.get("title") or ""},
            current=current.get("label"),
            note=current.get("note", ""),
            question=AUDIT_QUESTION if item.get("level") else QUESTION,
            claim_pieces=pieces,
            flags=any(flag for _, flag in pieces),
            pieces=marked_pieces(article, item["claim"]),
            **base,
        )
        return HTMLResponse(html)

    async def label(request: Request):
        origin = request.headers.get("origin") or request.headers.get("referer") or ""
        host = request.headers.get("host", "")
        if origin and host not in origin:
            return PlainTextResponse("posts only from this page", status_code=403)
        body = (await request.body()).decode("utf-8", "replace")
        form = {k: v[0] for k, v in parse_qs(body, keep_blank_values=True).items()}
        known = {i["item_id"] for i in items}
        iid = str(form.get("item_id") or "")
        if iid not in known:
            return PlainTextResponse("unknown item", status_code=400)
        append_jsonl(
            labels_path,
            label_row(
                iid, str(form.get("label")), str(form.get("note") or ""), labeller
            ),
        )
        raw = str(form.get("index") or "0")
        nxt = next_unlabelled(int(raw) if raw.isdigit() else -1)
        sprint = sprint_of({k: str(form.get(k) or "") for k in SPRINT_KEYS})
        if sprint:
            done = counts()["done"]
            if nxt is None or done - sprint["b"] >= sprint["n"]:  # sprint over
                last = raw if raw.isdigit() else None
                return RedirectResponse(
                    url("/", ok=1, last=last, **sprint), status_code=303
                )
            return RedirectResponse(url("/", i=nxt, ok=1, **sprint), status_code=303)
        return RedirectResponse("/" if nxt is None else f"/?i={nxt}", status_code=303)

    return Starlette(
        routes=[
            Route("/", page),
            Route("/start", start),
            Route("/label", label, methods=["POST"]),
        ]
    )
