"""A local page for yes/no labels on cited claims.

``just citecheck label --items <file> --labels <file> --articles <parquet>``
serves it on http://127.0.0.1:5078. Each screen shows one claim, the entity
whose page it comes from, and the whole article with the two sentences that
share the most words with the claim marked (word overlap only, no model, so
the marks say nothing about any checker's verdict). Keys: ``y`` yes, ``n``
no, ``u`` can't tell, ``b`` back. Each click is appended to the labels file;
clicking again on an item replaces the earlier label.

The question is narrow on purpose: "Ignoring tone and wording, does the
article state everything in this sentence?" Loaded words or a sentence on
the wrong page are real faults, but not ones a citation check can see.

Audit mode: items that carry a ``level`` (teacher labels) show that verdict
and ask whether it is right, so the same page measures how often the
teachers are wrong. The page binds to 127.0.0.1 and accepts posts only from
itself.
"""

import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs

from jinja2 import Environment
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from starlette.routing import Route

from src.citecheck.items import append_jsonl, label_row, latest_labels
from src.engine.profile_stubs import split_sentences

QUESTION = (
    "Ignoring tone and wording, does the article state everything in this sentence?"
)
AUDIT_QUESTION = "Is the checkers' verdict right for this sentence?"
_WORD = re.compile(r"[a-z0-9]+")
_STOP = set(
    "the a an and or of to in on at for by with from as is was were are be been "
    "that this it its his her their he she they said says had has have not".split()
)

PAGE = Environment(autoescape=True).from_string(
    """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Citation labels</title>
<style>
:root{--bg:#fbfaf7;--fg:#1d2327;--muted:#5c6770;--mark:#fff1b8;--accent:#2c5f7c;--card:#fff;--line:#e3e0d8}
@media (prefers-color-scheme:dark){:root{--bg:#15191c;--fg:#e6e8ea;--muted:#9aa5ae;--mark:#5a4b12;--accent:#7fb3d1;--card:#1d2226;--line:#2c3338}}
body{background:var(--bg);color:var(--fg);font:16px/1.55 Georgia,serif;margin:0;padding:16px}
main{max-width:860px;margin:auto}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:16px;margin-bottom:16px}
.claim{font-size:1.15rem}.muted{color:var(--muted);font:14px/1.4 system-ui,sans-serif}
.q{font:600 16px system-ui,sans-serif;margin:12px 0}
button{font:600 16px system-ui,sans-serif;padding:10px 18px;margin:4px 8px 4px 0;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
button.yes{border-color:#2e7d4f}button.no{border-color:#b3412e}
button.on{background:var(--accent);color:#fff}
input[type=text]{width:100%;padding:8px;font:15px system-ui,sans-serif;box-sizing:border-box;background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:6px}
mark{background:var(--mark);color:inherit}
.article{white-space:pre-wrap;max-height:60vh;overflow:auto}
a{color:var(--accent)}
</style></head><body><main>
<p class="muted">{{ done }} of {{ total }} labelled · item {{ index + 1 }} ·
{% if index > 0 %}<a href="/?i={{ index - 1 }}">back (b)</a> · {% endif %}
<a href="/">next unlabelled</a></p>
{% if item %}
<div class="card">
<div class="muted">From the page of <b>{{ item.entity_name }}</b>, citing article {{ item.article_id }}</div>
<p class="claim">{{ item.claim }}</p>
{% if item.level %}<p class="muted">Checkers' verdict: <b>{{ item.level }}</b>{% if item.reasoning %}: {{ item.reasoning }}{% endif %}</p>{% endif %}
<form method="post" action="/label" id="f">
<input type="hidden" name="item_id" value="{{ item.item_id }}">
<input type="hidden" name="index" value="{{ index }}">
<div class="q">{{ question }}</div>
<button class="yes {{ 'on' if current == 'yes' }}" name="label" value="yes">Yes (y)</button>
<button class="no {{ 'on' if current == 'no' }}" name="label" value="no">No (n)</button>
<button class="{{ 'on' if current == 'unsure' }}" name="label" value="unsure">Can't tell (u)</button>
<input type="text" name="note" placeholder="Optional note" value="{{ note }}">
</form></div>
<div class="card article">{% for piece, marked in pieces %}{% if marked %}<mark id="m">{{ piece }}</mark>{% else %}{{ piece }}{% endif %}{% endfor %}</div>
{% else %}<div class="card"><p>Every item is labelled. Thank you.</p></div>{% endif %}
</main>
<script>
var m=document.getElementById('m');if(m)m.scrollIntoView({block:'center'});
document.addEventListener('keydown',function(e){
 if(e.target.tagName==='INPUT')return;
 var k={y:'yes',n:'no',u:'unsure'}[e.key];
 if(k){var b=document.querySelector('button[value='+k+']');if(b)b.click();}
 if(e.key==='b'&&{{ index }}>0)location='/?i={{ index - 1 }}';
});
</script></body></html>"""
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


def make_app(
    items: List[Dict[str, Any]],
    labels_path: str,
    texts: Dict[str, str],
    labeller: str = "",
) -> Starlette:
    def next_unlabelled(after: int = -1) -> Optional[int]:
        labels = latest_labels(labels_path)
        order = list(range(after + 1, len(items))) + list(range(0, after + 1))
        return next((i for i in order if items[i]["item_id"] not in labels), None)

    async def page(request: Request) -> HTMLResponse:
        labels = latest_labels(labels_path)
        raw = request.query_params.get("i")
        index = int(raw) if raw and raw.isdigit() else next_unlabelled()
        item = items[index] if index is not None and index < len(items) else None
        current = labels.get(item["item_id"], {}) if item else {}
        audit = bool(item and item.get("level"))
        html = PAGE.render(
            item=item,
            index=index if index is not None else len(items),
            total=len(items),
            done=sum(1 for i in items if i["item_id"] in labels),
            current=current.get("label"),
            note=current.get("note", ""),
            question=AUDIT_QUESTION if audit else QUESTION,
            pieces=marked_pieces(texts.get(item["article_id"], ""), item["claim"])
            if item
            else [],
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
        index = int(str(form.get("index") or "0"))
        nxt = next_unlabelled(index)
        return RedirectResponse("/" if nxt is None else f"/?i={nxt}", status_code=303)

    return Starlette(
        routes=[Route("/", page), Route("/label", label, methods=["POST"])]
    )
