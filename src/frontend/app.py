"""The Hinbox reader: a small Starlette app that fills Jinja2 templates.

Pages: home (``/``), search (``/search``), one list per entity table
(``/people`` …), entity pages (``/people/<key>``), sources (``/sources``,
``/sources/<id>``) and the source panel fragment (``/panel``) that the
reader script loads beside the profile when a citation is clicked. Every
page works without JavaScript; the script only opens sources in the panel.

The domain comes from ``?domain=`` (default ``guantanamo``) and is carried on
every link by the ``u()`` template helper.
"""

import os
from typing import Any, Dict, List
from urllib.parse import parse_qs, quote, unquote, urlencode, urlsplit

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

from . import corrections_view as cv
from . import views
from .data_access import SPECS, TABLES, available_domains, load_store
from .profile_render import VERDICT_LABELS

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DOMAIN = "guantanamo"
templates = Jinja2Templates(directory=os.path.join(HERE, "templates"))
templates.env.globals.update(
    TABLES=TABLES,
    VERDICT_LABELS=VERDICT_LABELS,
    fmt_date=views.fmt_date,
    static_site=False,  # `export.py` sets it while writing the static copy
)
templates.env.filters["fmt_date"] = views.fmt_date


def _domain(request: Request) -> str:
    d = request.query_params.get("domain") or DEFAULT_DOMAIN
    return d if d == DEFAULT_DOMAIN or d in available_domains() else DEFAULT_DOMAIN


def _render(
    request: Request, name: str, ctx: Dict[str, Any], status: int = 200
) -> Response:
    domain = _domain(request)

    def u(path: str, **params: Any) -> str:
        """A link that keeps the current domain."""
        q = {k: v for k, v in params.items() if v not in (None, "", False)}
        if domain != DEFAULT_DOMAIN:
            q["domain"] = domain
        return f"{path}?{urlencode(q)}" if q else path

    base = {
        "request": request,
        "u": u,
        "domain": domain,
        "domains": available_domains(),
        "q": request.query_params.get("q", ""),
    }
    return templates.TemplateResponse(
        request, name, {**base, **ctx}, status_code=status
    )


def _not_found(request: Request, what: str) -> Response:
    return _render(request, "not_found.html", {"what": what}, status=404)


async def home(request: Request) -> Response:
    store = load_store(_domain(request))
    return _render(request, "home.html", views.home_page(store))


async def search(request: Request) -> Response:
    store = load_store(_domain(request))
    return _render(
        request, "search.html", views.search(store, request.query_params.get("q", ""))
    )


async def entity_list(request: Request) -> Response:
    table = request.path_params["table"]
    if table not in SPECS:
        return _not_found(request, "page")
    store = load_store(_domain(request))
    return _render(
        request, "list.html", views.list_page(store, table, dict(request.query_params))
    )


async def entity(request: Request) -> Response:
    table = request.path_params["table"]
    key = unquote(request.path_params["key"])
    store = load_store(_domain(request))
    ent = store.get(table, key) if table in SPECS else None
    if ent is None:
        return _not_found(request, SPECS[table].singular if table in SPECS else "page")
    version = request.query_params.get("version")
    try:
        v = int(version) if version else None
    except ValueError:
        v = None
    domain = _domain(request)
    dq = f"?domain={quote(domain)}" if domain != DEFAULT_DOMAIN else ""
    return _render(request, "entity.html", views.entity_page(store, ent, v, dq))


async def panel(request: Request) -> Response:
    """The source panel's contents, loaded by the reader script."""
    p = request.query_params
    store = load_store(_domain(request))
    ent = store.get(p.get("t", ""), p.get("k", ""))
    if ent is None:
        return HTMLResponse(
            "<p class='panel-msg'>That page has changed; reload it.</p>", 404
        )
    try:
        claim = int(p.get("c", ""))
    except ValueError:
        claim = None
    try:
        version = int(p.get("v", "")) if p.get("v") else None
    except ValueError:
        version = None
    ctx = views.source_panel(store, ent, p.get("a", ""), claim, version)
    return _render(request, "_panel.html", ctx)


async def sources(request: Request) -> Response:
    store = load_store(_domain(request))
    return _render(
        request, "sources.html", views.sources_page(store, dict(request.query_params))
    )


async def source(request: Request) -> Response:
    store = load_store(_domain(request))
    sid = request.path_params["sid"]
    if sid not in store.sources:
        return _not_found(request, "source")
    return _render(request, "source.html", views.source_page(store, sid))


# ── corrections ──────────────────────────────────────────────
# A historian's corrections are appended to the corrections file
# (src/utils/corrections.py) and drawn on top of the pages; the tables are
# never edited here. Forms post url-encoded bodies, read without a
# multipart library.


def _same_origin(request: Request) -> bool:
    """A form post must come from a page of this reader, so another web page
    open in the same browser can't add corrections behind your back."""
    host = request.headers.get("host", "")
    for header in ("origin", "referer"):
        value = request.headers.get(header)
        if value:
            return urlsplit(value).netloc == host
    return False


async def _form(request: Request) -> Dict[str, List[str]]:
    body = (await request.body()).decode("utf-8", "replace")
    return parse_qs(body, keep_blank_values=True)


def _back(request: Request, url: str, anchor: str = "") -> Response:
    domain = _domain(request)
    q = f"?domain={quote(domain)}" if domain != DEFAULT_DOMAIN else ""
    return RedirectResponse(f"{url}{q}{anchor}", status_code=303)


def _entity_from(request: Request, store: Any) -> Any:
    p = request.query_params
    return store.get(p.get("t", ""), p.get("k", ""))


async def correct_sentence(
    request: Request, error: str = "", form: Any = None
) -> Response:
    store = load_store(_domain(request))
    p = dict(request.query_params)
    if form:  # re-showing a form that failed
        p = {k: v[0] for k, v in form.items() if v}
    ent = store.get(p.get("t", ""), p.get("k", ""))
    try:
        index = int(p.get("c", ""))
    except ValueError:
        index = -1
    ctx = (
        cv.sentence_form(store, cv.load(store), ent, index, p.get("a", ""))
        if ent
        else None
    )
    if ctx is None:
        return _not_found(request, "sentence")
    ctx.update(error=error, posted=p)
    return _render(request, "correct_sentence.html", ctx, status=400 if error else 200)


async def correct_page(request: Request, error: str = "", form: Any = None) -> Response:
    store = load_store(_domain(request))
    p = dict(request.query_params)
    if form:
        p = {k: v[0] for k, v in form.items() if v}
    ent = store.get(p.get("t", ""), p.get("k", ""))
    if ent is None:
        return _not_found(request, "page")
    ctx = cv.page_form(store, cv.load(store), ent)
    ctx.update(error=error, posted=p, ticked=(form or {}).get("article_ids") or [])
    return _render(request, "correct_page.html", ctx, status=400 if error else 200)


async def corrections(request: Request) -> Response:
    store = load_store(_domain(request))
    if request.method == "POST":
        if not _same_origin(request):
            return HTMLResponse(
                "Corrections can only be sent from the reader's own pages.", 403
            )
        form = await _form(request)
        try:
            _, ent, anchor = cv.save(store, cv.load(store), form)
        except ValueError as e:
            kind = (form.get("kind") or [""])[0]
            again = (
                correct_sentence
                if kind in ("strike", "verdict", "heading")
                else correct_page
            )
            return await again(request, error=str(e), form=form)
        return _back(request, ent.url, anchor)
    kind = request.query_params.get("kind", "")
    return _render(
        request, "corrections.html", cv.corrections_page(store, cv.load(store), kind)
    )


async def withdraw(request: Request) -> Response:
    if not _same_origin(request):
        return HTMLResponse(
            "Corrections can only be sent from the reader's own pages.", 403
        )
    store = load_store(_domain(request))
    cs = cv.load(store)
    try:
        cv.withdraw(store, cs, await _form(request))
    except ValueError as e:
        ctx = cv.corrections_page(store, cs)
        ctx["error"] = str(e)
        return _render(request, "corrections.html", ctx, status=400)
    return _back(request, "/corrections")


async def old_articles(request: Request) -> Response:
    return RedirectResponse(f"/sources/{quote(request.path_params['sid'])}", 301)


routes = [
    Route("/", home),
    Route("/search", search),
    Route("/panel", panel),
    Route("/sources", sources),
    Route("/sources/{sid}", source),
    Route("/articles/{sid}", old_articles),
    Route("/correct/sentence", correct_sentence),
    Route("/correct/page", correct_page),
    Route("/corrections", corrections, methods=["GET", "POST"]),
    Route("/corrections/withdraw", withdraw, methods=["POST"]),
    Mount(
        "/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static"
    ),
    Route("/{table}", entity_list),
    Route("/{table}/{key:path}", entity),
]

app = Starlette(routes=routes)
