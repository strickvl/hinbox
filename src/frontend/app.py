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
from typing import Any, Dict
from urllib.parse import quote, unquote, urlencode

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

from . import views
from .data_access import SPECS, TABLES, available_domains, load_store
from .profile_render import VERDICT_LABELS

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DOMAIN = "guantanamo"
templates = Jinja2Templates(directory=os.path.join(HERE, "templates"))
templates.env.globals.update(
    TABLES=TABLES, VERDICT_LABELS=VERDICT_LABELS, fmt_date=views.fmt_date
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
    return _render(request, "entity.html", views.entity_page(store, ent, v))


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
    ctx = views.source_panel(store, ent, p.get("a", ""), claim)
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


async def old_articles(request: Request) -> Response:
    return RedirectResponse(f"/sources/{quote(request.path_params['sid'])}", 301)


routes = [
    Route("/", home),
    Route("/search", search),
    Route("/panel", panel),
    Route("/sources", sources),
    Route("/sources/{sid}", source),
    Route("/articles/{sid}", old_articles),
    Mount(
        "/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static"
    ),
    Route("/{table}", entity_list),
    Route("/{table}/{key:path}", entity),
]

app = Starlette(routes=routes)
