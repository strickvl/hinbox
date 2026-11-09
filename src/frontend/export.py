"""Write the reader as a static site: plain HTML files you can zip, host
anywhere or open from disk, with no Python server.

    uv run python -m src.frontend.export --out site [--domain guantanamo]

It starts at the home page, renders each page through the app exactly as the
server would, follows every link to another page, and saves each one as a
file (``people/Yusuf Qadir.html``, ``sources/<id>.html``; an older version is
``people/versions/<key>.v2.html``). Links are rewritten to relative file paths.

What differs from the server:

- Search runs in the browser over ``static/search-index.js`` (names and other
  names of every page and source titles); there is no ``/search`` page.
- The list filters and the domain picker need a server, so they are left out.
- The panel beside the profile is built in the browser: ``reader.js`` loads
  the source's exported page and reads the verdicts the entity page carries
  (``cite_data()``). Browsers refuse that for a site opened from disk, so
  there a citation opens the source page; serve the folder to get the panel.

The site holds the run's article text and profiles: share it only where the
articles may be shared.
"""

import argparse
import hashlib
import json
import os
import posixpath
import re
import shutil
from collections import deque
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import parse_qs, quote, unquote, urlsplit

from starlette.testclient import TestClient

from .app import DEFAULT_DOMAIN, HERE, app, templates
from .data_access import SPECS, TABLES, load_store

_LINK = re.compile(r'(href|src|action)="(/[^"]*)"')
_SKIP = ("/search", "/panel", "/static/", "/correct")
# Characters a file name can't hold safely (path, URL or Windows trouble);
# "%" so the encoding can be undone, "~" so only a cut name has one
_UNSAFE = re.compile(r'[%/\\?#:*"<>|~\x00-\x1f]')
_MAX_NAME_BYTES = 180  # well under the usual 255-byte limit, with room for ".vNN.html"


def safe_name(key: str) -> str:
    """A file name for ``key``: different keys never get the same name."""
    name = _UNSAFE.sub(lambda m: "%{:02X}".format(ord(m.group())), key)
    if name.startswith("."):
        name = "%2E" + name[1:]
    if len(name.encode("utf-8")) > _MAX_NAME_BYTES:
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
        cut = name.encode("utf-8")[:120].decode("utf-8", "ignore")
        name = f"{cut}~{digest}"
    return name


def page_url(raw: str) -> Optional[str]:
    """The page a link points at, as ``path`` or ``path?version=N``, or None
    for links the static copy has no file for (search, the panel, assets)."""
    parts = urlsplit(raw)
    path = parts.path.rstrip("/") or "/"
    if path.startswith(_SKIP) or path == "/search":
        return None
    version = parse_qs(parts.query).get("version", [""])[0]
    segs = path.strip("/").split("/")
    if version.isdigit() and len(segs) >= 2 and segs[0] in SPECS:
        return f"{path}?version={version}"
    return path


def file_for(url: str) -> str:
    """Where a page's HTML is saved, relative to the site root."""
    if url.startswith("/static/"):
        return url.lstrip("/")
    parts = urlsplit(url)
    if parts.path == "/":
        return "index.html"
    segs = parts.path.strip("/").split("/")
    first, rest = segs[0], [unquote(s) for s in segs[1:]]
    if not rest:
        return f"{first}.html"
    # "A/B" is "A%2FB", "A_B" stays "A_B"; see safe_name()
    name = safe_name("/".join(rest))
    version = parse_qs(parts.query).get("version", [""])[0]
    if version:  # its own folder, so "X?version=2" can't meet a key "X.v2"
        return f"{first}/versions/{name}.v{version}.html"
    return f"{first}/{name}.html"


def _rel(target: str, current: str) -> str:
    rel = posixpath.relpath(target, posixpath.dirname(current) or ".")
    return quote(rel, safe="/.-_~")


def rewrite(html: str, current: str, known: Set[str]) -> Tuple[str, List[str]]:
    """Point every site link in ``html`` at its file; return links with no file."""
    missing: List[str] = []

    def sub(m: "re.Match[str]") -> str:
        attr, raw = m.group(1), m.group(2)
        if raw.startswith("/static/"):
            return f'{attr}="{_rel(file_for(raw), current)}"'
        url = page_url(raw.replace("&amp;", "&"))
        if attr == "action" or url is None:
            return f'{attr}="#"'
        if url not in known:
            missing.append(url)
            return f'{attr}="#"'
        frag = raw.split("#", 1)[1] if "#" in raw else ""
        return f'{attr}="{_rel(file_for(url), current)}{"#" + frag if frag else ""}"'

    out = _LINK.sub(sub, html)
    root = posixpath.relpath(".", posixpath.dirname(current) or ".")
    root = "" if root == "." else root + "/"
    out = out.replace("<body", f'<body data-root="{root}"', 1)
    return out, missing


def _search_index(domain: str) -> List[Dict[str, object]]:
    store = load_store(domain)
    rows: List[Dict[str, object]] = []
    for spec in TABLES:
        for ent in store.entities.get(spec.table, {}).values():
            rows.append(
                {
                    "n": ent.name,
                    "o": ent.other_names,
                    "t": spec.singular,
                    "u": quote(file_for(ent.url)),
                    "c": len(ent.articles),
                }
            )
    for sid, src in store.sources.items():
        if store.by_source.get(sid):
            rows.append(
                {
                    "n": src.title or sid,
                    "t": "source",
                    "u": quote(file_for(f"/sources/{quote(sid, safe='')}")),
                }
            )
    return rows


def export_site(out: str, domain: str = DEFAULT_DOMAIN) -> Dict[str, int]:
    """Write the static copy of ``domain`` to ``out``; return counts."""
    q = f"?domain={domain}" if domain != DEFAULT_DOMAIN else ""
    templates.env.globals["static_site"] = True
    pages: Dict[str, str] = {}
    try:
        client = TestClient(app)
        todo, seen = deque(["/"]), {"/"}
        while todo:
            url = todo.popleft()
            sep = "&" if "?" in url else ""
            res = client.get(url + (sep + q[1:] if sep and q else q))
            if res.status_code != 200:
                continue
            pages[url] = res.text
            for _, raw in _LINK.findall(res.text):
                nxt = page_url(raw.replace("&amp;", "&"))
                if nxt and nxt not in seen:
                    seen.add(nxt)
                    todo.append(nxt)
    finally:
        templates.env.globals["static_site"] = False

    known = set(pages)
    broken: Set[str] = set()
    for url, html in pages.items():
        path = file_for(url)
        html, missing = rewrite(html, path, known)
        broken.update(missing)
        dest = os.path.join(out, path)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w", encoding="utf-8") as fh:
            fh.write(html)
    shutil.copytree(
        os.path.join(HERE, "static"), os.path.join(out, "static"), dirs_exist_ok=True
    )
    with open(
        os.path.join(out, "static", "search-index.js"), "w", encoding="utf-8"
    ) as fh:
        index = json.dumps(
            _search_index(domain), ensure_ascii=False, separators=(",", ":")
        )
        fh.write(f"window.HINBOX_INDEX={index};\n")
    return {"pages": len(pages), "broken_links": len(broken)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True, help="folder to write the site to")
    parser.add_argument("--domain", default=DEFAULT_DOMAIN)
    args = parser.parse_args()
    counts = export_site(args.out, args.domain)
    print(
        f"Wrote {counts['pages']} pages to {args.out}/ "
        f"({counts['broken_links']} links to pages that don't exist). "
        f"Open {args.out}/index.html, or serve the folder with any web server."
    )


if __name__ == "__main__":
    main()
