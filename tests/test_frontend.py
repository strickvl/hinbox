"""Tests for the web reader (src/frontend/) on small made-up entity tables.

Every name, sentence and article here is invented: the repository is public
and must not hold run outputs or article text.
"""

import os
from typing import Any, Dict, List

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from starlette.testclient import TestClient

from src.frontend import data_access
from src.frontend.data_access import make_location_key, to_date
from src.frontend.profile_render import parse_claims, render_profile, shift_headings
from src.frontend.views import checker_note
from src.utils.file_ops import write_entities_table
from src.utils.grounding import _extract_cited_claims

A1, A2, A3 = (
    "11111111-aaaa-4bbb-8ccc-000000000001",
    "11111111-aaaa-4bbb-8ccc-000000000002",
    "11111111-aaaa-4bbb-8ccc-000000000003",
)
ARTICLES = [
    {
        "id": A1,
        "title": "Suspect held after harbour raid",
        "url": "https://example.org/a1",
        "published_date": "1049241600",  # 2003-04-02, epoch seconds as text
        "content": "Police said they arrested Yusuf Qadir in a dawn raid.\nNeighbours said Qadir had studied engineering at a local college.",
    },
    {
        "id": A2,
        "title": "Prisoners moved to the base",
        "url": "https://example.org/a2",
        "published_date": "1157587200",  # 2006-09-07
        "content": "Fourteen men were flown to the base.\nAmong them was Yusuf Qadir, a made-up detainee.",
    },
    {
        "id": A3,
        "title": "Hunger strike spreads",
        "url": "https://example.org/a3",
        "published_date": "1365897600",  # 2013-04-14
        "content": "More than 100 prisoners joined a hunger strike. Some were force-fed.",
    },
]
TEXT = (
    f"Yusuf Qadir is a made-up detainee held at the base.^[{A2}]\n\n"
    "### Background\n\n"
    f"Qadir studied engineering\nat a local college.^[{A1}] "
    f"He worked for a shipping firm^[{A1}, {A2}].\n\n"
    "### Detention\n\n"
    "#### Hunger strike\n\n"
    f"He was force-fed for several weeks.^[{A3}] "
    f"Sources disagree on the year he arrived: 2006 in one report, 2007 in another.^[{A2}]\n\n"
    "A stray <script>alert(1)</script> tag the model wrote."
)


def _verif(aid: str, claim: str, level: str, reason: str = "") -> Dict[str, Any]:
    return {"article_id": aid, "citation": f"^[{aid}]", "claim": claim,
            "support_level": level, "reasoning": reason}  # fmt: skip


def _report() -> Dict[str, Any]:
    levels = {
        "force-fed": ("not_supported", "The article names no one who was force-fed."),
        "shipping": ("partial", "Only one article mentions the firm."),
    }
    out = []
    for c in _extract_cited_claims(TEXT):
        level, reason = next(
            (v for k, v in levels.items() if k in c["claim"]), ("supported", "")
        )
        if "shipping" in c["claim"] and c["article_id"] == A2:
            level, reason = "supported", ""
        out.append(_verif(c["article_id"], c["claim"], level, reason))
    return {"verifications": out, "model": "anthropic/claude-haiku-5-5"}


def _arts(*ids: str) -> List[Dict[str, Any]]:
    by = {a["id"]: a for a in ARTICLES}
    return [{"article_id": i, "article_title": by[i]["title"], "article_url": by[i]["url"],
             "article_published_date": int(by[i]["published_date"])} for i in ids]  # fmt: skip


@pytest.fixture
def client(tmp_path, monkeypatch):
    out = tmp_path / "entities"
    out.mkdir()
    person = {
        "name": "Yusuf Qadir",
        "type": "detainee",
        "alternative_names": [{"name": "Yousef Kadir", "type": "detainee"}],
        "articles": _arts(A1, A2, A3),
        "profile": {"text": TEXT, "sources": [A1, A2, A3]},
        "profile_grounding": _report(),
        "profile_versions": {
            "current_version": 2,
            "versions": [
                {"version_number": 1, "profile_data": {"text": f"Qadir was held.^[{A1}]"},
                 "created_at": "2026-11-01T10:00:00", "trigger_article_id": A1},
                {"version_number": 2, "profile_data": {"text": TEXT},
                 "created_at": "2026-11-02T10:00:00", "trigger_article_id": A3},
            ],
        },
    }  # fmt: skip
    stub = {
        "name": "Miriam Hale",
        "type": "military",
        "articles": _arts(A2),
        "profile": {
            "text": f"Among them was Yusuf Qadir, a made-up detainee.^[{A2}]",
            "stub": True,
        },
    }
    country = {
        "name": "Freedonia",
        "type": "country",
        "articles": _arts(A1, A3),
        "profile": {
            "text": f"Police said they arrested Yusuf Qadir.^[{A1}]",
            "stub": True,
        },
    }
    event = {
        "title": "Made-up hunger strike",
        "event_type": "EventType.PROTEST",
        "start_date": "2013-02-01 00:00:00+00:00",
        "articles": _arts(A3),
        "profile": {"text": f"A strike began.^[{A3}]"},
    }
    write_entities_table("people", [person, stub], str(out))
    write_entities_table("locations", [country], str(out))
    write_entities_table("events", [event], str(out))
    write_entities_table("organizations", [], str(out))
    src = tmp_path / "articles.parquet"
    pq.write_table(pa.Table.from_pylist(ARTICLES), src)
    monkeypatch.setenv("HINBOX_OUTPUT_DIR", str(out))
    monkeypatch.setenv("HINBOX_ARTICLES_PATH", str(src))
    data_access._CACHE.clear()
    from src.frontend.app import app

    return TestClient(app)


# ── rendering ────────────────────────────────────────────────


def test_claims_match_the_grounding_step():
    """The reader must split claims exactly as grounding did, or verdicts
    land on the wrong sentences."""
    claims, _ = parse_claims(TEXT)
    ours = [(aid, c.text) for c in claims for aid in c.articles]
    theirs = [(c["article_id"], c["claim"]) for c in _extract_cited_claims(TEXT)]
    assert ours == theirs


def test_verdicts_numbering_and_escaping():
    rp = render_profile(TEXT, _report())
    assert rp.order == [A2, A1, A3]  # numbered by first citation
    assert 'class="s v-not_supported"' in rp.html  # the force-fed sentence
    assert 'class="s v-partial"' in rp.html  # shipping: partly, worst of two
    assert "<script>" not in rp.html and "&lt;script&gt;" in rp.html
    assert rp.counts() == {"supported": 3, "partial": 1, "not_supported": 1}
    # ### is the top level here, so it becomes h2, #### becomes h3
    assert [(s.title, s.level) for s in rp.sections] == [
        ("Background", 2), ("Detention", 3 - 1), ("Hunger strike", 3)]  # fmt: skip
    detention = rp.sections[1]
    assert detention.claims == 2 and detention.flags == {"not_supported": 1}


def test_no_report_means_no_marks():
    rp = render_profile(TEXT, None)
    assert not rp.checked
    assert "v-not_supported" not in rp.html and "vm v-" not in rp.html


def test_shift_headings_keeps_relative_depth():
    assert shift_headings("### A\n#### B\n### C") == "## A\n### B\n## C"


def test_dates_and_checker_note():
    assert to_date(1049241600) == "2003-04-02"
    assert to_date("1049241600") == "2003-04-02"
    assert to_date("2013-02-01 00:00:00+00:00") == "2013-02-01"
    assert "2 of 42" in checker_note("anthropic/claude-haiku-5-5")["note"]
    assert "19 of 42" in checker_note("ollama/local-qwen36-32k")["note"]
    assert "not been hand-tested" in checker_note("openai/some-new-model")["note"]


# ── pages ────────────────────────────────────────────────────


def test_entity_page_shows_trust_line_and_sections(client):
    html = client.get("/people/Yusuf%20Qadir").text
    assert ">5</b> cited sentences" in html
    assert "1 not supported" in html and "1 partly supported" in html
    assert "Checked by claude-haiku-5-5" in html
    assert 'href="#hunger-strike"' in html  # contents list
    assert "Yousef Kadir" in html  # other names from dict rows
    assert 'id="disagree"' in html and "Sources disagree on the year" in html
    assert "Made-up hunger strike" in html  # event in the same article
    assert "Miriam Hale" in html  # appears with
    assert "2003-04-02" in html  # dates from epoch seconds


def test_old_version_has_no_marks(client):
    html = client.get("/people/Yusuf%20Qadir?version=1").text
    assert "version 1</b> of 2" in html
    assert "v-not_supported" not in html


def test_stub_and_link_only_labels(client):
    assert (
        "Stub: sentences copied word for word"
        in client.get("/people/Miriam%20Hale").text
    )
    key = make_location_key({"name": "Freedonia", "type": "country"})
    html = client.get(f"/locations/{key.replace(' ', '%20')}").text
    assert "Link-only page." in html


def test_panel_shows_verdict_reason_and_text(client):
    claims, _ = parse_claims(TEXT)
    idx = next(c.index for c in claims if "force-fed" in c.text)
    r = client.get(
        "/panel", params={"t": "people", "k": "Yusuf Qadir", "a": A3, "c": idx}
    )
    assert r.status_code == 200
    assert "Not supported" in r.text and "names no one" in r.text
    assert "More than 100 prisoners" in r.text
    # A sentence copied word for word from the article is highlighted
    idx = next(c.index for c in claims if "engineering" in c.text)
    r = client.get(
        "/panel", params={"t": "people", "k": "Yusuf Qadir", "a": A1, "c": idx}
    )
    assert "<mark>" not in r.text and "had studied engineering" in r.text
    r = client.get(
        "/panel", params={"t": "people", "k": "Miriam Hale", "a": A2, "c": 0}
    )
    assert "<mark>Among them was Yusuf Qadir, a made-up detainee</mark>" in r.text


def test_panel_for_an_old_version_quotes_that_version(client):
    r = client.get(
        "/panel", params={"t": "people", "k": "Yusuf Qadir", "a": A1, "c": 0, "v": 1}
    )
    assert "Qadir was held." in r.text and "Older version" in r.text
    assert "Not supported" not in r.text


def test_citation_links_keep_the_domain(client):
    from src.frontend import views

    store = data_access.load_store()
    ent = store.get("people", "Yusuf Qadir")
    page = views.entity_page(store, ent, None, "?domain=other")
    assert f"/sources/{A1}?domain=other" in str(page["body"])


def test_wrapped_sentence_keeps_its_verdict():
    rp = render_profile(TEXT, _report())
    claim = next(c for c in rp.claims if "engineering" in c.text)
    assert claim.verdicts == {A1: "supported"}  # matched despite the line break
    assert "Qadir studied engineering\nat a local college.</span>" in rp.html


def test_source_page_lists_citing_sentences(client):
    html = client.get(f"/sources/{A3}").text
    assert "Hunger strike spreads" in html and "News article" in html
    assert "He was force-fed for several weeks." in html
    assert client.get("/sources/nope").status_code == 404


def test_lists_search_and_home(client):
    assert "Yusuf Qadir" in client.get("/people").text
    assert "Yusuf Qadir" not in client.get("/people", params={"kind": "stub"}).text
    assert "Yusuf Qadir" in client.get("/people", params={"flagged": "1"}).text
    assert "Yusuf Qadir" in client.get("/search", params={"q": "kadir"}).text
    home = client.get("/").text
    assert "Needs attention" in home and "Yusuf Qadir" in home
    assert client.get("/people/Nobody").status_code == 404
    assert client.get("/nonsense").status_code == 404


def test_server_listens_on_localhost_by_default(monkeypatch):
    import src.frontend.__main__ as main

    seen: Dict[str, Any] = {}
    monkeypatch.delenv("HINBOX_FRONTEND_HOST", raising=False)
    monkeypatch.setattr(main.uvicorn, "run", lambda app, **kw: seen.update(kw))
    main.main()
    assert seen["host"] == "127.0.0.1"


def test_templates_exist():
    here = os.path.join(os.path.dirname(data_access.__file__), "templates")
    for name in [
        "base",
        "entity",
        "_panel",
        "home",
        "list",
        "search",
        "source",
        "sources",
    ]:
        assert os.path.exists(os.path.join(here, f"{name}.html"))


def test_demo_corpus_renders(tmp_path, monkeypatch):
    """`just demo-reader`'s made-up corpus loads and shows every kind of mark."""
    from scripts.demo_reader_data import build

    build(str(tmp_path))
    monkeypatch.setenv("HINBOX_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("HINBOX_ARTICLES_PATH", str(tmp_path / "articles.parquet"))
    data_access._CACHE.clear()
    from src.frontend.app import app

    c = TestClient(app)
    page = c.get("/people/Yusuf%20Qadir").text
    for cls in ("v-partial", "v-not_supported", "v-unchecked"):
        assert cls in page
    assert c.get("/").status_code == 200
