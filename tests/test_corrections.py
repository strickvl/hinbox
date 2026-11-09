"""Tests for a historian's corrections: the append-only file
(src/utils/corrections.py) and how the reader records and draws them.

Every name, sentence and article here is invented (see tests/test_frontend.py).
"""

import json
import re
from html import unescape
from typing import Any, Dict

import pytest

from src.frontend.profile_render import Overlay, render_profile
from src.utils.corrections import (
    append_correction,
    corrections_path,
    entity_ref,
    load_corrections,
    sentence_key,
    validate,
)
from tests.test_frontend import A1, A2, A3, TEXT, _report, client  # noqa: F401

PAGE = entity_ref("people", "Yusuf Qadir", "Yusuf Qadir", [A1, A2, A3])
ORIGIN = {"origin": "http://testserver"}
FORCE_FED = "He was force-fed for several weeks."


@pytest.fixture
def corr(tmp_path, monkeypatch):
    path = tmp_path / "corrections" / "corrections.jsonl"
    monkeypatch.setenv("HINBOX_CORRECTIONS_PATH", str(path))
    return path


def _lines(path) -> list:
    return [json.loads(x) for x in path.read_text().splitlines()]


# ── the file ─────────────────────────────────────────────────


def test_path_comes_from_env_then_config_then_default(monkeypatch):
    monkeypatch.delenv("HINBOX_CORRECTIONS_PATH", raising=False)
    assert corrections_path("gtmo") == "data/gtmo/corrections/corrections.jsonl"
    assert corrections_path("gtmo", {"corrections": {"path": "x.jsonl"}}) == "x.jsonl"
    monkeypatch.setenv("HINBOX_CORRECTIONS_PATH", "y.jsonl")
    assert corrections_path("gtmo", {"corrections": {"path": "x.jsonl"}}) == "y.jsonl"


def test_append_numbers_signs_and_checks(corr, monkeypatch):
    monkeypatch.setenv("HINBOX_REVIEWER", "A. Historian")
    a = append_correction(
        str(corr), {"kind": "strike", "entity": PAGE, "sentence": "X."}
    )
    b = append_correction(str(corr), {"kind": "rename", "entity": PAGE, "name": "Y"})
    assert (a["id"], b["id"]) == ("c-0001", "c-0002")
    assert a["by"] == "A. Historian" and a["made"].endswith("Z")
    assert [r["id"] for r in _lines(corr)] == ["c-0001", "c-0002"]
    with pytest.raises(ValueError, match="sentence is missing"):
        append_correction(str(corr), {"kind": "strike", "entity": PAGE})
    assert len(_lines(corr)) == 2  # a refused correction writes nothing


def test_validate_catches_bad_corrections():
    def errs(**kw: Any) -> str:
        return "; ".join(validate({"entity": PAGE, **kw}))

    assert "unknown kind" in errs(kind="delete")
    assert "verdict must be" in errs(
        kind="verdict", sentence="X.", article_id=A1, verdict="maybe"
    )
    assert "one line" in errs(
        kind="strike", sentence="X.", replacement=f"Y.^[{A1}]", replacement_cites=A1
    )
    assert "needs a source" in errs(kind="strike", sentence="X.", replacement="Y.")
    assert "own sources" in errs(kind="split", article_ids=["elsewhere"])
    assert "at least one" in errs(kind="split", article_ids=[A1, A2, A3])
    assert "itself" in errs(kind="join", other=PAGE)
    assert "already in that table" in errs(kind="wrong_table", to_table="people")
    assert errs(kind="split", article_ids=[A3]) == ""


def test_withdraw_and_replace_keep_every_line(corr):
    p = str(corr)
    one = append_correction(p, {"kind": "strike", "entity": PAGE, "sentence": "X."})
    two = append_correction(
        p, {"kind": "strike", "entity": PAGE, "sentence": "X.", "reason": "again"}
    )
    three = append_correction(p, {"kind": "rename", "entity": PAGE, "name": "Y"})
    append_correction(
        p, {"kind": "withdraw", "withdraws": three["id"], "reason": "wrong"}
    )
    cs = load_corrections(p)
    assert len(cs.lines) == 4
    assert cs.state(one) == "replaced" and cs.replaced[one["id"]] == two["id"]
    assert cs.state(three) == "withdrawn"
    assert [r["id"] for r in cs.active] == [two["id"]]


def test_unreadable_lines_are_listed_not_dropped(corr):
    corr.parent.mkdir(parents=True)
    corr.write_text(
        'not json\n{"id": "c-0001", "kind": "strike"}\n{"id": "c-0002", "kind": "withdraw", "withdraws": "c-0009"}\n'
    )
    cs = load_corrections(str(corr))
    assert len(cs.problems) == 3 and not cs.active
    assert "line 1" in cs.problems[0] and "c-0001" in cs.problems[1]
    assert "isn't in the file" in cs.problems[2]


def test_split_undone_by_joins_is_a_contradiction(corr):
    p = str(corr)
    other = entity_ref("people", "Y. Qadir", "Y. Qadir", ["b1"])
    append_correction(p, {"kind": "split", "entity": PAGE, "article_ids": [A3]})
    assert not load_corrections(p).contradictions
    # A3's man is the same as "Y. Qadir", and so is the rest of the page:
    # then A3 is back with the rest, which the split said it isn't
    third = entity_ref("people", "Qadir (A3)", "Qadir", [A3])
    append_correction(p, {"kind": "join", "entity": third, "other": other})
    assert not load_corrections(p).contradictions
    append_correction(p, {"kind": "join", "entity": PAGE, "other": other})
    found = load_corrections(p).contradictions
    assert len(found) == 1 and "c-0001" in found[0]


# ── drawing them ─────────────────────────────────────────────


def _rec(cid: str, **kw: Any) -> Dict[str, Any]:
    return {"id": cid, "entity": PAGE, **kw}


def test_overlay_strikes_writes_and_overrules():
    ov = Overlay()
    ov.struck[sentence_key(FORCE_FED)] = _rec("c-0001", kind="strike", sentence=FORCE_FED,
        replacement="He was *not* force-fed [says] <b>A3</b>.", replacement_cites=A3)  # fmt: skip
    shipping = "He worked for a shipping firm"
    ov.verdicts[(sentence_key(shipping), A1)] = _rec(
        "c-0002", kind="verdict", sentence=shipping, article_id=A1, verdict="supported"
    )
    ov.verdicts[("Gone sentence.", A1)] = _rec(
        "c-0003",
        kind="verdict",
        sentence="Gone sentence.",
        article_id=A1,
        verdict="partial",
    )
    rp = render_profile(TEXT, _report(), overlay=ov)
    by_text = {c.text: c for c in rp.claims}
    struck = by_text[FORCE_FED]
    assert struck.correction == "struck" and struck.verdict == "struck"
    mine = rp.claims[struck.index + 1]
    assert mine.correction == "mine" and mine.text.startswith("He was *not*")
    assert mine.articles == [A3] and mine.verdict == "unchecked"
    # The historian's words are shown as typed, never as Markdown or HTML
    assert "He was *not* force-fed [says] &lt;b&gt;A3&lt;/b&gt;." in rp.html
    assert "<em>" not in rp.html and "<b>A3" not in rp.html
    ship = by_text[shipping]
    assert ship.verdicts[A1] == "supported" and ship.checker[A1] == "partial"
    assert ship.correction == "verdict" and ship.verdict == "supported"
    assert [u["id"] for u in rp.unmatched] == ["c-0003"]
    assert 'class="s v-struck c-struck"' in rp.html and "c-mine" in rp.html
    # Without an overlay nothing changes
    plain = render_profile(TEXT, _report())
    assert (
        not any(c.correction for c in plain.claims)
        and len(plain.claims) == len(rp.claims) - 1
    )


# ── the reader ───────────────────────────────────────────────


def _sentence_form(client, c: int) -> Dict[str, str]:  # noqa: F811
    page = client.get(f"/correct/sentence?t=people&k=Yusuf%20Qadir&c={c}").text
    sent = unescape(re.search(r'name="sentence" value="([^"]*)"', page).group(1))
    return {"t": "people", "k": "Yusuf Qadir", "c": str(c), "sentence": sent}


def _index(client, text: str) -> int:  # noqa: F811
    page = client.get("/correct/page?t=people&k=Yusuf%20Qadir").text
    for m in re.finditer(r'c=(\d+)">([^<]*)</a>', page):
        if unescape(m.group(2)) == text:
            return int(m.group(1))
    raise AssertionError(text)


def test_strike_from_the_reader(client, corr):  # noqa: F811
    c = _index(client, FORCE_FED)
    form = {**_sentence_form(client, c), "kind": "strike", "reason": "Article says nothing of it",
            "replacement": "He refused food for several weeks.", "replacement_cites": A3}  # fmt: skip
    r = client.post("/corrections", data=form, headers=ORIGIN, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].endswith(f"#claim-{c}")
    rec = _lines(corr)[0]
    assert (
        rec["kind"] == "strike"
        and rec["sentence"] == FORCE_FED
        and rec["cites"] == [A3]
    )
    page = client.get("/people/Yusuf%20Qadir").text
    assert "c-struck" in page and "He refused food for several weeks." in page
    assert "Your corrections on this page" in page and "Shown on the page" in page
    panel = client.get(f"/panel?t=people&k=Yusuf%20Qadir&a={A3}&c={c + 1}").text
    assert "Your sentence (c-0001)" in panel
    assert "1 correction" in client.get("/").text


def test_overrule_the_checker(client, corr):  # noqa: F811
    c = _index(client, FORCE_FED)
    form = {**_sentence_form(client, c), "kind": "verdict", "article_id": A3, "verdict": "supported",
            "passage": "Some were force-fed."}  # fmt: skip
    assert client.post("/corrections", data=form, headers=ORIGIN).status_code == 200
    rec = _lines(corr)[0]
    assert rec["checker_verdict"] == "not_supported" and rec["verdict"] == "supported"
    page = client.get("/people/Yusuf%20Qadir").text
    assert f'id="claim-{c}"' in page and "v-supported c-verdict" in page
    panel = client.get(f"/panel?t=people&k=Yusuf%20Qadir&a={A3}&c={c}").text
    assert "your verdict, c-0001" in panel and "The checker said not supported" in panel


def test_posts_need_the_readers_own_page(client, corr):  # noqa: F811
    form = {**_sentence_form(client, 0), "kind": "strike"}
    assert client.post("/corrections", data=form).status_code == 403
    evil = {"origin": "http://evil.example"}
    assert client.post("/corrections", data=form, headers=evil).status_code == 403
    assert not corr.exists()


def test_a_changed_page_is_refused(client, corr):  # noqa: F811
    form = {**_sentence_form(client, 0), "kind": "strike"}
    form["sentence"] = "Something the page no longer says."
    r = client.post("/corrections", data=form, headers=ORIGIN)
    assert r.status_code == 400 and "page changed" in r.text
    assert not corr.exists()


def test_page_corrections_wait_for_a_run_and_can_be_withdrawn(client, corr):  # noqa: F811
    base = {"t": "people", "k": "Yusuf Qadir"}
    for extra in ({"kind": "split", "article_ids": A3}, {"kind": "join", "other": "Miriam Hale"},
                  {"kind": "wrong_table", "to_table": "none"}, {"kind": "rename", "name": "Yusuf al-Qadir"}):  # fmt: skip
        r = client.post("/corrections", data={**base, **extra}, headers=ORIGIN)
        assert r.status_code == 200, r.text
    kinds = [r["kind"] for r in _lines(corr)]
    assert kinds == ["split", "join", "not_entity", "rename"]
    assert _lines(corr)[1]["other"]["key"] == "Miriam Hale"
    listing = client.get("/corrections").text
    assert listing.count("carry out this kind yet") == 4
    r = client.post(
        "/corrections/withdraw", data={"id": "c-0004", "reason": "typo"}, headers=ORIGIN
    )
    assert r.status_code == 200
    assert _lines(corr)[-1] == {
        **_lines(corr)[-1],
        "kind": "withdraw",
        "withdraws": "c-0004",
    }
    assert "Withdrawn by c-0005: typo" in client.get("/corrections").text
    bad = client.post(
        "/corrections", data={**base, "kind": "join", "other": "Nobody"}, headers=ORIGIN
    )
    assert bad.status_code == 400 and "No single person page" in bad.text


def test_static_copy_shows_corrections_without_buttons(client, corr, tmp_path):  # noqa: F811
    from src.frontend.export import export_site

    form = {**_sentence_form(client, _index(client, FORCE_FED)), "kind": "strike"}
    client.post("/corrections", data=form, headers=ORIGIN)
    out = tmp_path / "site"
    assert export_site(str(out))["broken_links"] == 0
    page = (out / "people" / "Yusuf Qadir.html").read_text()
    assert "c-struck" in page and "Your corrections on this page" in page
    assert (
        "Correct this" not in page
        and "correct/" not in page
        and 'method="post"' not in page
    )


def test_corrections_files_are_never_committed(tmp_path):
    """The file quotes sentences written from articles and the repo is public."""
    import subprocess

    for path in (
        "data/guantanamo/corrections/corrections.jsonl",
        "elsewhere/corrections.jsonl",
    ):
        r = subprocess.run(["git", "check-ignore", "-q", path], capture_output=True)
        assert r.returncode == 0, f"{path} is not gitignored"


def test_static_copy_never_holds_the_file(client, corr, tmp_path):  # noqa: F811
    from src.frontend.export import export_site

    form = {**_sentence_form(client, 0), "kind": "strike", "reason": "private note"}
    client.post("/corrections", data=form, headers=ORIGIN)
    out = tmp_path / "site"
    export_site(str(out))
    files = [p for p in out.rglob("*") if p.is_file()]
    assert not [p for p in files if "correction" in p.name]
    assert not any("private note" in p.read_text(errors="ignore") for p in files)


def test_static_panel_data_says_whose_verdict_it_is(client, corr, tmp_path):  # noqa: F811
    """The static copy builds its panel from `cite_data()`: an overruled
    verdict must carry the checker's own verdict and the correction id, so
    the panel can't show the historian's verdict beside the checker's
    reasoning as if the checker had said it."""
    from src.frontend.export import export_site

    c = _index(client, FORCE_FED)
    form = {
        **_sentence_form(client, c),
        "kind": "verdict",
        "article_id": A3,
        "verdict": "supported",
    }
    client.post("/corrections", data=form, headers=ORIGIN)
    out = tmp_path / "site"
    export_site(str(out))
    page = (out / "people" / "Yusuf Qadir.html").read_text()
    data = json.loads(re.search(r'id="cite-data">(.*?)</script>', page).group(1))[
        str(c)
    ]
    assert data["v"][A3] == "supported" and data["c"][A3] == "not_supported"
    assert data["x"] == "verdict" and data["i"][A3] == "c-0001"
    js = (out / "static" / "reader.js").read_text()
    assert "your verdict, " in js and "The checker said " in js
