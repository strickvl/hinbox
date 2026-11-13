"""Joining places and organizations a run saved twice (``src/engine/kind_twins.py``)."""

from typing import Any, Dict, List

import pyarrow.parquet as pq

from src.engine import kind_twins, profile_stubs
from src.engine.kind_twins import donor_articles, find_twins, join_pair


def _row(name: str, kind: str, ids: List[str], **extra: Any) -> Dict[str, Any]:
    articles = [
        {"article_id": i, "article_published_date": f"2010-01-{n + 1:02d}"}
        for n, i in enumerate(ids)
    ]
    text = " ".join(f"Fact {i}.^[{i}]" for i in ids)
    return {
        "name": name, "type": kind, "articles": articles,
        "profile": {"text": text, "sources": list(ids)},
        "alternative_names": [], **extra,
    }  # fmt: skip


def _fake_update(calls: List[str], fail_on: str = ""):
    def update(singular, name, profile, versioned, text, article_id, *a, **kw):
        calls.append(article_id)
        if article_id == fail_on:
            return dict(profile), versioned, []
        new = {"text": f"{profile['text']} {text}^[{article_id}]",
               "sources": [*profile["sources"], article_id]}  # fmt: skip
        return new, versioned, [{"article": article_id}]

    return update


def test_find_twins_groups_kind_spellings_and_keeps_the_best_row():
    rows = [
        _row("Pentagon", "building", ["a1"]),
        _row("Pentagon", "OrganizationType.BUILDING", ["a1", "a2", "a3"]),
        _row("Pentagon", "military", ["a4"]),
        _row("Cuba", "country", ["a5"]),
    ]
    assert find_twins(rows) == [(1, [0])]


def test_donor_articles_are_the_ones_the_keeper_lacks_oldest_first():
    keeper = _row("X", "city", ["a1", "a2"])
    donor = _row("X", "city", ["a2", "a9", "a3"])
    donor["articles"][1]["article_published_date"] = "2009-05-01"
    assert [a["article_id"] for a in donor_articles(keeper, donor)] == ["a9", "a3"]


def test_join_pair_feeds_each_missing_article_in_and_keeps_the_input():
    keeper = _row("Pentagon", "OrganizationType.BUILDING", ["a1", "a2"])
    donor = _row("Pentagon", "building", ["a2", "a3"],
                 alternative_names=[{"name": "the Pentagon"}])  # fmt: skip
    calls: List[str] = []
    joined, report = join_pair(
        "organizations", keeper, donor, {"a3": "Third."}, "ollama", "guantanamo",
        update=_fake_update(calls),
    )  # fmt: skip
    assert calls == ["a3"] and report["updated"] == 1
    assert joined["type"] == "building"
    assert [a["article_id"] for a in joined["articles"]] == ["a1", "a2", "a3"]
    assert joined["profile"]["sources"] == ["a1", "a2", "a3"]
    assert joined["alternative_names"] == [{"name": "the Pentagon"}]
    assert keeper["type"] == "OrganizationType.BUILDING"
    assert len(keeper["articles"]) == 2


def test_join_pair_leaves_out_a_failed_article_and_feeds_in_the_rest():
    keeper = _row("Camp X", "detention_facility", ["a1"])
    donor = _row("Camp X", "LocationType.DETENTION_FACILITY", ["a2", "a3", "a4"])
    calls: List[str] = []
    texts = {"a2": "Two.", "a3": "Three.", "a4": "Four."}
    part, report = join_pair(
        "locations", keeper, donor, texts, "ollama", "guantanamo",
        update=_fake_update(calls, fail_on="a3"),
    )  # fmt: skip
    assert "a3" in report["failed"] and report["updated"] == 2
    assert calls == ["a2", "a3", "a4"]  # a3 asked once, a4 still fed in
    assert part["profile"]["sources"] == ["a1", "a2", "a4"]
    assert len(keeper["articles"]) == 1
    # the next run asks only for what is still missing
    assert [a["article_id"] for a in donor_articles(part, donor)] == ["a3"]
    done, report = join_pair(
        "locations", part, donor, texts, "ollama", "guantanamo",
        update=_fake_update(calls),
    )  # fmt: skip
    assert "failed" not in report and report["updated"] == 1
    assert done["profile"]["sources"] == ["a1", "a2", "a4", "a3"]


def test_join_pair_keeps_going_after_a_reply_that_raises():
    keeper, donor = _row("Y", "city", ["a1"]), _row("Y", "city", ["a2", "a3"])
    calls: List[str] = []
    ok = _fake_update(calls)

    def flaky(*a: Any, **kw: Any):
        if a[5] == "a2":
            raise ValueError("the output is incomplete")
        return ok(*a, **kw)

    part, report = join_pair(
        "locations", keeper, donor, {"a2": "Two.", "a3": "Three."}, "ollama",
        "guantanamo", update=flaky,
    )  # fmt: skip
    assert "a2 (ValueError)" in report["failed"] and calls == ["a3"]
    assert part["profile"]["sources"] == ["a1", "a3"]


def test_join_pair_fails_without_the_article_text():
    keeper, donor = _row("Y", "city", ["a1"]), _row("Y", "city", ["a2"])
    part, report = join_pair(
        "locations", keeper, donor, {}, "ollama", "guantanamo",
        update=_fake_update([]),
    )  # fmt: skip
    assert "no text" in report["failed"] and part["profile"]["sources"] == ["a1"]


def test_join_pair_only_links_a_country(monkeypatch):
    def no_model(*a: Any, **kw: Any):
        raise AssertionError("a country must not call the model")

    monkeypatch.setattr(profile_stubs, "update_profile", no_model)
    monkeypatch.setattr(profile_stubs, "create_profile", no_model)
    monkeypatch.setattr(kind_twins, "task_model", lambda *a: "ollama/test")
    keeper = _row("Cuba", "country", ["a1", "a2"])
    donor = _row("Cuba", "LocationType.COUNTRY", ["a3"])
    joined, report = join_pair(
        "locations", keeper, donor, {"a3": "Cuba news."}, "ollama", "guantanamo"
    )
    assert report["updated"] == 1
    assert joined["profile"]["text"] == keeper["profile"]["text"]
    assert joined["profile"]["sources"] == ["a1", "a2", "a3"]


def _script_run(tmp_path, monkeypatch, *flags: str, rows=None, texts=None):
    from scripts import join_kind_twins as script
    from src.utils.file_ops import write_entities_table

    if not (tmp_path / "locations.parquet").exists():
        write_entities_table("locations", rows or [
            _row("Camp X", "LocationType.DETENTION_FACILITY", ["a3"]),
            _row("Havana", "city", ["a4"]),
            _row("Camp X", "detention_facility", ["a1", "a2"]),
            _row("Delta", "LocationType.CITY", ["a6"]),
            _row("Delta", "city", ["a7", "a8"]),
            _row("Cuba", "country", ["a9", "a10"]),
            _row("Cuba", "LocationType.COUNTRY", ["a11"]),
            _row("Bad", "city", ["b1", "b2"]),
            _row("Bad", "LocationType.CITY", ["b3"]),
        ], str(tmp_path))  # fmt: skip
        write_entities_table(
            "organizations", [_row("ACLU", "ngo", ["a5"])], str(tmp_path)
        )
    calls: List[str] = []
    # no text for b3
    texts = texts or {"a3": "Three.", "a6": "Six.", "a11": "Eleven."}
    monkeypatch.setattr(script, "_article_texts", lambda *a: texts)
    monkeypatch.setattr(
        script, "_embed", lambda row: row.update(profile_embedding=[1.0])
    )
    monkeypatch.setattr(
        script, "join_pair",
        lambda *a, **kw: join_pair(*a, **kw, update=_fake_update(calls)),
    )  # fmt: skip
    loaded: List[Any] = []
    monkeypatch.setattr(script, "load_corrections", lambda path: "fixes")
    monkeypatch.setattr(script, "configure_corrections", loaded.append)
    monkeypatch.setattr("sys.argv", ["x", "--run", str(tmp_path), *flags])
    code = script.main()
    assert loaded == ["fixes"]  # the edit guard knows the corrections
    return (
        code,
        calls,
        {
            (r["name"], r["type"]): r["profile"]["sources"]
            for r in pq.read_table(tmp_path / "locations.parquet").to_pylist()
        },
    )


def test_script_joins_each_pair_after_earlier_rows_are_dropped(tmp_path, monkeypatch):
    code, calls, rows = _script_run(tmp_path, monkeypatch)
    assert code == 1  # "Bad" has no text for b3, so it is left as two rows
    assert sorted(calls) == ["a11", "a3", "a6"]
    assert rows == {
        ("Havana", "city"): ["a4"],
        ("Camp X", "detention_facility"): ["a1", "a2", "a3"],
        ("Delta", "city"): ["a7", "a8", "a6"],
        ("Cuba", "country"): ["a9", "a10", "a11"],
        ("Bad", "city"): ["b1", "b2"],
        ("Bad", "LocationType.CITY"): ["b3"],
    }
    code, calls, again = _script_run(tmp_path, monkeypatch)
    assert calls == [] and again == rows  # a rerun only retries "Bad"


def test_script_hubs_only_joins_only_the_country(tmp_path, monkeypatch):
    code, calls, rows = _script_run(tmp_path, monkeypatch, "--hubs-only")
    assert code == 0 and calls == ["a11"]
    assert rows[("Cuba", "country")] == ["a9", "a10", "a11"]
    assert ("Camp X", "LocationType.DETENTION_FACILITY") in rows


def test_script_saves_what_was_fed_in_and_a_rerun_finishes(tmp_path, monkeypatch):
    rows = [
        _row("Half", "city", ["h0", "h1", "h4"]),
        _row("Half", "LocationType.CITY", ["h2", "h3"]),
    ]
    code, calls, saved = _script_run(
        tmp_path, monkeypatch, rows=rows, texts={"h2": "Two."}
    )
    assert code == 1 and calls == ["h2"]
    assert saved == {
        ("Half", "city"): ["h0", "h1", "h4", "h2"],  # h2 kept, h3 had no text
        ("Half", "LocationType.CITY"): ["h2", "h3"],
    }
    code, calls, saved = _script_run(
        tmp_path, monkeypatch, texts={"h2": "Two.", "h3": "Three."}
    )
    assert code == 0 and calls == ["h3"]
    assert saved == {("Half", "city"): ["h0", "h1", "h4", "h2", "h3"]}
