"""Joining places and organizations a run saved twice (``src/engine/kind_twins.py``)."""

from typing import Any, Dict, List

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


def test_join_pair_fails_when_an_update_does_not_link_the_article():
    keeper = _row("Camp X", "detention_facility", ["a1"])
    donor = _row("Camp X", "LocationType.DETENTION_FACILITY", ["a2", "a3"])
    calls: List[str] = []
    texts = {"a2": "Two.", "a3": "Three."}
    joined, report = join_pair(
        "locations", keeper, donor, texts, "ollama", "guantanamo",
        update=_fake_update(calls, fail_on="a2"),
    )  # fmt: skip
    assert joined is None and "a2" in report["failed"] and calls == ["a2"]


def test_join_pair_fails_without_the_article_text():
    keeper, donor = _row("Y", "city", ["a1"]), _row("Y", "city", ["a2"])
    joined, report = join_pair(
        "locations", keeper, donor, {}, "ollama", "guantanamo",
        update=_fake_update([]),
    )  # fmt: skip
    assert joined is None and "no text" in report["failed"]


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


def test_script_joins_saves_and_drops_the_donor(tmp_path, monkeypatch):
    import pyarrow.parquet as pq

    from scripts import join_kind_twins as script
    from src.utils.file_ops import write_entities_table

    write_entities_table("locations", [
        _row("Camp X", "detention_facility", ["a1", "a2"]),
        _row("Camp X", "LocationType.DETENTION_FACILITY", ["a3"]),
        _row("Havana", "city", ["a4"]),
    ], str(tmp_path))  # fmt: skip
    write_entities_table("organizations", [_row("ACLU", "ngo", ["a5"])], str(tmp_path))
    calls: List[str] = []
    monkeypatch.setattr(script, "_article_texts", lambda *a: {"a3": "Three."})
    monkeypatch.setattr(
        script, "_embed", lambda row: row.update(profile_embedding=[1.0])
    )
    monkeypatch.setattr(
        script, "join_pair",
        lambda *a, **kw: join_pair(*a, **kw, update=_fake_update(calls)),
    )  # fmt: skip
    monkeypatch.setattr("sys.argv", ["x", "--run", str(tmp_path)])
    assert script.main() == 0
    rows = pq.read_table(tmp_path / "locations.parquet").to_pylist()
    assert [r["name"] for r in rows] == ["Camp X", "Havana"]
    assert rows[0]["type"] == "detention_facility" and calls == ["a3"]
    assert rows[0]["profile"]["sources"] == ["a1", "a2", "a3"]
    assert script.main() == 0 and calls == ["a3"]  # a rerun has nothing left
