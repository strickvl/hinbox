"""A historian's splits and joins carried out by a run (corrections step 3).

Made-up people and articles only: the repository is public.
"""

from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from src.engine import EntityMerger, MatchCheckResult
from src.engine.corrections_merges import (
    carry_out_merges,
    drop_sources,
    join_target,
    split_block,
    split_create_key,
)
from src.engine.corrections_pass import (
    configure_corrections,
    describe_report,
    run_corrections_pass,
)
from src.engine.profiles import VersionedProfile
from src.utils.corrections import append_correction, entity_ref, load_corrections
from tests.test_corrections import corr  # noqa: F401
from tests.test_entity_merger_merge_smoke import (
    StubEmbeddingManager,
    create_profile_stub,
    make_empty_entities,
)
from tests.test_frontend import client  # noqa: F401

TEXTS = {
    "a1": "Ann Lee, a lawyer, filed a motion in May. The court met.",
    "a2": "Ann Lee argued the appeal. Judges listened.",
    "a3": "Ann Lee, a reporter, wrote about the camp. Readers wrote back.",
    "a4": "Ann Lee published a second story on the camp.",
    "a9": "Ann Lee spoke again.",
    "b1": "Jane Lee joined the team.",
    "b2": "Jane Lee left the team.",
    "b3": "Jane Lee came back.",
    "j1": "J. Lee was mentioned once.",
}
DATES = {"a1": "2010-01-01", "a2": "2010-02-01", "a3": "2012-01-01", "a4": "2012-02-01"}


def _art(aid: str) -> Dict[str, Any]:
    return {
        "article_id": aid,
        "article_title": f"Title {aid}",
        "article_url": f"https://example.org/{aid}",
        "article_published_date": DATES.get(aid, "2011-01-01"),
    }


def _person(name: str, ids: List[str], text: str) -> Dict[str, Any]:
    return {
        "name": name,
        "type": "lawyer",
        "profile": {"text": text, "sources": list(ids)},
        "profile_versions": VersionedProfile().model_dump(),
        "articles": [_art(a) for a in ids],
        "alternative_names": [],
        "aliases": [],
        "profile_embedding": [1.0, 0.0, 0.0],
    }


def fake_update(singular, name, profile, versioned, text, aid, *a, **kw):
    """A profile update that adds one sentence citing the article."""
    out = {**profile, "text": profile["text"] + f"\n\n{name} again.^[{aid}]"}
    out["sources"] = [*(profile.get("sources") or []), aid]
    out.pop("stub", None)
    return out, versioned, []


def _no_embed(rows, domain):
    for r in rows:
        r["profile_embedding"] = [1.0, 0.0, 0.0]


ANN = (
    "Ann Lee is a lawyer.^[a1] She argued the appeal.^[a2] "
    "She wrote about the camp.^[a3]\n\nShe covered the hearings.^[a2, a4]"
)


def _ann_entities() -> Dict[str, Dict]:
    entities = make_empty_entities()
    entities["people"]["Ann Lee"] = _person("Ann Lee", ["a1", "a2", "a3", "a4"], ANN)
    return entities


def _file(tmp_path, *recs) -> str:
    path = str(tmp_path / "corrections.jsonl")
    for r in recs:
        append_correction(path, r)
    configure_corrections(load_corrections(path))
    return path


def _split(name: str = "Ann Lee (reporter)") -> Dict[str, Any]:
    ref = entity_ref("people", "Ann Lee", "Ann Lee", ["a1", "a2", "a3", "a4"])
    return {"kind": "split", "entity": ref, "article_ids": ["a3", "a4"], "name": name}


def _join() -> Dict[str, Any]:
    return {
        "kind": "join",
        "entity": entity_ref("people", "J. Lee", "J. Lee", ["j1"]),
        "other": entity_ref("people", "Jane Lee", "Jane Lee", ["b1", "b2", "b3"]),
    }


def _lee_entities() -> Dict[str, Dict]:
    entities = make_empty_entities()
    entities["people"]["Jane Lee"] = _person(
        "Jane Lee", ["b1", "b2", "b3"], "Jane Lee joined.^[b1] She left.^[b2, b3]"
    )
    entities["people"]["J. Lee"] = _person("J. Lee", ["j1"], "J. Lee was named.^[j1]")
    return entities


def _carry(entities, done=frozenset()):
    return carry_out_merges(
        entities, TEXTS, "gemini", "guantanamo", set(done),
        update=fake_update, embed=_no_embed,
    )  # fmt: skip


@pytest.fixture(autouse=True)
def _reset():
    yield
    configure_corrections(None)


# ── cutting a page's sources ─────────────────────────────────


def test_drop_sources_cuts_only_theirs_and_keeps_mixed_with_other_citations():
    text, cut, mixed = drop_sources(ANN, {"a3", "a4"})
    assert text == (
        "Ann Lee is a lawyer.^[a1] She argued the appeal.^[a2]\n\n"
        "She covered the hearings.^[a2]"
    )
    assert cut == ["She wrote about the camp."]
    assert mixed == ["She covered the hearings."]


def test_drop_sources_removes_a_marker_left_empty():
    text, cut, mixed = drop_sources("He won.^[a1]^[a3] He lost.^[a3]", {"a3"})
    assert text == "He won.^[a1]"
    assert cut == ["He lost."] and mixed == ["He won."]


# ── joins ────────────────────────────────────────────────────


def test_join_feeds_the_smaller_page_into_the_larger(tmp_path):
    _file(tmp_path, _join())
    entities = _lee_entities()
    out = _carry(entities)
    state, why = list(out["states"].values())[0]
    assert state == "applied" and "1 articles fed in" in why
    people = entities["people"]
    assert "J. Lee" not in people
    jane = people["Jane Lee"]
    assert [a["article_id"] for a in jane["articles"]] == ["b1", "b2", "b3", "j1"]
    assert "^[j1]" in jane["profile"]["text"]
    assert "J. Lee" in jane["alternative_names"]
    # nothing left to do on the next run
    assert list(_carry(entities, done={"c-0001"})["states"].values())[0][0] == "held"


def test_a_failed_join_changes_nothing(tmp_path):
    _file(tmp_path, _join())
    entities = _lee_entities()
    texts = {k: v for k, v in TEXTS.items() if k != "j1"}
    out = carry_out_merges(
        entities, texts, "gemini", "guantanamo", set(),
        update=fake_update, embed=_no_embed,
    )  # fmt: skip
    state, why = out["states"]["c-0001"]
    assert state == "waiting" and "no text for article j1" in why
    assert set(entities["people"]) == {"Jane Lee", "J. Lee"}


def test_joined_name_goes_straight_to_the_kept_page(tmp_path):
    _file(tmp_path, _join())
    entities = _lee_entities()
    assert join_target("people", "J. Lee", entities["people"]) is None  # not joined
    _carry(entities)
    assert join_target("people", "J. Lee", entities["people"]) == ("Jane Lee", "c-0001")
    # the kept page's own name still goes through the match check
    assert join_target("people", "Jane Lee", entities["people"]) is None


def test_merger_skips_the_match_check_for_a_joined_name(tmp_path):
    _file(tmp_path, _join())
    entities = _lee_entities()
    _carry(entities)
    asked: List[str] = []

    def check(new, existing, *a, **kw):
        asked.append(existing)
        return MatchCheckResult(is_match=False, confidence=0.9, reason="stub")

    with (
        patch("src.engine.profile_stubs.update_profile", side_effect=fake_update),
        patch(
            "src.engine.mergers.get_embedding_manager",
            return_value=StubEmbeddingManager(vec=[0.0, 1.0, 0.0]),
        ),
        patch("src.engine.mergers.cloud_model_check_match", side_effect=check),
    ):
        stats = EntityMerger("people").merge_entities(
            [{"name": "J. Lee", "type": "lawyer"}], entities, "b9", "t", "u",
            "2013-01-01", "J. Lee spoke.", "x", similarity_threshold=0.8,
        )  # fmt: skip
    assert asked == [] and stats.merged == 1
    assert "b9" in [a["article_id"] for a in entities["people"]["Jane Lee"]["articles"]]


# ── splits ───────────────────────────────────────────────────


def test_split_moves_ticked_sources_to_a_new_page(tmp_path):
    _file(tmp_path, _split())
    entities = _ann_entities()
    out = _carry(entities)
    state, why = out["states"]["c-0001"]
    assert state == "applied" and "2 sources moved" in why, why
    people = entities["people"]
    ann, other = people["Ann Lee"], people["Ann Lee (reporter)"]
    assert [a["article_id"] for a in ann["articles"]] == ["a1", "a2"]
    assert ann["profile"]["sources"] == ["a1", "a2"]
    assert "camp" not in ann["profile"]["text"]
    assert [a["article_id"] for a in other["articles"]] == ["a3", "a4"]
    # a stub from the oldest ticked source, then an update for the next
    assert "Ann Lee, a reporter, wrote about the camp.^[a3]" in other["profile"]["text"]
    assert "^[a4]" in other["profile"]["text"]
    assert out["mixed"] == [
        {
            "id": "c-0001",
            "page": "people/Ann Lee",
            "sentence": "She covered the hearings.",
        }
    ]
    versions = ann["profile_versions"]["versions"]
    assert versions[-1]["trigger_article_id"] == "corrections:c-0001"
    assert _carry(entities, done={"c-0001"})["states"]["c-0001"] == ("held", "")


def test_split_with_the_same_name_numbers_the_new_page(tmp_path):
    _file(tmp_path, _split(name=""))
    entities = _ann_entities()
    _carry(entities)
    assert set(entities["people"]) == {"Ann Lee", "Ann Lee (2)"}


def test_split_block_by_article_side_and_name(tmp_path):
    _file(tmp_path, _split())
    entities = _ann_entities()
    _carry(entities)
    ann = entities["people"]["Ann Lee"]
    other = entities["people"]["Ann Lee (reporter)"]
    assert split_block("people", "a3", "Ann Lee", ann) == "c-0001"
    assert split_block("people", "a1", "Ann Lee", other) == "c-0001"
    assert split_block("people", "a1", "Ann Lee", ann) is None
    assert split_block("people", "a9", "Ann Lee", ann) is None  # a new source
    # another entity named in the same articles still merges as usual
    camp = {"name": "Camp Delta", "articles": [_art("a1"), _art("a3")]}
    assert split_block("people", "a3", "Camp Delta", camp) is None


def test_merger_passes_over_the_split_page_without_a_model_call(tmp_path):
    _file(tmp_path, _split())
    entities = _ann_entities()  # a fresh copy, split not carried out yet
    asked: List[str] = []

    def check(new, existing, *a, **kw):
        asked.append(existing)
        return MatchCheckResult(is_match=True, confidence=0.9, reason="stub")

    with (
        patch("src.engine.mergers.create_profile", side_effect=create_profile_stub),
        patch(
            "src.engine.mergers.get_embedding_manager",
            return_value=StubEmbeddingManager(vec=[1.0, 0.0, 0.0]),
        ),
        patch("src.engine.mergers.cloud_model_check_match", side_effect=check),
    ):
        stats = EntityMerger("people").merge_entities(
            [{"name": "Ann Lee", "type": "journalist"}], entities, "a3", "t", "u",
            "2012-01-01", TEXTS["a3"], "x", similarity_threshold=0.8,
        )  # fmt: skip
    assert asked == [] and stats.new == 1
    other = entities["people"]["Ann Lee (reporter)"]
    assert "Ann Lee, a reporter" in other["profile"]["text"]  # found by her name


def test_split_create_key_only_for_a_blocked_same_name(tmp_path):
    _file(tmp_path, _split())
    pages = _ann_entities()["people"]
    assert split_create_key("people", "a3", "Ann Lee", pages) == "Ann Lee (reporter)"
    assert split_create_key("people", "a9", "Ann Lee", pages) is None
    assert split_create_key("people", "a3", "Bob Roe", pages) is None


def test_contradicting_splits_and_joins_are_not_carried_out(tmp_path):
    other = entity_ref("people", "Ann Leigh", "Ann Leigh", ["a3"])
    join = {
        "kind": "join",
        "entity": entity_ref("people", "Ann Lee", "Ann Lee", ["a1", "a2", "a3", "a4"]),
        "other": other,
    }
    path = _file(
        tmp_path, _split(), join, {**join, "entity": other, "other": join["entity"]}
    )
    cs = load_corrections(path)
    assert cs.contradictions
    entities = _ann_entities()
    out = _carry(entities)
    assert {s for s, _ in out["states"].values()} == {"needs a look"}
    assert set(entities["people"]) == {"Ann Lee"}
    assert split_block("people", "a3", "Ann Lee", entities["people"]["Ann Lee"]) is None


# ── the run's report ─────────────────────────────────────────


def test_report_carries_the_early_states(tmp_path):
    path = _file(tmp_path, _split(), _join())
    entities = _ann_entities()
    entities["people"].update(_lee_entities()["people"])
    early = _carry(entities)
    report = run_corrections_pass(entities, path, str(tmp_path / "out"), early=early)
    states = {r["id"]: r["state"] for r in report["rows"]}
    assert states == {"c-0001": "applied", "c-0002": "applied"}
    assert len(report["split_mixed"]) == 1
    assert "sentences cite sources a split moved" in describe_report(report)
    # without the early step they wait for a run
    report = run_corrections_pass(entities, path, str(tmp_path / "out2"))
    assert {r["state"] for r in report["rows"]} == {"waiting"}


# ── the reader, before and after a run ───────────────────────


def test_reader_says_what_the_run_did_with_a_split(client, corr, tmp_path):  # noqa: F811
    from src.engine import corrections_pass as cp
    from src.frontend import data_access
    from src.process_and_extract import write_entities_to_files
    from src.utils.file_ops import load_existing_entities
    from tests.test_corrections import ORIGIN
    from tests.test_frontend import A3

    form = {"t": "people", "k": "Yusuf Qadir", "kind": "split", "article_ids": A3,
            "name": "Yusuf Qadir (trader)"}  # fmt: skip
    assert client.post("/corrections", data=form, headers=ORIGIN).status_code == 200
    assert "The next pipeline run carries it out" in client.get("/corrections").text
    out = tmp_path / "entities"
    entities = load_existing_entities(str(out))
    configure_corrections(load_corrections(str(corr)))
    texts = {A3: "Yusuf Qadir, a trader, was released."}
    early = carry_out_merges(
        entities, texts, "gemini", "guantanamo", set(),
        update=fake_update, embed=_no_embed,
    )  # fmt: skip
    report = cp.run_corrections_pass(entities, str(corr), str(out), early=early)
    write_entities_to_files(entities, str(out))
    data_access._CACHE.clear()
    assert report["counts"] == {"applied": 1}
    listing = client.get("/corrections").text
    assert "Carried out by the run of" in listing and "1 sources moved" in listing
    page = client.get("/people/Yusuf%20Qadir%20(trader)")
    assert page.status_code == 200 and "a trader" in page.text
