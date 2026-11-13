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
    split_home,
)
from src.engine.corrections_pass import (
    configure_corrections,
    current_rules,
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
    assert out["changed"] == 0 and len(entities["people"]["Jane Lee"]["articles"]) == 3


def test_a_join_keeps_the_articles_that_went_in_when_one_fails(tmp_path):
    """One of the smaller page's two articles has no text: the other is fed
    in and kept, both pages stay, and the next run asks only for the rest."""
    _file(tmp_path, _join())
    entities = _lee_entities()
    j = entities["people"]["J. Lee"]
    j["articles"].append(_art("j2"))
    j["profile"]["sources"].append("j2")
    out = _carry(entities)
    state, why = out["states"]["c-0001"]
    assert state == "waiting" and "1 articles fed in" in why and "j2" in why
    jane = entities["people"]["Jane Lee"]
    assert [a["article_id"] for a in jane["articles"]] == ["b1", "b2", "b3", "j1"]
    assert "J. Lee" in entities["people"] and out["changed"] == 1
    assert "J. Lee" not in jane["alternative_names"]
    # with the text there, the next run feeds in only j2 and finishes the join
    texts = {**TEXTS, "j2": "J. Lee was mentioned twice."}
    out = carry_out_merges(
        entities, texts, "gemini", "guantanamo", set(),
        update=fake_update, embed=_no_embed,
    )  # fmt: skip
    assert out["states"]["c-0001"][0] == "applied"
    assert "1 articles fed in" in out["states"]["c-0001"][1]
    assert set(entities["people"]) == {"Jane Lee"}


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


def test_reader_finds_a_verdict_on_the_page_a_split_moved_its_source_to(
    client, corr, tmp_path
):  # noqa: F811
    """A historian overruled the checker on a sentence citing A3, on "Yusuf
    Qadir". A split moves A3 to a new page: the reader shows the verdict
    there, as the pipeline finds it, and not "the page can't be found"."""
    from src.frontend import corrections_view as cv
    from src.frontend import data_access
    from src.process_and_extract import write_entities_to_files
    from src.utils.file_ops import load_existing_entities
    from tests.test_frontend import A3

    out = tmp_path / "entities"
    entities = load_existing_entities(str(out))
    qadir = entities["people"]["Yusuf Qadir"]
    ids = [a["article_id"] for a in qadir["articles"]]
    ref = entity_ref("people", "Yusuf Qadir", "Yusuf Qadir", ids)
    sentence = "Yusuf Qadir, a trader, was released."
    append_correction(str(corr), {"kind": "verdict", "entity": ref, "sentence": sentence,
                                  "article_id": A3, "verdict": "supported"})  # fmt: skip
    append_correction(str(corr), {"kind": "split", "entity": ref, "article_ids": [A3],
                                  "name": "Yusuf Qadir (trader)"})  # fmt: skip
    configure_corrections(load_corrections(str(corr)))
    qadir["search_embedding"] = [0.5, 0.5, 0.0]
    carry_out_merges(
        entities, {A3: sentence}, "gemini", "guantanamo", set(),
        update=fake_update, embed=_no_embed,
    )  # fmt: skip
    assert not entities["people"]["Yusuf Qadir"]["search_embedding"]  # stale evidence
    write_entities_to_files(entities, str(out))
    data_access._CACHE.clear()
    assert client.get("/corrections").status_code == 200
    store, cs = data_access.load_store(), load_corrections(str(corr))
    verdict = next(r for r in cs.active if r["kind"] == "verdict")
    new, old = (
        store.get("people", "Yusuf Qadir (trader)"),
        store.get("people", "Yusuf Qadir"),
    )
    assert cv.find_page(store, cs, verdict) is new
    assert verdict in cv.for_page(store, cs, new) and verdict not in cv.for_page(
        store, cs, old
    )
    tone, words = cv.status(store, cs, verdict, {})
    assert tone == "ok", words


def test_a_ticked_article_seen_again_lands_on_the_new_page(tmp_path):
    """On the full-run copy the match check could turn the new page away and
    a third page, "... (test) (2)", was created."""
    _file(tmp_path, _split())
    entities = _ann_entities()
    _carry(entities)
    asked: List[str] = []

    def check(new, existing, *a, **kw):
        asked.append(existing)
        return MatchCheckResult(is_match=False, confidence=0.9, reason="stub")

    def merge(aid: str):
        with (
            patch("src.engine.mergers.create_profile", side_effect=create_profile_stub),
            patch(
                "src.engine.mergers.get_embedding_manager",
                return_value=StubEmbeddingManager(vec=[1.0, 0.0, 0.0]),
            ),
            patch("src.engine.mergers.cloud_model_check_match", side_effect=check),
        ):
            return EntityMerger("people").merge_entities(
                [{"name": "Ann Lee", "type": "journalist"}], entities, aid, "t", "u",
                "2012-01-01", TEXTS[aid], "x", similarity_threshold=0.8,
            )  # fmt: skip

    assert merge("a3").merged == 1 and merge("a1").merged == 1
    assert asked == []  # each side goes home without a match check
    assert set(entities["people"]) == {"Ann Lee", "Ann Lee (reporter)"}
    assert split_home("people", "a9", "Ann Lee", entities["people"]) is None


def test_a_shorter_split_name_is_not_swapped_onto_the_old_page(tmp_path):
    """The new page is named "A. Lee"; a ticked article seen again says "Ann
    Lee", the fuller name. The page keeps its name and "Ann Lee" its page."""
    _file(tmp_path, _split("A. Lee"))
    entities = _ann_entities()
    _carry(entities)
    assert set(entities["people"]) == {"Ann Lee", "A. Lee"}
    with (
        patch("src.engine.mergers.create_profile", side_effect=create_profile_stub),
        patch(
            "src.engine.mergers.get_embedding_manager",
            return_value=StubEmbeddingManager(vec=[1.0, 0.0, 0.0]),
        ),
    ):
        stats = EntityMerger("people").merge_entities(
            [{"name": "Ann Lee", "type": "journalist"}], entities, "a3", "t", "u",
            "2012-01-01", TEXTS["a3"], "x", similarity_threshold=0.8,
        )  # fmt: skip
    people = entities["people"]
    assert stats.merged == 1 and set(people) == {"Ann Lee", "A. Lee"}
    assert [a["article_id"] for a in people["Ann Lee"]["articles"]] == ["a1", "a2"]
    assert people["A. Lee"]["name"] == "A. Lee"


def test_a_second_split_of_the_same_page_divides_the_page_the_first_made(tmp_path):
    """Both splits were made on "Ann Lee": a3 and a4 to a reporter, then a4
    to a trader. The second finds a4 on the reporter's page."""
    ref = entity_ref("people", "Ann Lee", "Ann Lee", ["a1", "a2", "a3", "a4"])
    second = {"kind": "split", "entity": ref, "article_ids": ["a4"], "name": "Ann Lee (trader)"}  # fmt: skip
    _file(tmp_path, _split(), second)
    entities = _ann_entities()
    out = _carry(entities)
    assert {s for s, _ in out["states"].values()} == {"applied"}
    people = entities["people"]
    held = {k: [a["article_id"] for a in e["articles"]] for k, e in people.items()}
    assert held == {
        "Ann Lee": ["a1", "a2"],
        "Ann Lee (reporter)": ["a3"],
        "Ann Lee (trader)": ["a4"],
    }
    # each article's "Ann Lee" has one home, and the others are kept off it
    for aid, home in (("a1", "Ann Lee"), ("a3", "Ann Lee (reporter)"), ("a4", "Ann Lee (trader)")):  # fmt: skip
        assert split_home("people", aid, "Ann Lee", people)[0] == home
    assert split_block("people", "a4", "Ann Lee", people["Ann Lee (reporter)"])
    assert {s for s, _ in _carry(entities, done={"c-0001", "c-0002"})["states"].values()} == {"held"}  # fmt: skip


def test_a_split_of_a_page_a_split_made_sends_each_article_home(tmp_path):
    """a3 and a4 went to "A. Lee"; a later split, made on "A. Lee", sent a4
    to "Anne Lee". An "Ann Lee" from a4 goes there, not back beside a3."""
    later = {
        "kind": "split",
        "entity": entity_ref("people", "A. Lee", "A. Lee", ["a3", "a4"]),
        "article_ids": ["a4"],
        "name": "Anne Lee",
    }
    _file(tmp_path, _split("A. Lee"), later)
    entities = _ann_entities()
    assert {s for s, _ in _carry(entities)["states"].values()} == {"applied"}
    people = entities["people"]
    held = {k: [a["article_id"] for a in e["articles"]] for k, e in people.items()}
    assert held == {"Ann Lee": ["a1", "a2"], "A. Lee": ["a3"], "Anne Lee": ["a4"]}
    for aid, home in (("a1", "Ann Lee"), ("a3", "A. Lee"), ("a4", "Anne Lee")):
        assert split_home("people", aid, "Ann Lee", people)[0] == home


def test_a_strike_made_on_the_joined_in_page_holds_on_the_kept_page(tmp_path):
    """ "J. Lee" had a struck sentence with the historian's own in its place.
    After the join the page is gone: the strike is about "Jane Lee" now, the
    guard knows it while the article is fed in, and the sentence is kept."""
    strike = {
        "kind": "strike",
        "entity": entity_ref("people", "J. Lee", "J. Lee", ["j1"]),
        "sentence": "Jane Lee again.",
        "cites": ["j1"],
        "replacement": "J. Lee was named in a filing.",
        "replacement_cites": "j1",
    }
    path = _file(tmp_path, _join(), strike)
    entities = _lee_entities()
    seen: List[Any] = []

    def update(singular, name, profile, *a, **kw):
        seen.append(current_rules())
        return fake_update(singular, name, profile, *a, **kw)

    early = carry_out_merges(
        entities, TEXTS, "gemini", "guantanamo", set(), update=update, embed=_no_embed
    )
    assert seen[0] is not None and "jane lee again" in " ".join(seen[0].struck).lower()
    report = run_corrections_pass(entities, path, str(tmp_path), early=early)
    states = {r["kind"]: r["state"] for r in report["rows"]}
    assert states == {"join": "applied", "strike": "applied"}, report["rows"]
    text = entities["people"]["Jane Lee"]["profile"]["text"]
    assert "again" not in text and "J. Lee was named in a filing.^[j1]" in text


def test_a_strike_follows_its_sources_to_the_page_a_split_made(tmp_path):
    """A strike on "Ann Lee" replaced a sentence citing a3. A later split
    moves a3 away: the historian's sentence goes with it, and is not put back
    on the old page citing an article that page no longer holds."""
    strike = {
        "kind": "strike",
        "entity": entity_ref("people", "Ann Lee", "Ann Lee", ["a1", "a2", "a3", "a4"]),
        "sentence": "She wrote about the camp.",
        "cites": ["a3"],
        "replacement": "She reported from the camp.",
        "replacement_cites": "a3",
    }
    path = _file(tmp_path, strike)
    entities = _ann_entities()
    first = run_corrections_pass(entities, path, str(tmp_path))
    assert first["counts"] == {"applied": 1}
    append_correction(path, _split())
    configure_corrections(load_corrections(path))
    seen: List[Any] = []

    def update(singular, name, profile, *a, **kw):
        seen.append((name, current_rules()))
        return fake_update(singular, name, profile, *a, **kw)

    early = carry_out_merges(
        entities, TEXTS, "gemini", "guantanamo", {"c-0001"}, update=update, embed=_no_embed
    )  # fmt: skip
    assert seen and all(rules is not None for _, rules in seen)  # guarded
    report = run_corrections_pass(entities, path, str(tmp_path), early=early)
    people = entities["people"]
    old, new = people["Ann Lee"]["profile"]["text"], people["Ann Lee (reporter)"]["profile"]["text"]  # fmt: skip
    assert "a3" not in old and "reported from the camp" not in old
    assert "She reported from the camp.^[a3]" in new
    assert {r["kind"]: r["state"] for r in report["rows"]}["strike"] in ("held", "re-applied")  # fmt: skip


def test_two_joins_of_one_page_put_all_three_together(tmp_path):
    """ "J. Lee is Jane Lee" and "J. Lee is Jan Lee" both wait for a run. The
    first removes "J. Lee"; the second then joins the two pages left, and a
    later "J. Lee" goes to the one page."""
    other = {
        "kind": "join",
        "entity": entity_ref("people", "J. Lee", "J. Lee", ["j1"]),
        "other": entity_ref("people", "Jan Lee", "Jan Lee", ["a9"]),
    }
    _file(tmp_path, _join(), other)
    entities = _lee_entities()
    entities["people"]["Jan Lee"] = _person("Jan Lee", ["a9"], "Jan Lee spoke.^[a9]")
    out = _carry(entities)
    assert {s for s, _ in out["states"].values()} == {"applied"}, out["states"]
    people = entities["people"]
    assert set(people) == {"Jane Lee"}
    assert {a["article_id"] for a in people["Jane Lee"]["articles"]} == {"b1", "b2", "b3", "j1", "a9"}  # fmt: skip
    assert join_target("people", "J. Lee", people)[0] == "Jane Lee"
    assert join_target("people", "Jan Lee", people)[0] == "Jane Lee"
    again = _carry(entities, done={"c-0001", "c-0002"})
    assert {s for s, _ in again["states"].values()} == {"held"}
