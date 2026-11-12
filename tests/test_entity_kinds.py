"""One spelling per entity kind, so the same place keys the same way.

A kind reached the merger as an Enum (fresh extraction), as
"LocationType.MILITARY_BASE" (extraction cache) or as "military_base" (saved
table). Keys built from different spellings never met, and the 12 Nov full run
saved 51 locations and 60 organizations twice ("United States", "Pentagon").
"""

import json

import pyarrow as pa
import pyarrow.parquet as pq

from src.dynamic_models import get_location_model
from src.utils.entity_keys import plain_kind, plain_kinds
from src.utils.extraction_cache import build_cache_record
from src.utils.file_ops import load_existing_entities, sanitize_for_parquet


def _fresh():
    return get_location_model("guantanamo")(name="Camp 7", type="detention_facility")


def test_every_spelling_of_a_kind_is_the_configured_string():
    enum_value = _fresh().model_dump()["type"]
    for value in (enum_value, "LocationType.DETENTION_FACILITY", "detention_facility"):
        assert plain_kind(value) == "detention_facility"
    assert plain_kind("EventType.POLICY_CHANGE") == "policy_change"
    assert plain_kind("U.S. base") == "U.S. base"  # not an enum repr
    assert plain_kind(None) is None


def test_plain_kinds_copies_and_leaves_other_fields():
    row = {"name": "Camp 7", "type": "LocationType.DETENTION_FACILITY", "n": 1}
    out = plain_kinds(row)
    assert out == {"name": "Camp 7", "type": "detention_facility", "n": 1}
    assert row["type"] == "LocationType.DETENTION_FACILITY"


def test_cache_records_store_the_kind_value_not_its_repr():
    record = build_cache_record(
        output=[_fresh(), _fresh().model_dump()],
        model="m",
        entity_type="locations",
        temperature=0,
        content_hash="c",
        prompt_hash="p",
        schema_hash="s",
        cache_version=2,
    )
    stored = json.loads(json.dumps(record, default=str))
    assert [o["type"] for o in stored["output"]] == ["detention_facility"] * 2


def test_parquet_writer_saves_an_enum_as_its_value():
    value = _fresh().model_dump()["type"]
    assert sanitize_for_parquet({"name": "x", "type": value})["type"] == (
        "detention_facility"
    )


def _write(tmp_path, rows):
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "locations.parquet")
    return load_existing_entities(str(tmp_path))["locations"]


def test_a_saved_enum_repr_keys_like_the_plain_kind(tmp_path):
    loaded = _write(tmp_path, [{"name": "Camp 7", "type": "LocationType.CITY"}])
    assert list(loaded) == [("Camp 7", "city")]
    assert loaded[("Camp 7", "city")]["type"] == "city"


def test_rows_saved_twice_under_two_spellings_are_both_kept(tmp_path):
    rows = [
        {"name": "Pentagon", "type": "LocationType.BUILDING", "n": 1},
        {"name": "Pentagon", "type": "building", "n": 2},
        {"name": "Pentagon", "type": "building", "n": 3},
    ]
    loaded = _write(tmp_path, rows)
    assert sorted(e["n"] for e in loaded.values()) == [1, 2, 3]
    assert loaded[("Pentagon", "building")]["n"] == 1
    # The second row keeps its saved spelling, so saving and the reader still
    # tell it from the first; only an exact repeat (already one key) can't be.
    assert loaded[("Pentagon", "building", 2)]["type"] == "building"
    assert loaded[("Pentagon", "building")]["type"] == "building"


def test_a_kept_apart_row_still_has_its_own_reader_key_after_saving(tmp_path):
    from src.utils.entity_keys import KEY_FUNCS
    from src.utils.file_ops import write_entities_table

    rows = [
        {"name": "Pentagon", "type": "building", "n": 1},
        {"name": "Pentagon", "type": "LocationType.BUILDING", "n": 2},
    ]
    loaded = _write(tmp_path, rows)
    write_entities_table("locations", list(loaded.values()), str(tmp_path))
    saved = pq.read_table(tmp_path / "locations.parquet").to_pylist()
    assert len({KEY_FUNCS["locations"](r) for r in saved}) == 2


def test_a_cached_and_a_fresh_extraction_of_one_place_make_one_entity():
    """The match check says "different" here, as it can on a hub; before the
    fix the same-key guard missed (enum vs repr) and a second row was made."""
    from unittest.mock import patch

    from src.engine import EntityMerger, MatchCheckResult
    from tests.test_entity_merger_merge_smoke import StubEmbeddingManager

    entities = {"people": {}, "organizations": {}, "locations": {}, "events": {}}
    cached = {"name": "Camp 7", "type": "LocationType.DETENTION_FACILITY"}
    fresh = _fresh().model_dump()
    no = MatchCheckResult(is_match=False, reason="different", confidence=0.9)
    with (
        patch(
            "src.engine.mergers.get_embedding_manager",
            return_value=StubEmbeddingManager(),
        ),
        patch("src.engine.mergers.cloud_model_check_match", return_value=no),
        patch("src.engine.mergers.local_model_check_match", return_value=no),
    ):
        for n, item in enumerate([cached, fresh]):
            EntityMerger("locations").merge_entities(
                [item],
                entities,
                article_id=f"a{n}",
                article_title="t",
                article_url="u",
                article_published_date="2015-01-01",
                article_content="Camp 7 is a detention facility at the base.",
                extraction_timestamp="2025-01-01T00:00:00Z",
                model_type="gemini",
                similarity_threshold=0.99,
                domain="guantanamo",
            )
    assert list(entities["locations"]) == [("Camp 7", "detention_facility")]
    assert entities["locations"][("Camp 7", "detention_facility")]["type"] == (
        "detention_facility"
    )
