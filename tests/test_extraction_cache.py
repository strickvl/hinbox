"""Tests for the persistent extraction sidecar cache."""

import os
from typing import List, Optional
from unittest.mock import patch

from pydantic import BaseModel

from src.utils.extraction_cache import (
    ExtractionSidecarCache,
    _schema_hash,
    build_cache_record,
)

# ---------------------------------------------------------------------------
# Minimal Pydantic model for testing schema hashing
# ---------------------------------------------------------------------------


class FakePerson(BaseModel):
    name: str
    type: str = "detainee"
    aliases: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# Schema hashing
# ---------------------------------------------------------------------------


class TestSchemaHash:
    def test_list_model_produces_stable_hash(self):
        h1 = _schema_hash(List[FakePerson])
        h2 = _schema_hash(List[FakePerson])
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_bare_model_produces_stable_hash(self):
        h1 = _schema_hash(FakePerson)
        h2 = _schema_hash(FakePerson)
        assert h1 == h2

    def test_list_vs_bare_differ(self):
        assert _schema_hash(List[FakePerson]) != _schema_hash(FakePerson)

    def test_different_models_differ(self):
        class OtherModel(BaseModel):
            title: str

        assert _schema_hash(List[FakePerson]) != _schema_hash(List[OtherModel])

    def test_fallback_for_non_pydantic(self):
        h = _schema_hash(str)
        assert isinstance(h, str)
        assert len(h) == 64


# ---------------------------------------------------------------------------
# ExtractionSidecarCache core operations
# ---------------------------------------------------------------------------


class TestExtractionSidecarCache:
    def _make_cache(self, tmp_path, **kwargs):
        defaults = {
            "base_dir": str(tmp_path),
            "subdir": "cache/extractions",
            "version": 1,
            "enabled": True,
        }
        defaults.update(kwargs)
        return ExtractionSidecarCache(**defaults)

    def test_make_key_deterministic(self, tmp_path):
        cache = self._make_cache(tmp_path)
        k1 = cache.make_key(
            text="Hello world",
            system_prompt="Extract people",
            response_model=List[FakePerson],
            model="gemini/gemini-2.0-flash",
            entity_type="people",
            temperature=0,
        )
        k2 = cache.make_key(
            text="Hello world",
            system_prompt="Extract people",
            response_model=List[FakePerson],
            model="gemini/gemini-2.0-flash",
            entity_type="people",
            temperature=0,
        )
        assert k1 == k2
        assert len(k1) == 64

    def test_make_key_sensitive_to_text(self, tmp_path):
        cache = self._make_cache(tmp_path)
        common = dict(
            system_prompt="Extract people",
            response_model=List[FakePerson],
            model="gemini/gemini-2.0-flash",
            entity_type="people",
            temperature=0,
        )
        k1 = cache.make_key(text="Article A", **common)
        k2 = cache.make_key(text="Article B", **common)
        assert k1 != k2

    def test_make_key_sensitive_to_entity_type(self, tmp_path):
        cache = self._make_cache(tmp_path)
        common = dict(
            text="Some text",
            system_prompt="Extract",
            response_model=List[FakePerson],
            model="gemini/gemini-2.0-flash",
            temperature=0,
        )
        k1 = cache.make_key(entity_type="people", **common)
        k2 = cache.make_key(entity_type="organizations", **common)
        assert k1 != k2

    def test_make_key_sensitive_to_prompt(self, tmp_path):
        cache = self._make_cache(tmp_path)
        common = dict(
            text="Some text",
            response_model=List[FakePerson],
            model="gemini/gemini-2.0-flash",
            entity_type="people",
            temperature=0,
        )
        k1 = cache.make_key(system_prompt="Prompt A", **common)
        k2 = cache.make_key(system_prompt="Prompt B", **common)
        assert k1 != k2

    def test_make_key_sensitive_to_model(self, tmp_path):
        cache = self._make_cache(tmp_path)
        common = dict(
            text="Some text",
            system_prompt="Extract",
            response_model=List[FakePerson],
            entity_type="people",
            temperature=0,
        )
        k1 = cache.make_key(model="gemini/gemini-2.0-flash", **common)
        k2 = cache.make_key(model="ollama/llama3", **common)
        assert k1 != k2

    def test_make_key_sensitive_to_version(self, tmp_path):
        c1 = self._make_cache(tmp_path, version=1)
        c2 = self._make_cache(tmp_path, version=2)
        common = dict(
            text="Some text",
            system_prompt="Extract",
            response_model=List[FakePerson],
            model="gemini/gemini-2.0-flash",
            entity_type="people",
            temperature=0,
        )
        assert c1.make_key(**common) != c2.make_key(**common)

    def test_read_miss_returns_none(self, tmp_path):
        cache = self._make_cache(tmp_path)
        assert cache.read("nonexistent_key") is None

    def test_write_then_read_roundtrip(self, tmp_path):
        cache = self._make_cache(tmp_path)
        key = cache.make_key(
            text="Hello",
            system_prompt="Extract people",
            response_model=List[FakePerson],
            model="gemini/gemini-2.0-flash",
            entity_type="people",
            temperature=0,
        )

        record = {
            "cache_version": 1,
            "entity_type": "people",
            "output": [{"name": "Alice", "type": "detainee"}],
        }
        cache.write(key, record)

        result = cache.read(key)
        assert result is not None
        assert result["output"] == [{"name": "Alice", "type": "detainee"}]

    def test_disabled_cache_skips_read_write(self, tmp_path):
        cache = self._make_cache(tmp_path, enabled=False)
        key = "some_key"
        cache.write(key, {"output": []})
        assert cache.read(key) is None

    def test_stats_tracking(self, tmp_path):
        cache = self._make_cache(tmp_path)
        key = cache.make_key(
            text="Hello",
            system_prompt="Extract",
            response_model=List[FakePerson],
            model="gemini/gemini-2.0-flash",
            entity_type="people",
            temperature=0,
        )

        cache.read("miss1")
        cache.read("miss2")
        cache.write(key, {"output": []})
        cache.read(key)

        stats = cache.stats
        assert stats["misses"] == 2
        assert stats["hits"] == 1

    def test_sharded_directory_structure(self, tmp_path):
        cache = self._make_cache(tmp_path)
        key = cache.make_key(
            text="Test",
            system_prompt="Extract",
            response_model=List[FakePerson],
            model="gemini/gemini-2.0-flash",
            entity_type="people",
            temperature=0,
        )
        cache.write(key, {"output": []})

        # Verify the sharded directory structure: v1/{k[:2]}/{k[2:4]}/{key}.json
        expected_path = os.path.join(
            str(tmp_path),
            "cache/extractions",
            "v1",
            key[:2],
            key[2:4],
            f"{key}.json",
        )
        assert os.path.exists(expected_path)


# ---------------------------------------------------------------------------
# build_cache_record
# ---------------------------------------------------------------------------


class TestBuildCacheRecord:
    def test_serializes_dicts(self):
        output = [{"name": "Alice", "type": "detainee"}]
        rec = build_cache_record(
            output=output,
            entity_type="people",
            model="gemini/gemini-2.0-flash",
            temperature=0,
            content_hash="abc",
            prompt_hash="def",
            schema_hash="ghi",
            cache_version=1,
        )
        assert rec["output"] == output
        assert rec["entity_type"] == "people"
        assert rec["cache_version"] == 1
        assert "created_at" in rec

    def test_serializes_pydantic_models(self):
        output = [FakePerson(name="Bob", type="guard")]
        rec = build_cache_record(
            output=output,
            entity_type="people",
            model="gemini/gemini-2.0-flash",
            temperature=0,
            content_hash="abc",
            prompt_hash="def",
            schema_hash="ghi",
            cache_version=1,
        )
        assert rec["output"] == [{"name": "Bob", "type": "guard", "aliases": None}]

    def test_handles_none_output(self):
        rec = build_cache_record(
            output=None,
            entity_type="people",
            model="gemini/gemini-2.0-flash",
            temperature=0,
            content_hash="abc",
            prompt_hash="def",
            schema_hash="ghi",
            cache_version=1,
        )
        assert rec["output"] == []


# ---------------------------------------------------------------------------
# Integration: configure + extract with cache
# ---------------------------------------------------------------------------


class TestExtractionCacheIntegration:
    """Test that the cache wiring in extraction.py works end-to-end."""

    def test_configure_and_reset(self, tmp_path):
        from src.utils.extraction import (
            configure_extraction_sidecar_cache,
            reset_extraction_sidecar_cache,
        )

        configure_extraction_sidecar_cache(
            base_dir=str(tmp_path),
            cache_cfg={
                "enabled": True,
                "extraction": {
                    "enabled": True,
                    "subdir": "cache/extractions",
                    "version": 1,
                },
            },
        )

        from src.utils.extraction import _EXTRACTION_CACHE as cache_after

        assert cache_after is not None
        assert cache_after.enabled

        reset_extraction_sidecar_cache()
        from src.utils.extraction import _EXTRACTION_CACHE as cache_reset

        assert cache_reset is None

    def test_configure_disabled(self, tmp_path):
        from src.utils.extraction import (
            configure_extraction_sidecar_cache,
            reset_extraction_sidecar_cache,
        )

        configure_extraction_sidecar_cache(
            base_dir=str(tmp_path),
            cache_cfg={"enabled": False},
        )

        from src.utils.extraction import _EXTRACTION_CACHE as cache_after

        assert cache_after is None
        reset_extraction_sidecar_cache()

    def test_extract_uses_cache_on_second_call(self, tmp_path):
        """Verify that a second extraction call with identical inputs returns
        from cache without calling the LLM."""
        from src.utils.extraction import (
            configure_extraction_sidecar_cache,
            extract_entities_cloud,
            reset_extraction_sidecar_cache,
        )

        configure_extraction_sidecar_cache(
            base_dir=str(tmp_path),
            cache_cfg={
                "enabled": True,
                "extraction": {
                    "enabled": True,
                    "subdir": "cache/extractions",
                    "version": 1,
                },
            },
        )

        fake_result = [FakePerson(name="Alice", type="detainee")]

        with patch("src.utils.extraction.cloud_generation") as mock_gen:
            mock_gen.return_value = fake_result

            # First call — cache miss, calls LLM
            extract_entities_cloud(
                text="Article about Alice",
                system_prompt="Extract people",
                response_model=List[FakePerson],
                model="gemini/gemini-2.0-flash",
                temperature=0,
                entity_type="people",
            )
            assert mock_gen.call_count == 1

            # Second call — cache hit, should NOT call LLM again
            r2 = extract_entities_cloud(
                text="Article about Alice",
                system_prompt="Extract people",
                response_model=List[FakePerson],
                model="gemini/gemini-2.0-flash",
                temperature=0,
                entity_type="people",
            )
            assert mock_gen.call_count == 1  # Still 1 — cache hit!

            # Verify we got cached output (list of dicts from JSON roundtrip)
            assert isinstance(r2, list)
            assert len(r2) == 1
            assert r2[0]["name"] == "Alice"

        reset_extraction_sidecar_cache()

    def test_extract_without_entity_type_bypasses_cache(self, tmp_path):
        """When entity_type is not provided, caching is bypassed."""
        from src.utils.extraction import (
            configure_extraction_sidecar_cache,
            extract_entities_cloud,
            reset_extraction_sidecar_cache,
        )

        configure_extraction_sidecar_cache(
            base_dir=str(tmp_path),
            cache_cfg={
                "enabled": True,
                "extraction": {
                    "enabled": True,
                    "subdir": "cache/extractions",
                    "version": 1,
                },
            },
        )

        with patch("src.utils.extraction.cloud_generation") as mock_gen:
            mock_gen.return_value = []

            # Call without entity_type
            extract_entities_cloud(
                text="Some text",
                system_prompt="Extract",
                response_model=List[FakePerson],
            )
            extract_entities_cloud(
                text="Some text",
                system_prompt="Extract",
                response_model=List[FakePerson],
            )
            # Both calls should hit the LLM
            assert mock_gen.call_count == 2

        reset_extraction_sidecar_cache()
