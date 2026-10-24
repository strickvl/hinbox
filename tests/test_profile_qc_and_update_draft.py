"""Tests for profile QC in the live create/update path, the update-path draft,
and Parquet writes of entities whose keys differ.

These tests verify:
- create_profile / update_profile run profile QC and record it on the profile
- The QC result is part of the saved profile version
- QC never blocks a profile: failing profiles are kept, only flagged
- Profile updates draft a real profile before the first reflection,
  instead of reflecting on an empty "{}"
- write_entities_table keeps a key that is missing from the first row
"""

import json
from unittest.mock import MagicMock, patch

from src.engine import VersionedProfile, create_profile, update_profile
from src.engine.profiles import EntityProfile, _update_profile_internal
from src.utils.file_ops import read_entities_from_file, write_entities_table

GOOD_TEXT = (
    "### Background\n\nAlice Smith is a defense lawyer who represented several "
    "detainees at Guantanamo Bay in military commission hearings.^[art-001]"
)


def _good_profile(article_id: str = "art-001") -> dict:
    return {
        "text": GOOD_TEXT,
        "tags": ["lawyer"],
        "confidence": 0.9,
        "sources": [article_id],
    }


class TestProfileQCOnCreate:
    @patch("src.engine.profiles.generate_profile")
    def test_good_profile_passes_qc(self, mock_generate):
        mock_generate.return_value = (_good_profile(), [])

        profile, versioned, _ = create_profile(
            "person", "Alice Smith", "article text", "art-001"
        )

        assert profile["qc"] == {"passed": True, "flags": []}
        assert versioned.get_latest().profile_data["qc"] == profile["qc"]

    @patch("src.engine.profiles.generate_profile")
    def test_short_profile_is_flagged_but_kept(self, mock_generate):
        mock_generate.return_value = (
            {"text": "Too short.", "tags": [], "confidence": 1.4, "sources": []},
            [],
        )

        with patch("src.engine.profiles.ENABLE_PROFILE_VERSIONING", True):
            profile, versioned, _ = create_profile(
                "person", "Alice Smith", "article text", "art-001"
            )

        assert profile["text"] == "Too short.", "QC must not replace the profile"
        assert profile["qc"]["passed"] is False
        assert set(profile["qc"]["flags"]) >= {
            "text_too_short",
            "no_citations",
            "tags_below_minimum",
            "confidence_clamped",
        }
        # QC's safe fixes are applied and are part of the saved version
        assert profile["tags"] == ["needs-review"]
        assert profile["confidence"] == 1.0
        saved = versioned.get_latest().profile_data
        assert saved["tags"] == ["needs-review"]
        assert saved["qc"]["passed"] is False


class TestProfileQCOnUpdate:
    @patch("src.engine.profiles._update_profile_internal")
    def test_update_records_qc_in_new_version(self, mock_update):
        mock_update.return_value = (
            {"text": "No citations here.", "tags": ["x"], "confidence": 0.5},
            [],
        )
        versioned = VersionedProfile()

        with patch("src.engine.profiles.ENABLE_PROFILE_VERSIONING", True):
            profile, versioned, _ = update_profile(
                "person",
                "Alice Smith",
                _good_profile(),
                versioned,
                "new article text",
                "art-002",
            )

        assert profile["qc"]["passed"] is False
        assert "no_citations" in profile["qc"]["flags"]
        assert versioned.get_latest().profile_data["qc"] == profile["qc"]


class TestUpdateDraftsBeforeReflecting:
    """The first reflection in an update must grade a real draft, not "{}"."""

    def _run_update(self, mode_model_type: str):
        draft = EntityProfile(**_good_profile("art-002"))
        config = MagicMock()
        config.load_profile_prompt.side_effect = lambda kind: (
            "Update {entity_type} {entity_name}"
            if kind == "update"
            else "Reflect on {entity_type} {entity_name} {article_id}"
        )
        improve_mock = MagicMock(
            return_value={
                "text": draft.model_dump_json(),
                "reflection_history": [{"iteration": 1, "valid": True}],
            }
        )
        gen_target = (
            "src.engine.profiles.cloud_generation"
            if mode_model_type == "gemini"
            else "src.engine.profiles.local_generation"
        )
        gen_mock = MagicMock(return_value=draft)

        with (
            patch("src.engine.profiles.get_domain_config", return_value=config),
            patch("src.engine.profiles.iterative_improve", improve_mock),
            patch(gen_target, gen_mock),
        ):
            updated, history = _update_profile_internal(
                "person",
                "Alice Smith",
                _good_profile(),
                "new article text",
                "art-002",
                model_type=mode_model_type,
            )
        return gen_mock, improve_mock, draft, updated

    def test_cloud_update_reflects_on_generated_draft(self):
        gen_mock, improve_mock, draft, updated = self._run_update("gemini")

        gen_mock.assert_called_once()
        initial_text = improve_mock.call_args.kwargs["initial_text"]
        assert initial_text != "{}"
        assert json.loads(initial_text)["text"] == draft.text
        assert updated["text"] == draft.text
        assert set(updated["sources"]) == {"art-001", "art-002"}

    def test_local_update_reflects_on_generated_draft(self):
        gen_mock, improve_mock, draft, _ = self._run_update("ollama")

        gen_mock.assert_called_once()
        initial_text = improve_mock.call_args.kwargs["initial_text"]
        assert json.loads(initial_text)["text"] == draft.text


class TestWriteEntitiesKeepsAllKeys:
    def test_key_missing_from_first_row_is_kept(self, tmp_path):
        """A field only later rows have (e.g. profile_grounding) must survive."""
        people = [
            {"name": "Alice"},
            {"name": "Bob", "profile_grounding": {"grounding_score": 0.5}},
        ]
        write_entities_table("people", people, str(tmp_path))

        rows = {r["name"]: r for r in read_entities_from_file("people", str(tmp_path))}
        assert rows["Bob"]["profile_grounding"] == {"grounding_score": 0.5}
        assert rows["Alice"]["profile_grounding"] is None
