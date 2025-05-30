"""
Unit tests for profile versioning functionality.

Tests the ProfileVersion and VersionedProfile models, as well as
profile creation and update functions with versioning support.
"""

from datetime import datetime, timezone
from unittest.mock import patch

from src.constants import ENABLE_PROFILE_VERSIONING
from src.profiles import (
    ProfileVersion,
    VersionedProfile,
    create_profile,
    update_profile,
)


class TestProfileVersion:
    """Test ProfileVersion model functionality."""

    def test_profile_version_creation(self):
        """Test creating a ProfileVersion with all fields."""
        profile_data = {
            "text": "This is a test profile",
            "confidence": 0.8,
            "tags": ["test", "profile"],
            "sources": ["article1"],
        }

        version = ProfileVersion(
            version_number=1, profile_data=profile_data, trigger_article_id="article1"
        )

        assert version.version_number == 1
        assert version.profile_data == profile_data
        assert version.trigger_article_id == "article1"
        assert isinstance(version.created_at, datetime)

    def test_profile_version_default_timestamp(self):
        """Test that ProfileVersion gets default timestamp."""
        profile_data = {"text": "Test profile"}

        version = ProfileVersion(version_number=1, profile_data=profile_data)

        assert version.trigger_article_id is None
        assert isinstance(version.created_at, datetime)
        # Should be created within the last few seconds
        now = datetime.now(timezone.utc)
        time_diff = abs(
            (now - version.created_at.replace(tzinfo=timezone.utc)).total_seconds()
        )
        assert time_diff < 5  # Created within 5 seconds

    def test_profile_version_data_isolation(self):
        """Test that profile_data is properly isolated (copied)."""
        original_data = {"text": "Original", "tags": ["tag1"]}

        version = ProfileVersion(version_number=1, profile_data=original_data)

        # Modify original data
        original_data["text"] = "Modified"
        original_data["tags"].append("tag2")

        # Version should have the original data
        assert version.profile_data["text"] == "Original"
        assert version.profile_data["tags"] == ["tag1"]


class TestVersionedProfile:
    """Test VersionedProfile model functionality."""

    def test_versioned_profile_creation(self):
        """Test creating an empty VersionedProfile."""
        versioned = VersionedProfile()

        assert versioned.current_version == 1
        assert len(versioned.versions) == 0

    def test_add_version(self):
        """Test adding versions to profile."""
        versioned = VersionedProfile()
        profile_data1 = {"text": "Version 1", "confidence": 0.8}
        profile_data2 = {"text": "Version 2", "confidence": 0.9}

        # Add first version
        v1 = versioned.add_version(profile_data1, trigger_article_id="article1")
        assert v1.version_number == 1
        assert v1.profile_data == profile_data1
        assert v1.trigger_article_id == "article1"
        assert len(versioned.versions) == 1
        assert versioned.current_version == 1

        # Add second version
        v2 = versioned.add_version(profile_data2, trigger_article_id="article2")
        assert v2.version_number == 2
        assert v2.profile_data == profile_data2
        assert v2.trigger_article_id == "article2"
        assert len(versioned.versions) == 2
        assert versioned.current_version == 2

    def test_get_version(self):
        """Test retrieving specific versions."""
        versioned = VersionedProfile()
        versioned.add_version({"text": "Version 1"})
        versioned.add_version({"text": "Version 2"})
        versioned.add_version({"text": "Version 3"})

        # Get existing versions
        v1 = versioned.get_version(1)
        assert v1.profile_data["text"] == "Version 1"

        v2 = versioned.get_version(2)
        assert v2.profile_data["text"] == "Version 2"

        v3 = versioned.get_version(3)
        assert v3.profile_data["text"] == "Version 3"

        # Test non-existent version
        assert versioned.get_version(4) is None
        assert versioned.get_version(0) is None

    def test_get_latest(self):
        """Test getting latest version."""
        versioned = VersionedProfile()

        # No versions yet
        assert versioned.get_latest() is None

        # Add versions
        versioned.add_version({"text": "Version 1"})
        latest = versioned.get_latest()
        assert latest.profile_data["text"] == "Version 1"

        versioned.add_version({"text": "Version 2"})
        latest = versioned.get_latest()
        assert latest.profile_data["text"] == "Version 2"

    def test_version_data_independence(self):
        """Test that version data is independent between versions."""
        versioned = VersionedProfile()

        data1 = {"text": "Version 1", "tags": ["tag1"]}
        data2 = {"text": "Version 2", "tags": ["tag2"]}

        versioned.add_version(data1)
        versioned.add_version(data2)

        # Modify original data
        data1["text"] = "Modified"
        data2["tags"].append("modified")

        # Stored versions should be unchanged
        v1 = versioned.get_version(1)
        v2 = versioned.get_version(2)

        assert v1.profile_data["text"] == "Version 1"
        assert v1.profile_data["tags"] == ["tag1"]
        assert v2.profile_data["text"] == "Version 2"
        assert v2.profile_data["tags"] == ["tag2"]


class TestProfileFunctionsWithVersioning:
    """Test profile creation and update functions with versioning."""

    @patch("src.profiles.generate_profile")
    def test_create_profile_with_versioning_enabled(self, mock_generate):
        """Test create_profile when versioning is enabled."""
        # Mock the profile generation
        mock_profile = {
            "text": "Generated profile",
            "confidence": 0.8,
            "tags": ["test"],
            "sources": ["article1"],
        }
        mock_history = [{"iteration": 1, "valid": True}]
        mock_generate.return_value = (mock_profile, mock_history)

        with patch("src.profiles.ENABLE_PROFILE_VERSIONING", True):
            profile_dict, versioned_profile, history = create_profile(
                entity_type="person",
                entity_name="Test Person",
                article_text="Test article content",
                article_id="article1",
                model_type="gemini",
            )

            assert profile_dict == mock_profile
            assert isinstance(versioned_profile, VersionedProfile)
            assert len(versioned_profile.versions) == 1
            assert versioned_profile.current_version == 1
            assert versioned_profile.get_latest().profile_data == mock_profile
            assert versioned_profile.get_latest().trigger_article_id == "article1"
            assert history == mock_history

    @patch("src.profiles.generate_profile")
    def test_create_profile_with_versioning_disabled(self, mock_generate):
        """Test create_profile when versioning is disabled."""
        mock_profile = {
            "text": "Generated profile",
            "confidence": 0.8,
            "tags": ["test"],
            "sources": ["article1"],
        }
        mock_history = [{"iteration": 1, "valid": True}]
        mock_generate.return_value = (mock_profile, mock_history)

        with patch("src.profiles.ENABLE_PROFILE_VERSIONING", False):
            profile_dict, versioned_profile, history = create_profile(
                entity_type="person",
                entity_name="Test Person",
                article_text="Test article content",
                article_id="article1",
                model_type="gemini",
            )

            assert profile_dict == mock_profile
            assert isinstance(versioned_profile, VersionedProfile)
            assert (
                len(versioned_profile.versions) == 0
            )  # No version added when disabled
            assert versioned_profile.current_version == 1
            assert history == mock_history

    @patch("src.profiles._update_profile_internal")
    def test_update_profile_with_versioning_enabled(self, mock_update):
        """Test update_profile when versioning is enabled."""
        # Setup existing versioned profile
        existing_versioned = VersionedProfile()
        existing_versioned.add_version(
            {
                "text": "Original profile",
                "confidence": 0.7,
                "tags": ["old"],
                "sources": ["article1"],
            }
        )

        # Mock the internal update
        updated_profile = {
            "text": "Updated profile",
            "confidence": 0.9,
            "tags": ["new"],
            "sources": ["article1", "article2"],
        }
        mock_history = [{"iteration": 1, "valid": True}]
        mock_update.return_value = (updated_profile, mock_history)

        existing_profile = {
            "text": "Original profile",
            "confidence": 0.7,
            "tags": ["old"],
            "sources": ["article1"],
        }

        with patch("src.profiles.ENABLE_PROFILE_VERSIONING", True):
            result_profile, result_versioned, history = update_profile(
                entity_type="person",
                entity_name="Test Person",
                existing_profile=existing_profile,
                versioned_profile=existing_versioned,
                new_article_text="New article content",
                new_article_id="article2",
                model_type="gemini",
            )

            assert result_profile == updated_profile
            assert isinstance(result_versioned, VersionedProfile)
            assert len(result_versioned.versions) == 2  # Original + new version
            assert result_versioned.current_version == 2
            assert result_versioned.get_latest().profile_data == updated_profile
            assert result_versioned.get_latest().trigger_article_id == "article2"
            assert history == mock_history

    @patch("src.profiles._update_profile_internal")
    def test_update_profile_with_versioning_disabled(self, mock_update):
        """Test update_profile when versioning is disabled."""
        # Setup existing versioned profile (but versioning disabled)
        existing_versioned = VersionedProfile()

        updated_profile = {
            "text": "Updated profile",
            "confidence": 0.9,
            "tags": ["new"],
            "sources": ["article1", "article2"],
        }
        mock_history = [{"iteration": 1, "valid": True}]
        mock_update.return_value = (updated_profile, mock_history)

        existing_profile = {
            "text": "Original profile",
            "confidence": 0.7,
            "tags": ["old"],
            "sources": ["article1"],
        }

        with patch("src.profiles.ENABLE_PROFILE_VERSIONING", False):
            result_profile, result_versioned, history = update_profile(
                entity_type="person",
                entity_name="Test Person",
                existing_profile=existing_profile,
                versioned_profile=existing_versioned,
                new_article_text="New article content",
                new_article_id="article2",
                model_type="gemini",
            )

            assert result_profile == updated_profile
            assert isinstance(result_versioned, VersionedProfile)
            assert len(result_versioned.versions) == 0  # No version added when disabled
            assert result_versioned.current_version == 1
            assert history == mock_history


class TestFeatureFlag:
    """Test feature flag behavior."""

    def test_feature_flag_default_enabled(self):
        """Test that feature flag is enabled by default."""
        # Test the actual constant (should be True based on our recent change)
        assert ENABLE_PROFILE_VERSIONING is True

    @patch.dict("os.environ", {"ENABLE_PROFILE_VERSIONING": "false"})
    def test_feature_flag_can_be_disabled(self):
        """Test that feature flag can be disabled via environment variable."""
        # Need to reload the module to pick up env var change
        import importlib

        from src import constants

        importlib.reload(constants)

        assert constants.ENABLE_PROFILE_VERSIONING is False

    @patch.dict("os.environ", {"ENABLE_PROFILE_VERSIONING": "TRUE"})
    def test_feature_flag_case_insensitive(self):
        """Test that feature flag is case insensitive."""
        import importlib

        from src import constants

        importlib.reload(constants)

        assert constants.ENABLE_PROFILE_VERSIONING is True

    @patch.dict("os.environ", {"ENABLE_PROFILE_VERSIONING": "invalid"})
    def test_feature_flag_invalid_value_defaults_false(self):
        """Test that invalid values default to False."""
        import importlib

        from src import constants

        importlib.reload(constants)

        assert constants.ENABLE_PROFILE_VERSIONING is False
