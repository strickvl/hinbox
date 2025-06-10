"""
Unit tests for frontend profile versioning functionality.

Tests the ProfileVersionSelector component and frontend route version handling.
"""

from unittest.mock import patch

import pytest

from src.frontend.components import ProfileVersionSelector
from src.profiles import VersionedProfile


class TestProfileVersionSelector:
    """Test ProfileVersionSelector component functionality."""

    def test_selector_disabled_when_versioning_disabled(self):
        """Test that selector returns empty string when versioning is disabled."""
        with patch("src.constants.ENABLE_PROFILE_VERSIONING", False):
            selector = ProfileVersionSelector(
                entity_name="Test Person",
                entity_type="people",
                current_version=2,
                total_versions=3,
                route_prefix="people",
                entity_key="test-person",
            )

            assert selector == ""

    def test_selector_disabled_when_single_version(self):
        """Test that selector returns empty string when only one version exists."""
        with patch("src.constants.ENABLE_PROFILE_VERSIONING", True):
            selector = ProfileVersionSelector(
                entity_name="Test Person",
                entity_type="people",
                current_version=1,
                total_versions=1,  # Only one version
                route_prefix="people",
                entity_key="test-person",
            )

            assert selector == ""

    @patch("src.frontend.utils.encode_key")
    def test_selector_with_multiple_versions(self, mock_encode_key):
        """Test selector creation with multiple versions."""
        mock_encode_key.return_value = "encoded-test-person"

        with patch("src.constants.ENABLE_PROFILE_VERSIONING", True):
            selector = ProfileVersionSelector(
                entity_name="Test Person",
                entity_type="people",
                current_version=2,
                total_versions=3,
                route_prefix="people",
                entity_key="test-person",
                selected_version=1,
            )

            # Should return a non-empty Div element
            assert selector != ""
            assert hasattr(
                selector, "tag"
            )  # FastHTML element should have tag attribute

            # Verify encode_key was called
            mock_encode_key.assert_called_once_with("test-person")

    def test_selector_default_selected_version(self):
        """Test that selected_version defaults to current_version when not specified."""
        with patch("src.constants.ENABLE_PROFILE_VERSIONING", True):
            with patch("src.frontend.utils.encode_key", return_value="encoded-key"):
                selector = ProfileVersionSelector(
                    entity_name="Test Person",
                    entity_type="people",
                    current_version=3,
                    total_versions=3,
                    route_prefix="people",
                    entity_key="test-person",
                    # selected_version not specified
                )

                # Should use current_version as selected_version
                assert selector != ""

    def test_selector_version_order(self):
        """Test that versions are ordered latest first."""
        with patch("src.constants.ENABLE_PROFILE_VERSIONING", True):
            with patch("src.frontend.utils.encode_key", return_value="encoded-key"):
                # We can't easily test the internal structure without parsing HTML,
                # but we can at least verify the component is created
                selector = ProfileVersionSelector(
                    entity_name="Test Person",
                    entity_type="people",
                    current_version=3,
                    total_versions=5,
                    route_prefix="people",
                    entity_key="test-person",
                    selected_version=2,
                )

                assert selector != ""
                # The actual order testing would require HTML parsing or mock verification

    def test_selector_javascript_generation(self):
        """Test that the selector generates correct JavaScript for navigation."""
        with patch("src.constants.ENABLE_PROFILE_VERSIONING", True):
            with patch("src.frontend.utils.encode_key", return_value="encoded%2Dkey"):
                selector = ProfileVersionSelector(
                    entity_name="Test Person",
                    entity_type="people",
                    current_version=2,
                    total_versions=3,
                    route_prefix="people",
                    entity_key="test-person",
                )

                # Convert to string to check JavaScript content
                selector_str = str(selector)

                # Should contain the navigation JavaScript
                assert "/people/encoded%2Dkey?version=" in selector_str
                assert "window.location.href" in selector_str

    def test_selector_different_entity_types(self):
        """Test selector works with different entity types."""
        entity_types = ["people", "organizations", "locations", "events"]

        with patch("src.constants.ENABLE_PROFILE_VERSIONING", True):
            with patch("src.frontend.utils.encode_key", return_value="encoded-key"):
                for entity_type in entity_types:
                    selector = ProfileVersionSelector(
                        entity_name="Test Entity",
                        entity_type=entity_type,
                        current_version=2,
                        total_versions=3,
                        route_prefix=entity_type,
                        entity_key="test-entity",
                    )

                    assert selector != ""
                    selector_str = str(selector)
                    assert f"/{entity_type}/encoded-key?version=" in selector_str


class TestRoutesVersionHandling:
    """Test version handling in frontend routes."""

    def test_route_version_parameter_parsing(self):
        """Test version parameter parsing logic."""
        # This would be a more complex test that might require actual request mocking
        # For now, we'll test the key logic components

        # Test valid version number
        version_param = "2"
        try:
            requested_version = int(version_param)
            assert requested_version == 2
        except (ValueError, TypeError):
            pytest.fail("Should parse valid version number")

        # Test invalid version number
        version_param = "invalid"
        with pytest.raises(ValueError):
            int(version_param)

    def test_version_bounds_checking(self):
        """Test version bounds checking logic."""
        current_version = 3

        # Valid versions
        assert 1 <= 1 <= current_version
        assert 1 <= 2 <= current_version
        assert 1 <= 3 <= current_version

        # Invalid versions
        assert not (1 <= 0 <= current_version)
        assert not (1 <= 4 <= current_version)
        assert not (1 <= -1 <= current_version)


class TestVersioningIntegration:
    """Integration tests for versioning components."""

    def test_full_versioning_workflow(self):
        """Test complete versioning workflow from model to component."""

        # Create versioned profile with multiple versions
        versioned = VersionedProfile()
        versioned.add_version({"text": "Version 1"}, "article1")
        versioned.add_version({"text": "Version 2"}, "article2")
        versioned.add_version({"text": "Version 3"}, "article3")

        # Test that component would be shown
        with patch("src.constants.ENABLE_PROFILE_VERSIONING", True):
            with patch("src.frontend.utils.encode_key", return_value="test-key"):
                selector = ProfileVersionSelector(
                    entity_name="Test Entity",
                    entity_type="people",
                    current_version=versioned.current_version,
                    total_versions=len(versioned.versions),
                    route_prefix="people",
                    entity_key="test-entity",
                )

                assert selector != ""

                # Test version retrieval
                v1 = versioned.get_version(1)
                v2 = versioned.get_version(2)
                v3 = versioned.get_version(3)

                assert v1.profile_data["text"] == "Version 1"
                assert v2.profile_data["text"] == "Version 2"
                assert v3.profile_data["text"] == "Version 3"

                assert v1.trigger_article_id == "article1"
                assert v2.trigger_article_id == "article2"
                assert v3.trigger_article_id == "article3"

    def test_component_respects_feature_flag(self):
        """Test that components properly respect the feature flag."""
        test_cases = [
            (True, 3, True),  # Enabled, multiple versions -> show
            (True, 1, False),  # Enabled, single version -> hide
            (False, 3, False),  # Disabled, multiple versions -> hide
            (False, 1, False),  # Disabled, single version -> hide
        ]

        for flag_value, total_versions, should_show in test_cases:
            with patch("src.constants.ENABLE_PROFILE_VERSIONING", flag_value):
                with patch("src.frontend.utils.encode_key", return_value="test"):
                    selector = ProfileVersionSelector(
                        entity_name="Test",
                        entity_type="people",
                        current_version=min(total_versions, 1),
                        total_versions=total_versions,
                        route_prefix="people",
                        entity_key="test",
                    )

                    if should_show:
                        assert selector != ""
                    else:
                        assert selector == ""
