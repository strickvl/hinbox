"""Tests for embeddings configuration integration."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from src.config_loader import DomainConfig
from src.utils.embeddings.manager import EmbeddingManager, EmbeddingMode


class TestEmbeddingsConfigIntegration:
    """Test embeddings configuration integration with domain configs."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary config directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create config structure
            config_dir = Path(tmpdir) / "configs" / "test_domain"
            config_dir.mkdir(parents=True)

            # Create categories and prompts directories
            (config_dir / "categories").mkdir()
            (config_dir / "prompts").mkdir()

            yield config_dir

    def create_config_file(self, config_dir, embeddings_config):
        """Create a config.yaml file with given embeddings config."""
        config = {
            "domain": "test_domain",
            "description": "Test domain",
            "data_sources": {"default_path": "data/test/articles.parquet"},
            "output": {"directory": "data/test/entities"},
            "similarity_threshold": 0.75,
            "embeddings": embeddings_config,
        }

        config_path = config_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        return config_path

    def test_config_validation_valid(self, temp_config_dir):
        """Test valid embeddings configuration."""
        embeddings_config = {
            "mode": "cloud",
            "cloud": {
                "model": "jina_ai/jina-embeddings-v3",
                "batch_size": 100,
            },
            "local": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "batch_size": 32,
            },
        }

        self.create_config_file(temp_config_dir, embeddings_config)

        # Mock the config directory lookup
        with patch("os.path.exists", return_value=True):
            with patch(
                "src.config_loader.DomainConfig.config_dir", str(temp_config_dir)
            ):
                config = DomainConfig("test_domain")
                assert config.validate_embeddings_config() is True

    def test_config_validation_auto_mode(self, temp_config_dir):
        """Test that 'auto' is accepted as a valid embeddings mode."""
        embeddings_config = {
            "mode": "auto",
            "cloud": {"model": "jina_ai/jina-embeddings-v3"},
            "local": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
        }

        self.create_config_file(temp_config_dir, embeddings_config)

        with patch("os.path.exists", return_value=True):
            with patch(
                "src.config_loader.DomainConfig.config_dir", str(temp_config_dir)
            ):
                config = DomainConfig("test_domain")
                assert config.validate_embeddings_config() is True

    def test_config_validation_device(self, temp_config_dir):
        """Test device validation in local embeddings config."""
        embeddings_config = {
            "mode": "local",
            "local": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "device": "cuda",
            },
        }

        self.create_config_file(temp_config_dir, embeddings_config)

        with patch("os.path.exists", return_value=True):
            with patch(
                "src.config_loader.DomainConfig.config_dir", str(temp_config_dir)
            ):
                config = DomainConfig("test_domain")
                assert config.validate_embeddings_config() is True

    def test_config_validation_invalid_device(self, temp_config_dir):
        """Test invalid device in local embeddings config."""
        embeddings_config = {
            "mode": "local",
            "local": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "device": "tpu",  # not a valid device
            },
        }

        self.create_config_file(temp_config_dir, embeddings_config)

        with patch("os.path.exists", return_value=True):
            with patch(
                "src.config_loader.DomainConfig.config_dir", str(temp_config_dir)
            ):
                config = DomainConfig("test_domain")
                with pytest.raises(ValueError, match="Invalid local embedding device"):
                    config.validate_embeddings_config()

    def test_config_validation_invalid_mode(self, temp_config_dir):
        """Test invalid mode in embeddings configuration."""
        embeddings_config = {
            "mode": "invalid_mode",
            "cloud": {"model": "jina_ai/jina-embeddings-v3"},
            "local": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
        }

        self.create_config_file(temp_config_dir, embeddings_config)

        with patch("os.path.exists", return_value=True):
            with patch(
                "src.config_loader.DomainConfig.config_dir", str(temp_config_dir)
            ):
                config = DomainConfig("test_domain")

                with pytest.raises(ValueError, match="Invalid embeddings mode"):
                    config.validate_embeddings_config()

    def test_config_validation_missing_cloud_model(self, temp_config_dir):
        """Test missing cloud model when using cloud mode."""
        embeddings_config = {
            "mode": "cloud",
            "cloud": {"batch_size": 100},  # Missing model
            "local": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
        }

        self.create_config_file(temp_config_dir, embeddings_config)

        with patch("os.path.exists", return_value=True):
            with patch(
                "src.config_loader.DomainConfig.config_dir", str(temp_config_dir)
            ):
                config = DomainConfig("test_domain")

                with pytest.raises(
                    ValueError, match="Cloud embeddings model must be specified"
                ):
                    config.validate_embeddings_config()

    def test_get_embeddings_config_with_defaults(self, temp_config_dir):
        """Test getting embeddings config with defaults applied."""
        # Create minimal config
        embeddings_config = {
            "mode": "local",
            "local": {
                "model": "custom-local-model",
            },
        }

        self.create_config_file(temp_config_dir, embeddings_config)

        with patch("os.path.exists", return_value=True):
            with patch(
                "src.config_loader.DomainConfig.config_dir", str(temp_config_dir)
            ):
                config = DomainConfig("test_domain")
                embeddings = config.get_embeddings_config()

                # Check mode
                assert embeddings["mode"] == "local"

                # Check local config
                assert embeddings["local"]["model"] == "custom-local-model"
                assert embeddings["local"]["batch_size"] == 32  # Default

                # Check cloud config has defaults
                assert embeddings["cloud"]["model"] == "jina_ai/jina-embeddings-v3"
                assert embeddings["cloud"]["batch_size"] == 100
                assert embeddings["cloud"]["max_retries"] == 3
                assert embeddings["cloud"]["timeout"] == 30

    def create_config_with_dedup(self, config_dir, dedup_config, legacy_threshold=None):
        """Create a config.yaml with dedup section."""
        config = {
            "domain": "test_domain",
            "description": "Test domain",
            "data_sources": {"default_path": "data/test/articles.parquet"},
            "output": {"directory": "data/test/entities"},
            "embeddings": {"mode": "cloud", "cloud": {"model": "test-model"}},
        }
        if legacy_threshold is not None:
            config["similarity_threshold"] = legacy_threshold
        if dedup_config is not None:
            config["dedup"] = dedup_config

        config_path = config_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        return config_path

    def test_threshold_per_type_override(self, temp_config_dir):
        """Per-type threshold should take priority over default."""
        dedup = {
            "similarity_thresholds": {
                "default": 0.75,
                "people": 0.82,
                "events": 0.76,
            }
        }
        self.create_config_with_dedup(temp_config_dir, dedup)

        with patch("os.path.exists", return_value=True):
            with patch(
                "src.config_loader.DomainConfig.config_dir", str(temp_config_dir)
            ):
                cfg = DomainConfig("test_domain")
                assert cfg.get_similarity_threshold("people") == 0.82
                assert cfg.get_similarity_threshold("events") == 0.76
                # Unspecified type falls back to default
                assert cfg.get_similarity_threshold("organizations") == 0.75

    def test_threshold_default_fallback(self, temp_config_dir):
        """When no per-type threshold exists, fall back to dedup.default."""
        dedup = {"similarity_thresholds": {"default": 0.80}}
        self.create_config_with_dedup(temp_config_dir, dedup)

        with patch("os.path.exists", return_value=True):
            with patch(
                "src.config_loader.DomainConfig.config_dir", str(temp_config_dir)
            ):
                cfg = DomainConfig("test_domain")
                assert cfg.get_similarity_threshold("people") == 0.80
                assert cfg.get_similarity_threshold() == 0.80

    def test_threshold_legacy_fallback(self, temp_config_dir):
        """When no dedup section exists, fall back to legacy similarity_threshold."""
        self.create_config_with_dedup(
            temp_config_dir, dedup_config=None, legacy_threshold=0.65
        )

        with patch("os.path.exists", return_value=True):
            with patch(
                "src.config_loader.DomainConfig.config_dir", str(temp_config_dir)
            ):
                cfg = DomainConfig("test_domain")
                assert cfg.get_similarity_threshold("people") == 0.65
                assert cfg.get_similarity_threshold() == 0.65

    def test_threshold_hardcoded_fallback(self, temp_config_dir):
        """When neither dedup nor legacy threshold exist, fall back to 0.75."""
        self.create_config_with_dedup(
            temp_config_dir, dedup_config=None, legacy_threshold=None
        )

        with patch("os.path.exists", return_value=True):
            with patch(
                "src.config_loader.DomainConfig.config_dir", str(temp_config_dir)
            ):
                cfg = DomainConfig("test_domain")
                assert cfg.get_similarity_threshold() == 0.75

    def test_lexical_blocking_config_from_dedup(self, temp_config_dir):
        """Lexical blocking config should load from dedup section."""
        dedup = {
            "similarity_thresholds": {"default": 0.75},
            "lexical_blocking": {
                "enabled": True,
                "threshold": 70,
                "max_candidates": 30,
            },
        }
        self.create_config_with_dedup(temp_config_dir, dedup)

        with patch("os.path.exists", return_value=True):
            with patch(
                "src.config_loader.DomainConfig.config_dir", str(temp_config_dir)
            ):
                cfg = DomainConfig("test_domain")
                lb = cfg.get_lexical_blocking_config()
                assert lb["enabled"] is True
                assert lb["threshold"] == 70
                assert lb["max_candidates"] == 30

    def test_lexical_blocking_defaults_when_missing(self, temp_config_dir):
        """Missing lexical_blocking section should return safe defaults."""
        self.create_config_with_dedup(temp_config_dir, dedup_config=None)

        with patch("os.path.exists", return_value=True):
            with patch(
                "src.config_loader.DomainConfig.config_dir", str(temp_config_dir)
            ):
                cfg = DomainConfig("test_domain")
                lb = cfg.get_lexical_blocking_config()
                assert lb["enabled"] is False
                assert lb["threshold"] == 60
                assert lb["max_candidates"] == 50

    def test_manager_loads_from_config(self, temp_config_dir, monkeypatch):
        """Test that EmbeddingManager loads configuration from domain config."""
        embeddings_config = {
            "mode": "hybrid",
            "cloud": {
                "model": "custom-cloud-model",
                "batch_size": 50,
            },
            "local": {
                "model": "custom-local-model",
                "batch_size": 16,
            },
        }

        self.create_config_file(temp_config_dir, embeddings_config)

        # Set up the configs directory path
        configs_root = temp_config_dir.parent.parent
        monkeypatch.chdir(configs_root)

        # Create manager
        manager = EmbeddingManager(domain="test_domain")

        # Check mode
        assert manager.mode == EmbeddingMode.HYBRID

        # Check providers were initialized
        assert manager.cloud_provider is not None
        assert manager.local_provider is not None

        # Check configurations
        assert manager.cloud_provider.config.model_name == "custom-cloud-model"
        assert manager.cloud_provider.config.batch_size == 50
        assert manager.local_provider.config.model_name == "custom-local-model"
        assert manager.local_provider.config.batch_size == 16

    def test_environment_override(self, temp_config_dir, monkeypatch):
        """Test that environment variable overrides config file."""
        embeddings_config = {
            "mode": "local",  # Config says local
            "cloud": {"model": "jina_ai/jina-embeddings-v3"},
            "local": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
        }

        self.create_config_file(temp_config_dir, embeddings_config)

        configs_root = temp_config_dir.parent.parent
        monkeypatch.chdir(configs_root)

        # Set environment variable to override
        with patch.dict(os.environ, {"EMBEDDING_MODE": "cloud"}):
            manager = EmbeddingManager(domain="test_domain")

            # Should use cloud mode from environment
            assert manager.mode == EmbeddingMode.CLOUD
            assert manager.cloud_provider is not None
            assert manager.local_provider is None
