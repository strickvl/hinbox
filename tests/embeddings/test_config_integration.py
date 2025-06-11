"""Tests for embeddings configuration integration."""

import os
import tempfile
from pathlib import Path

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


# Add missing import at the top
from unittest.mock import patch
