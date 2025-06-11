"""Domain configuration loader and management."""

import os
from functools import lru_cache
from typing import Any, Dict, List

import yaml


class DomainConfig:
    """Domain configuration manager."""

    def __init__(self, domain: str):
        self.domain = domain
        self.config_dir = f"configs/{domain}"
        self._validate_domain()

    def _validate_domain(self):
        """Validate that the domain configuration exists."""
        if not os.path.exists(self.config_dir):
            available_domains = self.get_available_domains()
            raise ValueError(
                f"Domain '{self.domain}' not found. "
                f"Available domains: {', '.join(available_domains)}"
            )

    @staticmethod
    def get_available_domains() -> List[str]:
        """Get list of available domain configurations."""
        configs_dir = "configs"
        if not os.path.exists(configs_dir):
            return []

        domains = []
        for item in os.listdir(configs_dir):
            domain_path = os.path.join(configs_dir, item)
            if (
                os.path.isdir(domain_path)
                and item not in ["template"]
                and os.path.exists(os.path.join(domain_path, "config.yaml"))
            ):
                domains.append(item)
        return sorted(domains)

    @lru_cache(maxsize=None)
    def load_config(self) -> Dict[str, Any]:
        """Load the main domain configuration."""
        config_path = os.path.join(self.config_dir, "config.yaml")
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    @lru_cache(maxsize=None)
    def load_categories(self, entity_type: str) -> Dict[str, Any]:
        """Load category definitions for an entity type."""
        categories_path = os.path.join(
            self.config_dir, "categories", f"{entity_type}.yaml"
        )
        if not os.path.exists(categories_path):
            raise ValueError(f"Categories file not found: {categories_path}")

        with open(categories_path, "r") as f:
            return yaml.safe_load(f)

    @lru_cache(maxsize=None)
    def load_prompt(self, entity_type: str) -> str:
        """Load extraction prompt for an entity type."""
        prompt_path = os.path.join(self.config_dir, "prompts", f"{entity_type}.md")
        if not os.path.exists(prompt_path):
            raise ValueError(f"Prompt file not found: {prompt_path}")

        with open(prompt_path, "r") as f:
            return f.read()

    @lru_cache(maxsize=None)
    def load_profile_prompt(self, prompt_type: str) -> str:
        """Load profile-related prompt (generation, update, reflection)."""
        prompt_path = os.path.join(
            self.config_dir, "prompts", f"profile_{prompt_type}.md"
        )
        if not os.path.exists(prompt_path):
            raise ValueError(f"Profile prompt file not found: {prompt_path}")

        with open(prompt_path, "r") as f:
            return f.read()

    def get_data_path(self) -> str:
        """Get the default data path for this domain."""
        config = self.load_config()
        return config["data_sources"]["default_path"]

    def get_output_dir(self) -> str:
        """Get the output directory for this domain."""
        config = self.load_config()
        return config["output"]["directory"]

    def get_similarity_threshold(self) -> float:
        """Get the similarity threshold for entity deduplication."""
        config = self.load_config()
        return config.get("similarity_threshold", 0.75)

    def get_entity_types(self, entity_category: str) -> List[str]:
        """Get available entity types for a category."""
        categories = self.load_categories(entity_category)

        # Handle different possible structures
        # Map plural entity categories to their singular forms
        singular_map = {
            "people": "person",
            "events": "event",
            "organizations": "organization",
            "locations": "location",
        }

        # First try singular form (event_types, person_types, etc.)
        singular = singular_map.get(entity_category, entity_category.rstrip("s"))
        entity_type_key = f"{singular}_types"
        if entity_type_key in categories:
            return list(categories[entity_type_key].keys())
        # Then try with original name (events_types, etc.)
        elif f"{entity_category}_types" in categories:
            return list(categories[f"{entity_category}_types"].keys())
        elif "types" in categories:
            return list(categories["types"].keys())
        else:
            # Fallback: assume top-level keys are the types
            return list(categories.keys())

    def get_entity_type_info(
        self, entity_category: str, entity_type: str
    ) -> Dict[str, Any]:
        """Get information about a specific entity type."""
        categories = self.load_categories(entity_category)

        # Handle different possible structures
        types_dict = None
        if f"{entity_category}_types" in categories:
            types_dict = categories[f"{entity_category}_types"]
        elif "types" in categories:
            types_dict = categories["types"]
        else:
            types_dict = categories

        if entity_type not in types_dict:
            raise ValueError(
                f"Entity type '{entity_type}' not found in {entity_category}"
            )

        return types_dict[entity_type]

    def validate_embeddings_config(self) -> bool:
        """Validate embeddings configuration.

        Returns:
            True if valid, raises ValueError if invalid
        """
        config = self.load_config()
        embeddings_config = config.get("embeddings", {})

        # Check mode
        mode = embeddings_config.get("mode", "local")
        valid_modes = ["local", "cloud", "hybrid"]
        if mode not in valid_modes:
            raise ValueError(
                f"Invalid embeddings mode '{mode}'. Must be one of: {', '.join(valid_modes)}"
            )

        # Check cloud config if needed
        if mode in ["cloud", "hybrid"]:
            cloud_config = embeddings_config.get("cloud", {})
            if not cloud_config.get("model"):
                raise ValueError(
                    "Cloud embeddings model must be specified when using cloud or hybrid mode"
                )

        # Check local config if needed
        if mode in ["local", "hybrid"]:
            local_config = embeddings_config.get("local", {})
            if not local_config.get("model"):
                raise ValueError(
                    "Local embeddings model must be specified when using local or hybrid mode"
                )

        return True

    def get_embeddings_config(self) -> Dict[str, Any]:
        """Get embeddings configuration with defaults."""
        config = self.load_config()
        embeddings_config = config.get("embeddings", {})

        # Set defaults
        defaults = {
            "mode": "local",
            "cloud": {
                "model": "jina_ai/jina-embeddings-v3",
                "batch_size": 100,
                "max_retries": 3,
                "timeout": 30,
            },
            "local": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "batch_size": 32,
            },
        }

        # Merge with defaults
        result = defaults.copy()
        if embeddings_config:
            result["mode"] = embeddings_config.get("mode", defaults["mode"])
            if "cloud" in embeddings_config:
                result["cloud"].update(embeddings_config["cloud"])
            if "local" in embeddings_config:
                result["local"].update(embeddings_config["local"])

        return result


# Global instances for common domains
@lru_cache(maxsize=None)
def get_domain_config(domain: str = "guantanamo") -> DomainConfig:
    """Get a cached domain configuration instance."""
    return DomainConfig(domain)


# Convenience functions for backward compatibility
def load_domain_config(domain: str = "guantanamo") -> Dict[str, Any]:
    """Load domain configuration (backward compatibility)."""
    return get_domain_config(domain).load_config()


def get_prompt(domain: str, entity_type: str) -> str:
    """Get extraction prompt for an entity type (backward compatibility)."""
    return get_domain_config(domain).load_prompt(entity_type)


def get_system_prompt(entity_type: str, domain: str = "guantanamo") -> str:
    """Get system prompt for an entity type (backward compatibility with existing code)."""
    return get_domain_config(domain).load_prompt(entity_type)
