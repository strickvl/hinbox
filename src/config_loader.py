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
