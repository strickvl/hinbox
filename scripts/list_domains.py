#!/usr/bin/env python3
"""List available domain configurations."""

import sys

from src.config_loader import DomainConfig
from src.logging_config import log


def list_domains() -> int:
    """List available domain configurations."""
    try:
        domains = DomainConfig.get_available_domains()
        if not domains:
            log("No domain configurations found", level="warning")
            return 0

        log("Available domains:", level="info")
        for domain in domains:
            try:
                config = DomainConfig(domain)
                domain_config = config.load_config()
                description = domain_config.get("description", "No description")
                log(f"  • {domain}: {description}", level="info")
            except Exception as e:
                log(f"  • {domain}: (error loading config: {e})", level="warning")

        return 0

    except Exception as e:
        log(f"Failed to list domains: {e}", level="error")
        return 1


if __name__ == "__main__":
    sys.exit(list_domains())
