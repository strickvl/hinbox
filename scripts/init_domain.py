#!/usr/bin/env python3
"""Initialize a new domain configuration by copying the template directory."""

import shutil
import sys
from pathlib import Path

from src.logging_config import log


def init_domain(domain_name: str) -> int:
    """Initialize a new domain configuration."""
    if not domain_name.isalnum():
        log(f"Domain name '{domain_name}' must be alphanumeric", level="error")
        return 1

    target_dir = Path(f"configs/{domain_name}")
    if target_dir.exists():
        log(f"Domain '{domain_name}' already exists", level="error")
        return 1

    template_dir = Path("configs/template")
    if not template_dir.exists():
        log("Template directory not found", level="error")
        return 1

    try:
        shutil.copytree(template_dir, target_dir)
        log(f"Created domain configuration: {target_dir}", level="success")

        for template_file in target_dir.rglob("*.template"):
            new_file = template_file.with_suffix("")
            content = template_file.read_text()
            content = content.replace("{DOMAIN_NAME}", domain_name)
            content = content.replace(
                "{DOMAIN_DESCRIPTION}",
                f"Articles and analysis related to {domain_name}",
            )
            new_file.write_text(content)
            template_file.unlink()

        log(f"Domain '{domain_name}' initialized successfully!", level="success")
        log(
            f"Edit files in {target_dir} to customize your domain configuration",
            level="info",
        )
        return 0

    except Exception as e:
        log(f"Failed to initialize domain: {e}", level="error")
        return 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/init_domain.py <domain_name>")
        sys.exit(1)
    sys.exit(init_domain(sys.argv[1]))
