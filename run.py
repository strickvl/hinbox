#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hinbox - Command-line interface for processing articles and extracting entities.

This script provides a user-friendly interface to the main functionality of the Hinbox project.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from src.config_loader import DomainConfig
from src.logging_config import log


def run_command(cmd):
    """Run a command and handle errors."""
    log(f"Running: {' '.join(cmd)}", level="processing")
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        log(f"Command failed with exit code {e.returncode}", level="error")
        return e.returncode
    except KeyboardInterrupt:
        log("Process interrupted by user", level="warning")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Hinbox - Process articles and extract entities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s process                     # Process 5 articles (default)
  %(prog)s process -n 10               # Process 10 articles
  %(prog)s process --relevance         # Process with relevance checking
  %(prog)s process --force --local     # Force reprocess using local models
  %(prog)s check                       # Check database statistics
  %(prog)s check --sample              # Show a sample article
  %(prog)s reset                       # Reset processing status
  %(prog)s frontend                    # Start the web interface
  %(prog)s fetch-miami                 # Fetch Miami Herald articles
  %(prog)s import-miami                # Import Miami Herald articles
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Process command
    process_parser = subparsers.add_parser(
        "process", help="Process articles and extract entities"
    )
    process_parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=5,
        help="Number of articles to process (default: 5)",
    )
    process_parser.add_argument(
        "--local",
        action="store_true",
        help="Use local models (Ollama/spaCy) instead of cloud APIs",
    )
    process_parser.add_argument(
        "--relevance",
        "--relevance-check",
        action="store_true",
        help="Perform relevance check before processing",
    )
    process_parser.add_argument(
        "--force",
        "--force-reprocess",
        action="store_true",
        help="Force reprocessing of already processed articles",
    )
    process_parser.add_argument(
        "--articles-path", type=str, help="Custom path to articles Parquet file"
    )
    process_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    process_parser.add_argument(
        "--domain",
        type=str,
        default="guantanamo",
        help="Domain configuration to use (default: guantanamo)",
    )

    # Check command
    check_parser = subparsers.add_parser(
        "check", help="Check article database statistics"
    )
    check_parser.add_argument(
        "--sample", action="store_true", help="Display a sample article"
    )

    # Reset command
    subparsers.add_parser("reset", help="Reset processing status of all articles")

    # Frontend command
    subparsers.add_parser(
        "frontend", help="Start the web interface", aliases=["web", "ui"]
    )

    # Miami Herald commands
    subparsers.add_parser("fetch-miami", help="Fetch Miami Herald articles")
    subparsers.add_parser(
        "import-miami", help="Import Miami Herald articles from JSONL"
    )

    # Domain management commands
    init_parser = subparsers.add_parser(
        "init", help="Initialize a new domain configuration"
    )
    init_parser.add_argument("domain", help="Name of the new domain to create")

    subparsers.add_parser("domains", help="List available domain configurations")

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    # Execute commands
    if args.command == "process":
        cmd = [sys.executable, "-m", "src.process_and_extract"]
        if args.local:
            cmd.append("--local")
        if args.limit:
            cmd.extend(["--limit", str(args.limit)])
        if args.relevance:
            cmd.append("--relevance-check")
        if args.force:
            cmd.append("--force-reprocess")
        if args.articles_path:
            cmd.extend(["--articles-path", args.articles_path])
        if args.verbose:
            cmd.append("--verbose")
        if args.domain:
            cmd.extend(["--domain", args.domain])
        return run_command(cmd)

    elif args.command == "check":
        cmd = [sys.executable, "-m", "scripts.check_articles_parquet"]
        if args.sample:
            cmd.append("--sample")
        return run_command(cmd)

    elif args.command == "reset":
        log("This will reset the processing status of all articles.", level="warning")
        response = input("Are you sure? (y/N): ")
        if response.lower() == "y":
            return run_command(
                [sys.executable, "-m", "scripts.reset_processing_status"]
            )
        else:
            log("Reset cancelled.", level="info")
            return 0

    elif args.command in ["frontend", "web", "ui"]:
        log("Starting web interface...", level="info")
        log("Open http://localhost:5001 in your browser", level="info")
        return run_command([sys.executable, "-m", "src.frontend"])

    elif args.command == "fetch-miami":
        return run_command([sys.executable, "-m", "scripts.get_miami_herald_articles"])

    elif args.command == "import-miami":
        return run_command(
            [sys.executable, "-m", "scripts.import_miami_herald_articles"]
        )

    elif args.command == "init":
        return init_domain(args.domain)

    elif args.command == "domains":
        return list_domains()

    else:
        parser.print_help()
        return 1


def init_domain(domain_name: str) -> int:
    """Initialize a new domain configuration."""
    # Validate domain name
    if not domain_name.isalnum():
        log(f"Domain name '{domain_name}' must be alphanumeric", level="error")
        return 1

    # Check if domain already exists
    target_dir = Path(f"configs/{domain_name}")
    if target_dir.exists():
        log(f"Domain '{domain_name}' already exists", level="error")
        return 1

    # Copy template
    template_dir = Path("configs/template")
    if not template_dir.exists():
        log("Template directory not found", level="error")
        return 1

    try:
        shutil.copytree(template_dir, target_dir)
        log(f"Created domain configuration: {target_dir}", level="success")

        # Process template files
        for template_file in target_dir.rglob("*.template"):
            new_file = template_file.with_suffix("")
            content = template_file.read_text()

            # Replace placeholders
            content = content.replace("{DOMAIN_NAME}", domain_name)
            content = content.replace(
                "{DOMAIN_DESCRIPTION}",
                f"Articles and analysis related to {domain_name}",
            )

            new_file.write_text(content)
            template_file.unlink()  # Remove .template file

        log(f"Domain '{domain_name}' initialized successfully!", level="success")
        log(
            f"Edit files in {target_dir} to customize your domain configuration",
            level="info",
        )
        return 0

    except Exception as e:
        log(f"Failed to initialize domain: {e}", level="error")
        return 1


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
    sys.exit(main())
