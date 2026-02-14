# Hinbox task runner
# Run `just` or `just --list` to see available commands.

set dotenv-load

# Default: show available recipes
default:
    @just --list

# ─── Processing ───────────────────────────────────────

# Process articles and extract entities
process *args:
    uv run python -m src.process_and_extract {{args}}

# Process articles for a specific domain
process-domain domain *args:
    uv run python -m src.process_and_extract --domain {{domain}} {{args}}

# Quick: process one article with verbose output
test-run:
    uv run python -m src.process_and_extract --limit 1 -v --force

# ─── Web Interface ────────────────────────────────────

# Start the web interface (http://localhost:5001)
frontend:
    @echo "Open http://localhost:5001 in your browser"
    uv run python -m src.frontend

alias web := frontend
alias ui := frontend

# ─── Data Management ──────────────────────────────────

# Check article database statistics
check *args:
    uv run python scripts/check_articles_parquet.py {{args}}

alias stats := check

# Reset processing status of all articles
reset:
    #!/usr/bin/env bash
    read -p "This will reset ALL articles. Are you sure? (y/N): " response
    if [[ "$response" == [yY] ]]; then
        uv run python scripts/reset_processing_status.py
    else
        echo "Reset cancelled."
    fi

# Fetch Miami Herald articles
fetch-miami:
    uv run python scripts/get_miami_herald_articles.py

# Import Miami Herald articles from JSONL
import-miami:
    uv run python scripts/import_miami_herald_articles.py

# ─── Domain Management ────────────────────────────────

# Initialize a new domain configuration
init domain:
    uv run python scripts/init_domain.py {{domain}}

# List available domain configurations
domains:
    uv run python scripts/list_domains.py

# ─── Code Quality ─────────────────────────────────────

# Format code (ruff fix + format)
format:
    uv run ruff check . --select F401,F841 --fix --exclude "__init__.py" --isolated
    uv run ruff check . --select I --fix --ignore D
    uv run ruff format .

# Lint code (ruff check + format verification)
lint:
    uv run ruff check .
    uv run ruff format --check .

# Run tests
test *args:
    uv run pytest tests/ -v {{args}}

# Run CI-equivalent checks (what GitHub Actions runs on PRs)
ci:
    uv run ruff check .
    uv run ruff format --check .
    uv run pytest tests/ -v -m "not asyncio" --tb=short

# Format then lint
check-code: format lint
