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

# ─── Evaluation ───────────────────────────────────────

# Score a model on hand-labelled gold sets (e.g. `just eval relevance --model ollama/qwen3.6:35b-a3b`)
eval *args:
    uv run python -m src.evals {{args}}

# Grow long profiles by replaying well-covered entities' articles in date order
eval-replay *args:
    uv run python -m src.evals.profile_replay {{args}}

# Build profile-update test cases from replay snapshots (e.g. `just eval-cases --model openai/gpt-6-luna`)
eval-cases *args:
    uv run python -m src.evals.profile_cases {{args}}

# Mine hard merge pairs from existing entities for labelling
eval-mine *args:
    uv run python -m src.evals.mine_merge_pairs {{args}}

# ─── DSPy experiments (own environment in experiments/dspy_opt/, no LiteLLM) ──

# Optimise the merge check with DSPy (e.g. `just dspy-merge --model ollama/qwen3.6:35b-a3b-q4_K_M --optimizer mipro`)
dspy-merge *args:
    uv run --project experiments/dspy_opt python -m experiments.dspy_opt.merge {{args}}

# Offline tests for the DSPy experiments
dspy-test *args:
    uv run --project experiments/dspy_opt --with pytest pytest experiments/dspy_opt/tests {{args}}

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

# Fail if any tracked Python file is over 1000 lines
check-file-length:
    uv run python scripts/check_file_length.py

# Run tests
test *args:
    uv run pytest tests/ -v {{args}}

# Run tests the way CI does (async tests excluded)
test-ci:
    uv run pytest tests/ -v -m "not asyncio" --tb=short

# Run CI-equivalent checks (GitHub Actions calls these same recipes)
ci: lint check-file-length test-ci

# Format then lint
check-code: format lint
