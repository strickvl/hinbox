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

# The reader as a static site in <out> (plain HTML, no server; holds article text)
export-site out *args:
    uv run python -m src.frontend.export --out {{out}} {{args}}

# The reader on a small made-up corpus (data/demo), for trying it without a run
demo-reader:
    uv run python -m scripts.demo_reader_data data/demo
    HINBOX_OUTPUT_DIR=data/demo HINBOX_ARTICLES_PATH=data/demo/articles.parquet HINBOX_CORRECTIONS_PATH=data/demo/corrections.jsonl uv run python -m src.frontend

# Trace viewer: Phoenix on http://localhost:6006 (own uvx env, not in Hinbox's lockfile).
# Then run the pipeline with OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://localhost:6006/v1/traces
phoenix:
    PHOENIX_WORKING_DIR="${PHOENIX_WORKING_DIR:-$HOME/.phoenix}" uvx --python 3.12 --from arize-phoenix phoenix serve
alias ui := frontend

# Speed + quality baseline on a fixed article set, fresh and cached passes
# (e.g. `just speed-baseline run --out bench --label before --set max_buffered_articles=1000`)
speed-baseline *args:
    uv run python -m scripts.speed_baseline {{args}}

# Give a finished run's grounding verdicts the article passages their reasoning quotes, no model call
# (e.g. `just grounding-passages --entities <run> --articles <parquet> --out <copy>`)
grounding-passages *args:
    uv run python -m scripts.grounding_passages {{args}}

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

# Turn trace files into chat-format training examples, one file per task
# (e.g. `just export-training data/guantanamo/entities/traces/*.jsonl --out data/guantanamo/entities/training`)
export-training *args:
    uv run python -m scripts.export_training_data {{args}}

# ─── DSPy experiments (own environment in experiments/dspy_opt/, no LiteLLM) ──

# Optimise the edit-list profile updater with DSPy GEPA (e.g. `just dspy-profile --model ollama/local-qwen36-16k:latest --reflection-model openai/gpt-6-luna`)
dspy-profile *args:
    uv run --project experiments/dspy_opt python -m experiments.dspy_opt.profile_update {{args}}

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
