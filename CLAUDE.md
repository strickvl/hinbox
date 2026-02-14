# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository. Any design docs that you write can be stored in the
`design/` folder (which is ignored by git).

## Running commands

We use a `justfile` as the single task runner. Run `just` to see all available
commands, or `just <command>` to execute one. All recipes use `uv run` to
guarantee the lockfile venv is used (matching CI).

## Common Commands

### Development
- **Format code**: `just format` (ruff fix + format via `uv run`)
- **Lint code**: `just lint` (ruff check + format verification)
- **Run CI checks locally**: `just ci` (exactly what GitHub Actions runs)
- **Install dependencies**: `uv sync`

### Main Application
- **Process articles**: `just process` (CLI interface to main functionality)
- **Check database**: `just check` (view article statistics)
- **Start web interface**: `just frontend` (FastHTML web UI on localhost:5001)
- **Reset processing**: `just reset` (reset article processing status)
- **Domain management**: `just domains` / `just init <name>`

### Testing
- Run `just test` to execute the suite under `tests/`.
- Pass extra pytest args: `just test -k test_merger --tb=short`
- CI runs lint + tests on every PR via `.github/workflows/test.yml`.
- Key coverage areas include embedding similarity, lexical blocking, per-type threshold resolution, embedding fingerprints, entity mergers, profile versioning, and frontend version navigation.
- Async tests (`@pytest.mark.asyncio`) are excluded in CI due to a missing pytest-asyncio configuration.

## Architecture Overview

### Entity Processing Pipeline
The CLI entry point `src/process_and_extract.py` coordinates the pipeline end to end:

1. **Configuration Loading**: `DomainConfig` reads `configs/<domain>/` to resolve Parquet input paths and output directories.
2. **Article Loading**: PyArrow loads the domain's article table and normalises rows before processing.
3. **Relevance Checking**: `ArticleProcessor.check_relevance` calls the helpers in `src/engine/relevance.py` (Gemini or Ollama) to skip irrelevant sources.
4. **Entity Extraction**: `ArticleProcessor.extract_all_entities` dispatches to `EntityExtractor` for people, organizations, locations, and events using dynamic Pydantic models.
5. **Quality Controls**: `src/utils/quality_controls.py` validates extraction output (required fields, name normalisation, within-article dedup) and profile quality (min length, citation regex, tag count, confidence range). Failures are captured in `PhaseOutcome` objects (`src/utils/outcomes.py`) rather than silently swallowed.
6. **Entity Merging**: `EntityMerger` pre-filters candidates with RapidFuzz lexical blocking, compares embeddings via `src/utils/embeddings`, optionally consults the LLM match checker, and updates in-memory stores. Per-entity-type similarity thresholds and lexical blocking settings are loaded from the `dedup` section of domain config. Embedding fingerprints (`"{model}:{dim}"`) are stored on each entity for model-change detection.
7. **Profile Versioning**: `src/engine/profiles.py` maintains `VersionedProfile` history whenever entity content changes.
8. **Persistence**: Updated article rows and entity tables are written back with `src/utils/file_ops.write_entity_to_file` and atomically swapped Parquet files.

### Data Flow Architecture
- **Input**: Domain configs point to Parquet files with columns such as `id`, `title`, `content`, `url`, and `published_date`.
- **Processing**: `ArticleProcessor` orchestrates extraction for four entity types, records reflection metadata, runs QC checks, and keeps track of processing status. Each phase returns a `PhaseOutcome` carrying success/failure context.
- **Output**: Each run updates people/organizations/locations/events tables under the domain's output directory (`DomainConfig.get_output_dir()`).
- **Storage**: Entities include embeddings (with model/dimension/fingerprint metadata), provenance metadata, processing timestamps, and profile version histories.

### Model Architecture
- **Cloud Models**: Defaults come from `CLOUD_MODEL` in `src/constants.py` and are executed through LiteLLM wrappers in `src/utils/llm.py`.
- **Local Models**: Ollama is accessed via `OLLAMA_MODEL` for extraction, relevance checks, and match verification.
- **Embeddings**: `EmbeddingManager` (`src/utils/embeddings/manager.py`) chooses cloud/local/hybrid providers and caches vectors for similarity scoring.
- **Structured Output**: Dynamic Pydantic models in `src/dynamic_models.py` and list factories enforce schema consistency for both cloud and local responses.

### Frontend Architecture
The web interface (`src/frontend/`) uses FastHTML and is organized as:
- **Routes**: Modular route handlers in `routes/` (home, people, organizations, locations, events)
- **Data Access**: Centralized data loading from Parquet files (`data_access.py`)
- **Filtering**: Search and filter utilities (`filters.py`)
- **Components & Helpers**: Shared UI building blocks (`components.py`) and helpers for profile versions (`entity_helpers.py`)
- **Configuration**: App setup and shared state (`app_config.py`)
- **Static Assets**: CSS and JS bundling lives under `static/`

### File Structure Patterns
- **Engine Modules**: `article_processor.py`, `extractors.py`, `mergers.py`, `match_checker.py`, and `profiles.py` are surfaced via `src/engine/__init__.py` for a stable import path.
- **LLM Helpers**: `src/utils/llm.py` and `src/utils/extraction.py` wrap LiteLLM/Ollama interactions and Instructor responses.
- **Embeddings**: Providers, manager, and similarity helpers live in `src/utils/embeddings/`.
- **Quality Controls**: `src/utils/quality_controls.py` (extraction QC) and `src/utils/outcomes.py` (`PhaseOutcome` structured results) provide deterministic validation.
- **Tests**: `tests/` covers embedding accuracy, merger behaviour (including lexical blocking and per-type thresholds), config threshold resolution, embedding fingerprints, profile versioning, domain path resolution, and frontend history rendering.
- **Scripts**: Utility scripts in `scripts/` support data fetching, resets, and diagnostics.

### Configuration
- **Models**: Configured in `src/constants.py` with cloud/local model specifications
- **Dedup**: Per-entity-type similarity thresholds and lexical blocking configured in the `dedup` section of `configs/<domain>/config.yaml`
- **Logging**: Centralized Rich-based logging in `src/logging_config.py` with color-coded levels
- **Environment**: Requires `GEMINI_API_KEY` for cloud processing, optional
  `OLLAMA_API_URL` for local

## Workflow Notes
- When finishing a chunk of work, check with the user to confirm the fix, then:
  1. Run `just format` to auto-fix formatting
  2. Run `just lint` to verify no remaining issues
  3. Run `just test` to execute the test suite
  4. Commit and push changes
- **Before pushing / opening a PR**, run `just ci` which executes the exact same
  checks as GitHub Actions:
  ```bash
  just ci
  ```
  All justfile recipes use `uv run` so there are no version mismatches between
  local and CI.

## Development Guidance
- The application has no users yet, so don't worry too much about backwards compatibility. Just make it work.

## Code Conventions
- When using type hints for dicts or tuples, prefer `typing.Dict` / `typing.Tuple` over the built-in generics for consistency with existing code.
