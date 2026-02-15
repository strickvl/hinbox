# Repository Guidelines

This document helps contributors work effectively in this repository.

## Project Structure & Module Organization
- `src/` — application code
  - `engine/` — core pipeline modules:
    - `article_processor.py` — relevance → extraction → QC retry orchestration
    - `extractors.py` — unified cloud/local entity extraction via Instructor
    - `mergers.py` — lexical blocking → batched embeddings → similarity → match check → dispute agent → canonical name selection
    - `match_checker.py` — LLM-based match verification
    - `merge_dispute_agent.py` — second-stage arbitration for gray-band entity matches
    - `profiles.py` — `VersionedProfile` history management
    - `relevance.py` — domain-specific relevance filtering
  - `process_and_extract.py` — CLI pipeline entry point; parallel producer/consumer with `ThreadPoolExecutor` workers and a single-threaded merge actor
  - `dynamic_models.py` — Pydantic models generated from domain configs
  - `frontend/` — FastHTML "Archival Elegance" UI (routes, components, config). See `src/frontend/CLAUDE.md` for design system details.
  - `utils/` — shared utilities:
    - `embeddings/` — `EmbeddingManager`, cloud/local providers, similarity helpers
    - `extraction_cache.py` — persistent sidecar cache for extraction results (keyed on content hash, model, prompt, schema, temperature)
    - `cache_utils.py` — thread-safe LRU cache and stable hashing helpers
    - `name_variants.py` — deterministic name normalisation, acronym detection, equivalence expansion, canonical name scoring
    - `quality_controls.py` — extraction QC, profile QC, and profile grounding verification
    - `processing_status.py` — sidecar JSON tracker for article processing status
    - `outcomes.py` — `PhaseOutcome` structured result objects
    - `llm.py` — LiteLLM/Instructor wrappers with multi-tool-call recovery
    - `extraction.py` — extraction dispatch with sidecar cache integration
    - `file_ops.py` — Parquet I/O, batched writes, atomic file operations
    - `error_handler.py`, `logging.py` — error handling and logging utilities
  - `constants.py` — model defaults, embedding settings, thresholds, privacy controls
  - `config_loader.py` — `DomainConfig` loader (per-type thresholds, lexical blocking, cache settings)
  - `logging_config.py` — Rich-based structured logging with colour-coded decision lines
  - `exceptions.py` — custom exception types used across the pipeline
- `configs/` — domain definitions (types, prompts, dedup settings) by domain (e.g., `guantanamo/`).
- `tests/` — pytest tests (`test_*.py`, `embeddings/` sub-suite). Covers mergers, dispute routing, extraction caching/retry, canonical names, name variants, profile grounding/versioning, privacy mode, and frontend.
- `scripts/` — domain management (`init_domain.py`, `list_domains.py`), data fetching, resets, diagnostics.
- `assets/`, `data/`, `design/` — static assets, datasets, design notes (design/ is gitignored).

## Build, Test, and Development Commands
- **Task runner**: `just` (see `justfile`). All recipes use `uv run` to match CI.
- Install: `uv sync` (cloud only) or `uv sync --extra local-embeddings` (with PyTorch)
- Run tests: `just test` (pass extra pytest args: `just test -k test_merger`)
- Lint: `just lint`
- Format: `just format`
- CI-equivalent checks: `just ci` (exactly what GitHub Actions runs)
- Process articles: `just process --domain <name> --limit N`
- Start frontend: `just frontend` (http://localhost:5001)
- Domain management: `just domains` / `just init <name>`

## Coding Style & Naming Conventions
- Python 3.12+; 4-space indent; max line length 88.
- Use type hints (`typing.Dict` / `typing.Tuple` preferred over built-in generics for consistency).
- Pydantic models for schemas; Instructor for structured LLM output.
- Names: modules `snake_case.py`, classes `CapWords`, constants `UPPER_SNAKE`.
- Linting and formatting via Ruff (see `pyproject.toml`).

## Testing Guidelines
- Framework: pytest (`tests/`, files `test_*.py`).
- Write focused unit tests per module; prefer pure functions and fixtures.
- All tests run without API keys or GPU — use mocks/stubs for LLM and embedding calls.
- Async tests (`@pytest.mark.asyncio`) are excluded in CI; mark them appropriately.
- Key coverage areas: embedding similarity, lexical blocking, per-type thresholds, fingerprints, merge dispute routing, extraction caching/retry, canonical name selection, name variants, profile grounding, profile versioning, privacy mode, LLM multi-tool-call recovery, frontend version navigation.

## Commit & Pull Request Guidelines
- Messages: imperative, concise headers (e.g., "Add X", "Fix Y").
- Include context in the body when non-trivial; reference issues.
- PRs: clear description, rationale, and testing notes; attach screenshots for UI changes.
- Before pushing: run `just ci` to match what GitHub Actions checks.

## Security & Configuration Tips
- Model/API settings live in `src/constants.py`; override via env vars (`HINBOX_CLOUD_MODEL`, `HINBOX_OLLAMA_MODEL`). Avoid committing secrets.
- `--local` flag enforces local-only embeddings and disables all LLM telemetry callbacks.
- Dynamic schemas come from `configs/<domain>/`; prefer `dynamic_models.py` over static models.
- Extraction cache version (`cache.extraction.version` in domain config) controls cache invalidation.
- Some `scripts/` are legacy; validate before use.

## Architecture Patterns
- **Evidence-first merge**: Cheap checks (lexical blocking, embedding similarity) run before expensive LLM calls (match checker, dispute agent).
- **Producer/consumer pipeline**: Extraction workers run in parallel; a single merge actor on the main thread consumes results in order (no locking needed).
- **Sidecar files**: Processing status and extraction cache use sidecar JSON files instead of mutating the source Parquet.
- **QC retry**: When extraction QC detects severe issues, the extractor retries once with a repair hint describing what went wrong.
- **Structured logging**: `DecisionKind` enum (`NEW`, `MERGE`, `SKIP`, `DISPUTE`, `DEFER`, `ERROR`) provides colour-coded per-entity decision lines.

## Agent-Specific Notes
- Treat `configs/` as the source of truth for entity types/tags.
- Prefer `src/engine/extractors.py` abstractions over duplicating cloud/local logic.
- When modifying merge behaviour, update both `mergers.py` and the corresponding test in `test_entity_merger_*.py` or `test_merge_dispute_agent_routing.py`.
- Name handling logic lives in `src/utils/name_variants.py` — don't duplicate in QC or merger code.
- The frontend has its own CLAUDE.md at `src/frontend/CLAUDE.md` with FastHTML patterns and CSS class references.
