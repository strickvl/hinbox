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
- **Run CI checks locally**: `just ci` (lint, file-length check, tests — the same recipes GitHub Actions calls)
- **File length limit**: `just check-file-length` fails if any tracked `.py` file is over 1000 lines. It runs in `just ci` and in GitHub Actions. Split the file rather than raising the limit.
- **Install dependencies**: `uv sync`

### Main Application
- **Process articles**: `just process` (CLI interface to main functionality)
- **Check database**: `just check` (view article statistics)
- **Start web interface**: `just frontend` (FastHTML web UI on localhost:5001)
- **Reset processing**: `just reset` (reset article processing status)
- **Domain management**: `just domains` / `just init <name>`
- **Evaluate a model**: `just eval <relevance|merge> --model <model>` scores it against the hand-labelled gold sets in `{output_dir}/eval/`; `just eval show` prints the scoreboard; `just eval-mine` picks hard merge pairs to label

### Testing
- Run `just test` to execute the suite under `tests/`.
- Pass extra pytest args: `just test -k test_merger --tb=short`
- CI runs lint + tests on every PR via `.github/workflows/test.yml`.
- Key coverage areas include embedding similarity, lexical blocking, per-type threshold resolution, embedding fingerprints, entity mergers, merge dispute agent routing, extraction caching/retry, canonical name selection, name variant detection, profile grounding, profile versioning, privacy mode enforcement, LLM multi-tool-call recovery, and frontend version navigation.
- Async tests (`@pytest.mark.asyncio`) are excluded in CI due to a missing pytest-asyncio configuration.

## Architecture Overview

### Entity Processing Pipeline
The CLI entry point `src/process_and_extract.py` coordinates the pipeline end to end using a parallel producer/consumer model:

1. **Configuration Loading**: `DomainConfig` reads `configs/<domain>/` to resolve Parquet input paths and output directories.
2. **Article Loading**: PyArrow loads the domain's article table and normalises rows before processing.
3. **Relevance Checking**: `ArticleProcessor.check_relevance` calls the helpers in `src/engine/relevance.py` (Gemini or Ollama) to skip irrelevant sources. Only runs with `--relevance-check`. If the check itself fails (API error, bad response), the article is skipped with reason `relevance_error`, kept separate from a real `not_relevant` verdict, and retried on the next run. A `not_relevant` verdict is stored in the processing-status sidecar with the article's content hash and a fingerprint of the model + relevance prompt (`ArticleProcessor.relevance_fingerprint()`); later runs reuse it without an LLM call while all three are unchanged (respects `cache.articles.skip_if_unchanged`; `--force-reprocess` re-checks).
4. **Parallel Extraction**: Multiple extraction workers process articles concurrently via `ThreadPoolExecutor`. Within each article, the 4 entity-type extractions also run in parallel. A shared LLM semaphore (`configure_llm_concurrency()`) bounds cloud API concurrency.
5. **Extraction Caching**: `ExtractionSidecarCache` (`src/utils/extraction_cache.py`) stores extraction outputs as JSON files keyed on content hash, model, entity type, prompt hash, schema hash, and temperature. Version-based invalidation lets you bump the cache version in config to force re-extraction without deleting files.
6. **Quality Controls & Retry**: `src/utils/quality_controls.py` validates extraction output (required fields, name normalisation, within-article dedup) and profile quality (min length, citation regex, tag count, confidence range). Failures are captured in `PhaseOutcome` objects (`src/utils/outcomes.py`). When severe QC flags (`zero_entities`, `high_drop_rate`, `many_duplicates`, `many_low_quality_names`) are detected, the extractor automatically retries once with a repair hint. Profile QC runs inside `create_profile()` / `update_profile()` before each version is saved: it applies safe fixes (default tags, clamped confidence) and records `{"passed", "flags"}` under `profile["qc"]`. A failing profile is kept and logged, never regenerated.
7. **Entity Merging**: `EntityMerger` follows an evidence-first cost structure — cheap checks run before expensive LLM calls:
   - **Lexical blocking**: RapidFuzz pre-filters candidates using configurable thresholds from the `dedup` section of domain config.
   - **Batched embeddings**: `embed_batch_result_sync()` computes vectors for all new entities at once rather than one-by-one.
   - **Similarity scoring**: Per-entity-type similarity thresholds and embedding fingerprints (`"{model}:{dim}"`) ensure model-change detection.
   - **Match checking**: LLM-based match verification only runs for candidates that pass the cheap filters.
   - **Merge dispute agent**: When a match result falls in the "gray band" (similarity within ±`MERGE_GRAY_BAND_DELTA` of threshold) with confidence below `MERGE_UNCERTAIN_CONFIDENCE_CUTOFF`, `MergeDisputeAgent` provides a second-stage LLM analysis that can override the initial merge/skip decision. Deferred cases are appended to `{output_dir}/merge_review_queue.jsonl` for human review. When the final verdict is skip or defer, the incoming entity is created as a new entity rather than dropped (unless an entity with the same key already exists, which is never overwritten).
   - **Canonical name selection**: 5-layer deterministic scoring (`score_canonical_name()` in `src/utils/name_variants.py`) picks the best display name, penalizing acronyms, generic phrases, and contextual suffixes.
8. **Profile Versioning**: `src/engine/profiles.py` maintains `VersionedProfile` history whenever entity content changes. Both creation and updates generate a first draft, then run up to 3 reflect/regenerate rounds (`iterative_improve()`).
9. **Profile Grounding**: Post-processing verification (`verify_profile_grounding()`) extracts citation markers from profile text, looks up the cited source articles, and uses an LLM to check whether each claim is supported. The `GroundingReport` includes a grounding score and per-claim support levels.
10. **Persistence**: Batched Parquet writes per entity type via `write_entities_table()` avoid write amplification. Every row is given every key before writing, because Arrow takes its columns from the first row only. Article processing status is tracked in a sidecar JSON file (`ProcessingStatus`) rather than rewriting the articles Parquet.

### Data Flow Architecture
- **Input**: Domain configs point to Parquet files with columns such as `id`, `title`, `content`, `url`, and `published_date`.
- **Processing**: `ArticleProcessor` orchestrates extraction for four entity types, records reflection metadata, runs QC checks (with retry), and keeps track of processing status. Each phase returns a `PhaseOutcome` carrying success/failure context. A single merge actor (the main thread) consumes extraction results in article order and is the only writer to shared state, so no locking is needed.
- **Output**: Each run updates people/organizations/locations/events tables under the domain's output directory (`DomainConfig.get_output_dir()`).
- **Storage**: Entities include embeddings (with model/dimension/fingerprint metadata), provenance metadata, processing timestamps, profile version histories, and optional grounding reports.

### Model Architecture
- **Cloud Models**: Defaults come from `CLOUD_MODEL` in `src/constants.py` (default: `gemini/gemini-2.0-flash`) and are executed through provider-routed SDK clients (OpenAI SDK for OpenAI-compatible endpoints, Anthropic SDK for Claude models) with Instructor wrappers in `src/utils/llm.py`. Provider routing is handled by `src/utils/provider_routing.py`. Multi-tool-call recovery handles Instructor edge cases.
- **Local Models**: Ollama is accessed via `OLLAMA_MODEL` (default: `ollama/qwen2.5:32b-instruct-q5_K_M`) for extraction, relevance checks, and match verification. Both models can be overridden via environment variables (`HINBOX_CLOUD_MODEL`, `HINBOX_OLLAMA_MODEL`). Ollama calls send `reasoning_effort="none"` (thinking off) unless `HINBOX_OLLAMA_REASONING_EFFORT` says otherwise. `_sampling_kwargs()` in `src/utils/llm.py` adapts request parameters per model: OpenAI gpt-5/gpt-6/o-series get `max_completion_tokens` (plus `reasoning_effort="none"` where supported, so tool calling works), and Claude 5-family models get no `temperature`.
- **Embeddings**: `EmbeddingManager` (`src/utils/embeddings/manager.py`) chooses cloud/local/hybrid providers and caches vectors for similarity scoring. When `--local` is active, `ensure_local_embeddings_available()` enforces local-only mode.
- **Structured Output**: Dynamic Pydantic models in `src/dynamic_models.py` and list factories enforce schema consistency for both cloud and local responses.
- **Privacy Mode**: `--local` CLI flag calls `disable_llm_callbacks()` to disable all telemetry and forces local embedding mode.

### Frontend Architecture
The web interface (`src/frontend/`) uses FastHTML with an "Archival Elegance" design theme:
- **Routes**: Modular route handlers in `routes/` (home, people, organizations, locations, events)
- **Data Access**: Centralized data loading from Parquet files (`data_access.py`)
- **Filtering**: Search and filter utilities (`filters.py`)
- **Components & Helpers**: Shared UI building blocks (`components.py`) — confidence badges, version selectors, tag pills, alias display — and helpers for profile versions (`entity_helpers.py`)
- **Configuration**: App setup, shared state, and `main_layout()` sidebar+content layout (`app_config.py`)
- **Design System**: Crimson Pro headings, IBM Plex Sans body, warm teal-slate primary (`#2c5f7c`), amber accent (`#c97b3a`) — CSS variables in `static/styles.css`
- **Static Assets**: CSS and font loading under `static/`

### File Structure Patterns
- **Engine Modules**: `article_processor.py`, `extractors.py`, `mergers.py` (with helpers `merge_candidates.py` for lexical blocking + similarity search and `merge_evidence.py` for evidence text + batch embedding), `match_checker.py`, `merge_dispute_agent.py`, and `profiles.py` are surfaced via `src/engine/__init__.py` for a stable import path.
- **LLM Helpers**: `src/utils/llm.py` and `src/utils/extraction.py` wrap SDK/Instructor interactions. `src/utils/provider_routing.py` resolves model prefixes to SDK targets. Multi-tool-call recovery in `llm.py` handles Instructor edge cases.
- **Embeddings**: Providers, manager, and similarity helpers live in `src/utils/embeddings/`.
- **Caching**: `src/utils/extraction_cache.py` provides the persistent sidecar cache; `src/utils/cache_utils.py` has a thread-safe LRU cache and stable hashing helpers shared across modules.
- **Name Handling**: `src/utils/name_variants.py` provides deterministic name normalisation, acronym detection/generation, equivalence expansion, and canonical name scoring — used by both QC and the merge pipeline.
- **Quality Controls**: `src/utils/quality_controls.py` (extraction QC, profile QC, profile grounding verification) and `src/utils/outcomes.py` (`PhaseOutcome` structured results) provide deterministic validation.
- **Processing Status**: `src/utils/processing_status.py` manages a sidecar JSON file tracking which articles have been processed, replacing the old in-Parquet status approach.
- **Tests**: `tests/` covers embedding accuracy, merger behaviour (lexical blocking, per-type thresholds, fingerprints), merge dispute agent routing, extraction caching and retry, canonical name selection, name variant detection, profile grounding, profile versioning, privacy mode enforcement, domain path resolution, and frontend history rendering.
- **Evaluation**: `src/evals/` holds the scoreboard: `datasets.py` loads gold sets into `EvalItem`s and assigns stable train/dev/test splits, `metrics.py` computes balanced accuracy and friends, `runner.py` runs a predictor (default: the pipeline's own relevance/match-check functions) and appends to `eval/scoreboard.jsonl`, and `mine_merge_pairs.py` mines hard look-alike entity pairs for labelling.
- **Scripts**: Utility scripts in `scripts/` support domain management (`init_domain.py`, `list_domains.py`), data fetching, resets, and diagnostics.

### Configuration
- **Models**: Configured in `src/constants.py` with cloud/local model specifications. Override defaults via `HINBOX_CLOUD_MODEL` and `HINBOX_OLLAMA_MODEL` env vars.
- **Dedup**: Per-entity-type similarity thresholds, lexical blocking, and merge gray-band/confidence settings configured in the `dedup` section of `configs/<domain>/config.yaml`
- **Extraction Cache**: Version-based invalidation via `cache.extraction.version` in domain config; cache files live under `{output_dir}/cache/extractions/v{version}/`
- **Logging**: Structured Rich-based logging in `src/logging_config.py` with colour-coded decision lines (`DecisionKind`: NEW, MERGE, SKIP, DISPUTE, DEFER, ERROR) and gated profile panels (`--show-profiles`)
- **Privacy**: `--local` flag calls `disable_llm_callbacks()` and forces local embedding mode. The `_CALLBACKS_ENABLED` flag in `constants.py` controls telemetry.
- **Environment**: Requires `GEMINI_API_KEY` for cloud processing, optional `OLLAMA_API_URL` for local

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
