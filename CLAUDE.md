# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository. Any design docs that you write can be stored in the
`design/` folder (which is ignored by git).

## Running commands

We have a `justfile` which allows you to run commands with `just <command>`. We
also have a `run.py` script which does similar things. When you update one, make
sure to update the other.

## Common Commands

### Development
- **Format code**: `./scripts/format.sh` (uses ruff for import sorting and formatting)
- **Lint code**: `./scripts/lint.sh` (runs ruff check and format)
- **Install dependencies**: `uv sync`

### Main Application
- **Process articles**: `./run.py process` (CLI interface to main functionality)
- **Check database**: `./run.py check` (view article statistics)
- **Start web interface**: `./run.py frontend` (FastHTML web UI on localhost:5001)
- **Reset processing**: `./run.py reset` (reset article processing status)

### Testing
No test suite is currently configured. The project uses runtime validation through Pydantic models.

## Architecture Overview

### Entity Processing Pipeline
The core functionality processes articles through a multi-stage pipeline:

1. **Article Loading**: Loads articles from `data/raw_sources/miami_herald_articles.parquet`
2. **Relevance Checking**: Filters articles relevant to Guantánamo Bay using LLMs (`src/relevance.py`)
3. **Entity Extraction**: Extracts 4 entity types using structured LLM calls:
   - People (`src/people.py`)
   - Organizations (`src/organizations.py`) 
   - Locations (`src/locations.py`)
   - Events (`src/events.py`)
4. **Entity Merging**: Deduplicates entities using embeddings similarity (`src/merge.py`)
5. **Profile Generation**: Creates comprehensive profiles for entities (`src/profiles.py`)

### Data Flow Architecture
- **Input**: Articles in Parquet format with columns like `title`, `content`, `url`, `processed`
- **Processing**: Each entity type has parallel extraction functions for cloud (Gemini) and local (Ollama) models
- **Output**: Separate Parquet files for each entity type in `data/entities/` with embeddings for similarity matching
- **Storage**: All entities include profile text, source article metadata, processing timestamps, and vector embeddings

### Model Architecture
- **Cloud Models**: Uses Gemini 2.0 Flash via LiteLLM for production processing
- **Local Models**: Supports Ollama with Gemma 27B for offline processing
- **Embeddings**: Uses Jina v3 embeddings for entity similarity and deduplication
- **Structured Output**: All LLM calls use Pydantic models with Instructor for type safety

### Frontend Architecture
The web interface (`src/frontend/`) uses FastHTML and is organized as:
- **Routes**: Modular route handlers in `routes/` (home, people, organizations, locations, events)
- **Data Access**: Centralized data loading from Parquet files (`data_access.py`)
- **Filtering**: Search and filter utilities (`filters.py`)
- **Configuration**: App setup and shared state (`app_config.py`)

### File Structure Patterns
- **Entity Processing**: Each entity type follows the pattern: `{type}.py` (extraction) + merge logic in `merge.py`
- **Dual Model Support**: All extraction modules have both `gemini_extract_*` and `ollama_extract_*` functions
- **Utilities**: Common functionality in `src/utils/` (embeddings, extraction, file ops, LLM wrappers)
- **Scripts**: Utility scripts in `scripts/` directory for maintenance tasks

### Configuration
- **Models**: Configured in `src/constants.py` with cloud/local model specifications
- **Logging**: Centralized Rich-based logging in `src/logging_config.py` with color-coded levels
- **Environment**: Requires `GEMINI_API_KEY` for cloud processing, optional
  `OLLAMA_API_URL` for local

## Workflow Notes
- When finishing a chunk of work, check with the user to confirm the fix, then:
  1. Run the formatting script (`./scripts/format.sh`)
  2. Fix any formatting errors
  3. Commit and push changes

## Development Guidance
- The application has no users yet, so don't worry too much about backwards compatibility. Just make it work.

## Code Conventions
- When using type hints for dicts or tuples, we use the Tuple or Dict from the `typing` module and not `tuple` or `dict` FYI