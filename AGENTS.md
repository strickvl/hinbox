# Repository Guidelines

This document helps contributors work effectively in this repository.

## Project Structure & Module Organization
- `src/` — application code
  - `dynamic_models.py` (Pydantic models generated from domain configs)
  - `frontend/` (FastHTML app: routes, components, config)
  - `utils/` (embeddings, extraction helpers, logging, errors)
  - Other modules: `events.py`, `people.py`, `organizations.py`, etc.
- `configs/` — domain definitions (types, prompts) by domain (e.g., `guantanamo/`).
- `tests/` — pytest tests (`test_*.py`).
- `scripts/` — ad‑hoc utilities (may be stale; prefer core modules).
- `assets/`, `data/`, `design/` — static assets, datasets, notes.

## Build, Test, and Development Commands
- Create env + install (editable + dev):
  - `source .venv/bin/activate`
  - `uv pip install -e .[dev]`
- Run tests: `pytest`
- Lint: `bash scripts/lint.sh`
- Format: `bash scripts/format.sh`
- Launch frontend locally (default port 5001):
  - `python -m src.frontend`
  - Alt: `uvicorn src.frontend:app --port 5001`

## Coding Style & Naming Conventions
- Python 3.10+; 4‑space indent; max line length 88.
- Use type hints; Pydantic models for schemas.
- Names: modules `snake_case.py`, classes `CapWords`, constants `UPPER_SNAKE`.
- Import order and linting via Ruff; auto‑format with Black (see `pyproject.toml`).

## Testing Guidelines
- Framework: pytest (`tests/`, files `test_*.py`).
- Write focused unit tests per module; prefer pure functions and fixtures.
- Run locally: `pytest -q`; add regression tests for bug fixes.

## Commit & Pull Request Guidelines
- Messages: imperative, concise headers (e.g., “Add X”, “Fix Y”).
- Include context in the body when non‑trivial; reference issues.
- PRs: clear description, rationale, and testing notes; attach screenshots for UI changes.

## Security & Configuration Tips
- Model/API settings live in `src/constants.py`; avoid committing secrets.
- Dynamic schemas come from `configs/<domain>/`; prefer `dynamic_models.py` over static models.
- Some `scripts/` are legacy; validate before use.

## Agent‑Specific Notes
- Treat `configs/` as the source of truth for entity types/tags.
- Prefer `src/extractors.py` abstractions over duplicating cloud/local logic.
