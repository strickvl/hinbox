"""Pick the model for each pipeline task.

By default every task uses the one model the run is set up for:
``HINBOX_OLLAMA_MODEL`` with ``--local``, else ``HINBOX_CLOUD_MODEL``. Setting
``HINBOX_<TASK>_MODEL`` (for example ``HINBOX_MATCH_MODEL=ollama/qwen3.6:35b``)
gives that one task its own model. The model's prefix then decides whether it
runs locally (``ollama/``) or in the cloud, whatever the run's default is.

A few tasks have their own cloud default (``CLOUD_TASK_DEFAULTS``). Grounding
uses Claude Haiku 5.5 in cloud runs: on 64 hand-labelled claims it passed 2 of
42 wrong or partly wrong claims as "supported", where GPT-6 Luna passed 4 and
local Qwen 17. Local runs keep the local model, so nothing leaves the machine.
"""

import os
from typing import Dict

from src.constants import CLOUD_MODEL, OLLAMA_MODEL

# task -> what it covers
TASKS: Dict[str, str] = {
    "relevance": "article relevance check (--relevance-check)",
    "extraction": "entity extraction from articles",
    "match": "merge check: are two profiles the same entity?",
    "dispute": "second-opinion merge dispute agent",
    "profile": "profile writing, updating and reflection",
    "grounding": "grounding check: does each cited article support its claims?",
}


# Cloud runs only; needs ANTHROPIC_API_KEY and ``uv sync --extra anthropic``.
CLOUD_TASK_DEFAULTS: Dict[str, str] = {"grounding": "anthropic/claude-haiku-5-5"}


def _env_name(task: str) -> str:
    return f"HINBOX_{task.upper()}_MODEL"


def task_model(task: str, model_type: str) -> str:
    """The model for ``task``: its own override, else the run's default model."""
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}; expected one of {sorted(TASKS)}")
    override = os.getenv(_env_name(task), "").strip()
    if override:
        return override
    if model_type == "ollama":
        return OLLAMA_MODEL
    return CLOUD_TASK_DEFAULTS.get(task, CLOUD_MODEL)


def is_local_model(model: str) -> bool:
    return model.startswith("ollama/")


def check_task_models(model_type: str) -> Dict[str, str]:
    """Resolve every task's model, refusing cloud models in a local run.

    ``--local`` promises that nothing leaves the machine, so a cloud override
    there is a configuration error rather than something to quietly allow.
    """
    models = {task: task_model(task, model_type) for task in TASKS}
    if model_type == "ollama":
        cloud = {t: m for t, m in models.items() if not is_local_model(m)}
        if cloud:
            names = ", ".join(f"{_env_name(t)}={m}" for t, m in cloud.items())
            raise ValueError(
                f"--local run, but these tasks are set to cloud models: {names}. "
                "Use ollama/ models or drop the override."
            )
    return models
