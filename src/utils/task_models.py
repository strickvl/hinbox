"""Pick the model for each pipeline task.

By default every task uses the one model the run is set up for:
``HINBOX_OLLAMA_MODEL`` with ``--local``, else ``HINBOX_CLOUD_MODEL``. Setting
``HINBOX_<TASK>_MODEL`` (for example ``HINBOX_MATCH_MODEL=ollama/qwen3.6:35b``)
gives that one task its own model. The model's prefix then decides whether it
runs locally (``ollama/``) or in the cloud, whatever the run's default is.
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
}


def _env_name(task: str) -> str:
    return f"HINBOX_{task.upper()}_MODEL"


def task_model(task: str, model_type: str) -> str:
    """The model for ``task``: its own override, else the run's default model."""
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}; expected one of {sorted(TASKS)}")
    override = os.getenv(_env_name(task), "").strip()
    if override:
        return override
    return OLLAMA_MODEL if model_type == "ollama" else CLOUD_MODEL


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
