"""Run a command before and after the citation check, to change model servers.

The citation check (grounding pass) runs after all merging, so it can use
another local model than the rest of the run without both being loaded at
once: the pipeline model does its work, the server is swapped, the judge
checks every cited claim, and the server is swapped back. On one 24 GB card
that is the only way to pair a fast pipeline model with a careful judge (on
64 hand-labelled claims Inkling 2 Mini passed 6 of 42 wrong claims as
supported, local Qwen 19, but it is about 11 times slower, see README
"Choosing a grounding judge").

``HINBOX_BEFORE_GROUNDING_CMD`` is a shell command Hinbox runs once the
merged entities are saved and before the first judge call, for example a
script that stops the pipeline model's server and starts the judge's at the
same address. Hinbox then waits up to ``HINBOX_JUDGE_WAIT_SECONDS`` (default
600) for the local server to answer. ``HINBOX_AFTER_GROUNDING_CMD`` runs when
the check has ended, also when it failed or was interrupted, so the pipeline
model comes back. Neither is set by default, and a run without them is
unchanged.
"""

import os
import subprocess
import time
from contextlib import contextmanager
from typing import Callable, Iterator, Optional

import requests

from src.constants import OLLAMA_API_URL
from src.logging_config import log
from src.utils import server_health
from src.utils.server_health import ModelServerUnreachable
from src.utils.task_models import is_local_model, task_model

BEFORE_ENV = "HINBOX_BEFORE_GROUNDING_CMD"
AFTER_ENV = "HINBOX_AFTER_GROUNDING_CMD"
WAIT_ENV = "HINBOX_JUDGE_WAIT_SECONDS"


class JudgeSwapFailed(ModelServerUnreachable):
    """The swap command failed, or the judge's server never answered. The run
    stops the way it does for an unreachable server: what is merged is saved
    and the citation check is left for the next run."""


def before_command() -> str:
    return os.getenv(BEFORE_ENV, "").strip()


def after_command() -> str:
    return os.getenv(AFTER_ENV, "").strip()


def wait_seconds() -> float:
    return float(os.getenv(WAIT_ENV, "600"))


def server_answers(base_url: str = OLLAMA_API_URL, timeout: float = 5.0) -> bool:
    """True when the local server lists its models (Ollama and llama.cpp both
    serve ``/v1/models``); a server still loading its model does not."""
    try:
        return requests.get(f"{base_url.rstrip('/')}/models", timeout=timeout).ok
    except requests.RequestException:
        return False


def run_command(command: str, what: str) -> None:
    """Run ``command`` through the shell; raise if it exits with an error."""
    log(f"Running the {what} command: {command}", level="processing")
    done = subprocess.run(command, shell=True, check=False)
    if done.returncode != 0:
        raise JudgeSwapFailed(
            f"the {what} command exited with {done.returncode}: {command}"
        )


def wait_for_server(
    answers: Optional[Callable[[], bool]] = None,
    limit: Optional[float] = None,
    pause: float = 5.0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    """Wait until the local server answers; raise after ``limit`` seconds."""
    answers = answers or server_answers
    limit = wait_seconds() if limit is None else limit
    start = clock()
    while not answers():
        if clock() - start >= limit:
            raise JudgeSwapFailed(
                f"the model server at {OLLAMA_API_URL} did not answer within "
                f"{limit:.0f} s of the swap ({WAIT_ENV})"
            )
        sleep(pause)


def swap_to_judge(local_judge: bool) -> bool:
    """Run the before-grounding command, if one is set, and wait for a local
    judge's server. Returns whether a command ran."""
    command = before_command()
    if not command:
        return False
    run_command(command, "before-grounding")
    if local_judge:
        wait_for_server()
        log("The judge's model server is answering", level="success")
    return True


def swap_back() -> bool:
    """Run the after-grounding command, if one is set. A failure is logged,
    not raised: the check's results are already in memory and must be saved."""
    command = after_command()
    if not command:
        return False
    try:
        run_command(command, "after-grounding")
    except JudgeSwapFailed as e:
        log(f"{e}. Start the pipeline's model server by hand.", level="error")
    return True


@contextmanager
def judge_server(model_type: str, save: Callable[[], None]) -> Iterator[None]:
    """The citation check goes inside this block. With a before-grounding
    command set, ``save()`` writes the merged entities first, so a swap that
    goes wrong loses nothing; the after-grounding command runs on the way
    out, whatever happened inside."""
    if before_command():
        save()
    try:
        judge = task_model("grounding", model_type)
        if swap_to_judge(is_local_model(judge)):
            # Calls that failed against the old server say nothing about this one
            server_health.reset()
        yield
    finally:
        swap_back()
