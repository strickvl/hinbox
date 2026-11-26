"""Run one recorded ``profile.update``, ``match_check``, ``extract.<type>`` or
``grounding`` (citation check) step again and record it in Kitaru.

Kitaru's worker starts this script for a replay of a step session (one that
``scripts/kitaru_traces.py --step <step>`` wrote, or ``just citecheck
kitaru-sessions``). The session's inputs are the step's own arguments; this
script:

1. reads them from the task (``KITARU_TASK_INPUTS``, or the task's spec when
   they were too long for the environment);
2. applies the replay's model override, if any (``KITARU_REPLAY_ID``);
3. calls the function the step ran, with those arguments: ``update_profile()``
   for a pipeline step, ``edit_step()`` (``edit_profile()``) for an eval step,
   as the recorded ``call`` says, ``_check_match()`` for a match check
   (its ``backend`` set from the model, so a local model is asked locally; a
   failed call fails the run, since its ``is_match=False`` is no verdict),
   ``ArticleProcessor._timed_extract()`` for an extraction step (raising when
   extraction failed so an empty fallback is never counted as a verdict), and
   ``judge_claims()`` for a citation check (a ``tinker://`` checkpoint or
   ``tinker/<base model>`` is sampled on Tinker,
   ``src/citecheck/tinker_judge.py``);
4. turns the spans that call left (the step and its model calls) into a
   session with the same code that converts traces for import
   (``step_sessions()``), so a replay and its baseline have the same shape;
5. writes it to Kitaru in one request (``origin: replay``) and exits 1 if
   the step raised.

Register it as each step agent's run command,
``kitaru agent register hinbox-profile-update --spec spec.yaml`` (and
``hinbox-match-check``, ``hinbox-extract``, ``hinbox-citation-check``), with::

    version:
      run_spec:
        command: uv run --extra kitaru python -m scripts.kitaru_step
        working_dir: <this checkout>
        runtime_capabilities: {overrides: true, tool_policies: false}

(Hinbox has no tools; Instructor's tool call is structured output.) Run the
worker with ``just kitaru-worker``, which gives it ``.env``'s API keys. Of the
replay overrides ``model`` and ``model_params`` are applied: the parameters
(``temperature``, ``max_tokens``, ``top_p``, ``seed``, ``reasoning_effort``,
``frequency_penalty``) go
on every model call the step makes, over Hinbox's own choice
(``set_model_params()``), and any other key fails the run. ``system_prompt``
and ``prompt`` fail it too, since a step's prompts are built in code (a
profile update makes several calls with different ones). Every replay makes
live model calls.

Outside a worker, ``--inputs <json> --out <file>`` runs a step from a file of
recorded inputs and writes the session as a Kitaru JSONL line instead.
"""

import argparse
import asyncio
import json
import os
import sys
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple

from scripts.kitaru_traces import step_sessions

PROFILE_STEP = "profile.update"
MATCH_STEP = "match_check"
GROUNDING_STEP = "grounding"
# Recorded beside update_profile()'s arguments but not one of them
NOT_ARGUMENTS = ("call", "answering_model", "labels", "guarded_corrections")


def hinbox_model(name: str) -> str:
    """Kitaru writes ``provider:model``; Hinbox ``provider/model``. A colon
    before the first slash is the provider's (``openrouter:anthropic/x``);
    one after it is a tag (``ollama/qwen2.5:32b``), kept. A Tinker checkpoint
    (``tinker://...``) is a URL, kept as it is; ``tinker:<path>`` without the
    slashes is read as one."""
    if name.startswith("tinker://"):
        return name
    if name.startswith("tinker:") and "/sampler_weights/" in name:
        return "tinker://" + name.split(":", 1)[1].lstrip("/")
    return name.replace(":", "/", 1) if ":" in name.split("/", 1)[0] else name


def step_name(inputs: Dict[str, Any]) -> str:
    """Which step recorded these arguments: only ``_check_match()`` takes an
    existing profile's text, only extraction takes ``article_content``, and
    only ``judge_claims()`` takes a list of claims."""
    if "existing_profile_text" in inputs:
        return MATCH_STEP
    if "article_content" in inputs and "entity_type" in inputs:
        return f"extract.{inputs['entity_type']}"
    if "claims" in inputs and "article_text" in inputs:
        return GROUNDING_STEP
    return PROFILE_STEP


def _model_of(inputs: Dict[str, Any]) -> str:
    return str(inputs.get("model") or inputs.get("answering_model") or "")


def apply_override(inputs: Dict[str, Any], override: Any) -> Dict[str, Any]:
    """The inputs with the replay override applied (``model`` only).

    A string replaces the model; a map replaces the step's model when it is
    one of its keys (in either spelling, or without the provider).
    """
    if override is None:
        return inputs
    unsupported = [f for f in ("system_prompt", "prompt") if getattr(override, f)]
    if unsupported:
        raise ValueError(
            f"Replay override {', '.join(unsupported)} is not supported: the "
            "step's prompts are built in code (a profile update makes several "
            "calls with different ones). Test a prompt change as a new agent "
            "version instead."
        )
    model = getattr(override, "model", None)
    if not model:
        return inputs
    current = _model_of(inputs)
    if isinstance(model, str):
        return {**inputs, "model": hinbox_model(model)}
    for old, new in model.items():
        if hinbox_model(old) in (current, current.split("/", 1)[-1]):
            return {**inputs, "model": hinbox_model(new)}
    raise ValueError(f"The model override has no entry for {current!r}")


def _match_check(**args: Any) -> Any:
    """``_check_match()``, raising when its call failed: the checker then
    returns ``is_match=False``, which a replay must not count as a verdict."""
    from src.engine.match_checker import _check_match, check_failed

    result = _check_match(**args)
    if check_failed(result):
        raise RuntimeError(result.reason)
    return result


def _extract_step(
    *,
    entity_type: str,
    article_content: str,
    article_id: str = "",
    domain: str = "guantanamo",
    model: str = "",
    model_type: Optional[str] = None,
) -> Any:
    """Run one ``extract.<entity_type>`` step (extraction, QC and retry).

    Raises when extraction failed so Kitaru marks the replay failed instead of
    treating an empty fallback list as a clean zero-entity result.
    """
    from src.engine.article_processor import ArticleProcessor
    from src.utils.task_models import is_local_model

    resolved_type = (
        ("ollama" if is_local_model(model) else "gemini")
        if model
        else (model_type or "gemini")
    )
    processor = ArticleProcessor(
        domain=domain,
        model_type=resolved_type,
        extraction_model=model or None,
    )
    outcome = processor._timed_extract(entity_type, article_content, article_id)
    if not outcome.success:
        err = outcome.error
        raise RuntimeError(f"{err.type}: {err.message}" if err else "extraction failed")
    return outcome


def apply_model_params(override: Any) -> None:
    """Put the override's ``model_params`` on every model call this process
    makes; an unsupported key raises."""
    params = getattr(override, "model_params", None)
    if params:
        from src.utils.llm import set_model_params

        set_model_params(dict(params))


def step_call(inputs: Dict[str, Any]) -> Tuple[Callable[..., Any], Dict[str, Any]]:
    """The function the recorded step ran, and the arguments to call it with.

    Refuses a pipeline step that ran with a historian's corrections guarded:
    the replay has no correction scope, so the guard would let a struck
    sentence back in and the replay would differ for that reason alone.
    """
    if inputs.get("guarded_corrections"):
        raise ValueError(
            f"This step ran with {inputs['guarded_corrections']} of the "
            "historian's sentences guarded (struck or written), which a replay "
            "can't restore yet; replaying it would let them be rewritten"
        )
    args = {k: v for k, v in inputs.items() if k not in NOT_ARGUMENTS}
    name = step_name(inputs)
    if name == GROUNDING_STEP:
        from src.utils.grounding import judge_claims

        model = str(args.get("model") or "")
        return judge_claims, {
            **args,
            "claims": [tuple(c) for c in args["claims"]],
            "model_type": "local" if model.startswith("ollama/") else "cloud",
        }
    if name == MATCH_STEP:
        return _match_check, {
            **args,
            "backend": "local" if args["model"].startswith("ollama/") else "cloud",
        }
    if name.startswith("extract."):
        return _extract_step, args
    if inputs.get("call") == "edit_profile":
        from src.evals.profile_update import edit_step

        return (lambda **kw: edit_step(kw)), args
    from src.engine.profiles import VersionedProfile, update_profile

    # The model that answered: a step recorded with no model argument used
    # the defaults of the run's environment, which a replay may not share
    args["model"] = _model_of(inputs) or None
    return update_profile, {**args, "versioned_profile": VersionedProfile()}


def _capture() -> Any:
    """Keep this process's finished spans in memory as well."""
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):  # tracing off or not set up
        raise RuntimeError("Tracing must be on to record the step")
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter


def recorded_session(
    rows: List[Dict[str, Any]], replay: bool, name: str = PROFILE_STEP
) -> Dict[str, Any]:
    """The ``name`` step's session from the re-run's spans."""
    step = next((r for r in rows if r["name"] == name), None)
    if step is None:
        raise RuntimeError(f"The re-run left no {name} span")
    trace_id = step["context"]["trace_id"]
    sessions = step_sessions(
        [r for r in rows if r["context"]["trace_id"] == trace_id], name
    )
    if len(sessions) != 1:
        raise RuntimeError(f"Expected one {name} step, found {len(sessions)}")
    session = sessions[0]
    session["framework"] = "hinbox"
    if replay:  # the re-run is no article of its own
        session["metadata"].pop("article_session", None)
    return session


def with_labels(session: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
    """The re-run's session keeping the baseline's eval-case labels (inputs
    ``labels``), which the step never sees, so the evaluator can score it by
    its case's kind."""
    labels = inputs.get("labels")
    if labels:
        session["inputs"] = {**session["inputs"], "labels": labels}
        session["metadata"].update(labels)
    return session


def run_step(
    inputs: Dict[str, Any], replay: bool
) -> Tuple[Dict[str, Any], Optional[BaseException]]:
    """Run the step with tracing on; its session, and what it raised."""
    from src.utils import tracing

    os.environ["HINBOX_TRACING"] = "on"
    model = _model_of(inputs)
    with tempfile.TemporaryDirectory() as tmp:
        tracing.configure_tracing(
            tmp,
            local=model.startswith("ollama/"),
            run_name="kitaru-step",
            domain=str(inputs.get("domain") or ""),
        )
        exporter = _capture()
        fn, args = step_call(inputs)
        error: Optional[BaseException] = None
        try:
            fn(**args)
        except Exception as e:  # recorded, then raised again by main()
            error = e
        rows = [
            json.loads(s.to_json(indent=None)) for s in exporter.get_finished_spans()
        ]
        tracing.shutdown_tracing()
    session = recorded_session(rows, replay, step_name(inputs))
    return with_labels(session, inputs), error


def ingest_request(session: Dict[str, Any], replay: bool) -> Any:
    """The session as Kitaru's ``ImportedSession`` (nodes nested by parent)."""
    from kitaru.task.importer import ImportedNode, ImportedSession

    by_index = {
        n["index"]: {k: v for k, v in n.items() if k not in ("index", "parent_index")}
        for n in session["nodes"]
    }
    roots: List[Dict[str, Any]] = []
    for n in session["nodes"]:
        node = by_index[n["index"]]
        node.setdefault("children", [])
        parent = n["parent_index"]
        if parent is None:
            roots.append(node)
        else:
            by_index[parent].setdefault("children", []).append(node)

    def build(node: Dict[str, Any]) -> Any:
        return ImportedNode(
            **{**node, "children": [build(c) for c in node["children"]]}
        )

    return ImportedSession(
        **{k: v for k, v in session.items() if k != "nodes"},
        nodes=[build(r) for r in roots],
    )


async def _replay_override() -> Any:
    import uuid

    from kitaru.client import KitaruAPIClient

    async with KitaruAPIClient() as client:
        replay = await client.replays.get(uuid.UUID(os.environ["KITARU_REPLAY_ID"]))
    return replay.override


async def _send(parsed: Any, replay: bool) -> Any:
    from kitaru.api_models.v1.session import SessionOrigin
    from kitaru.client import KitaruAPIClient
    from kitaru.task.importer import ingest_session

    origin = SessionOrigin.REPLAY if replay else SessionOrigin.RECORDED
    async with KitaruAPIClient() as client:
        # Agent None: the task's own token says which agent it is. The
        # provider marks the session as built from a trace, as Kitaru's own
        # ImporterBackedAdapter does: the server takes the nodes of a finished
        # session only then
        return await ingest_session(client, parsed, None, "hinbox", origin)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--inputs", help="JSON file of recorded step inputs (outside a worker)"
    )
    ap.add_argument(
        "--out", help="write the session here as Kitaru JSONL, don't send it"
    )
    a = ap.parse_args(argv)
    if a.inputs:
        with open(a.inputs, encoding="utf-8") as f:
            inputs = json.load(f)
    else:
        from kitaru.task import get_task_inputs

        inputs = get_task_inputs()
    if not isinstance(inputs, dict):
        ap.error("no step inputs: run under a Kitaru worker or pass --inputs")
    if not a.out and not os.environ.get("KITARU_TASK_ID"):
        ap.error("outside a Kitaru worker, pass --out")
    replay = bool(os.environ.get("KITARU_REPLAY_ID"))
    if replay:
        override = asyncio.run(_replay_override())
        inputs = apply_override(inputs, override)
        apply_model_params(override)
    session, error = run_step(inputs, replay)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(json.dumps(session, ensure_ascii=False) + "\n")
    else:
        if asyncio.run(_send(ingest_request(session, replay), replay)) is None:
            # ingest_session() reads any 409 as a duplicate; it can also be
            # a task that already has its session, or one no longer running
            raise RuntimeError(
                f"Kitaru refused session {session['external_id']} (409 Conflict)"
            )
    print(f"{session['name']}: {session['status']}", file=sys.stderr)
    if error is not None:
        raise error
    return 0


if __name__ == "__main__":
    sys.exit(main())
