"""The citation checker's test set in Kitaru: sessions, labelling, experiments.

The whole loop, each step a ``just citecheck`` command:

1. ``kitaru-sessions``: every test claim is asked of a baseline judge through
   the pipeline's own ``judge_claims()``, one claim per call, run with
   tracing as a recorded step (``scripts/kitaru_step.run_step()``). Each call
   becomes one Kitaru session for the ``hinbox-citation-check`` agent, whose
   inputs are the judge's arguments plus ``labels`` (item id, stratum, the
   hand label when there is one yet). The file is imported with
   ``kitaru session import``.
2. ``kitaru-investigate``: one Kitaru investigation over those sessions with
   one yes/no question each, the claim highlighted in the inputs, so the
   labelling can happen in Kitaru's own investigation view instead of (or
   beside) the local labelling page.
3. ``kitaru-pull``: the investigation's answers become label rows (the same
   file the labelling page writes) and are written into each session's
   ``labels.gold`` and metadata, so a replay and its evaluator see them.
4. ``kitaru-experiment``: a cohort of those sessions, an experiment whose
   override is another judge (Haiku, Opus, a Tinker checkpoint, a local
   model), the ``hinbox-citation-check-checks`` evaluator (quality, time,
   tokens, cost), a run, and its aggregates side by side with the baseline.

One claim per session keeps Kitaru's means equal to the per-claim shares
(``src/citecheck/kitaru_eval.py``). The Kitaru calls are thin; what they
send is built by the plain functions here, which the tests check.
"""

import asyncio
import json
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.citecheck.items import LABELS, append_jsonl, label_row, read_jsonl
from src.utils.grounding import GROUNDING_MAX_ARTICLE_CHARS

AGENT = "hinbox-citation-check"
EVALUATOR = "hinbox-citation-check-checks"
QUESTION_KEY = "supported"
QUESTION = (
    "Does the article state everything in this sentence about {entity} "
    "(tone aside)? Answer yes, no or unsure.\n\n{claim}"
)


def step_inputs(
    item: Dict[str, Any], text: str, model: str, gold: Optional[str] = None
) -> Dict[str, Any]:
    """``judge_claims()``'s arguments for one test claim, with its labels."""
    return {
        "article_id": item["article_id"],
        "article_text": text[:GROUNDING_MAX_ARTICLE_CHARS],
        "claims": [[item["entity_name"], item["claim"]]],
        "model_type": "local" if model.startswith("ollama/") else "cloud",
        "ask_passage": False,
        "model": model,
        "labels": {
            "item_ids": [item["item_id"]],
            "gold": [gold],
            "strata": [item.get("stratum")],
            "entity_table": item.get("entity_table"),
        },
    }


def shard(items: List[Dict[str, Any]], part: str) -> List[Dict[str, Any]]:
    """Part ``"i/N"`` of ``items``: every N-th item from the i-th (counting
    from 0), so N processes, each writing its own file, cover the items once.
    One process at a time per file, because ``run_step()`` sets up tracing
    for the whole process; several processes run the judge side by side."""
    i, n = (int(x) for x in part.split("/"))
    if not 0 <= i < n:
        raise ValueError(f"part {part!r}: need 0 <= i < N")
    return items[i::n]


def build_sessions(
    items: List[Dict[str, Any]],
    texts: Dict[str, str],
    labels: Dict[str, Dict[str, Any]],
    model: str,
    out_path: str,
    run: Any = None,
) -> Dict[str, int]:
    """Append one session per test claim to ``out_path`` (Kitaru JSONL);
    claims already there are skipped, so a stopped run resumes. ``run`` is
    ``scripts.kitaru_step.run_step`` (passed in by tests)."""
    if run is None:
        from scripts.kitaru_step import run_step as run
    done = {
        iid
        for row in read_jsonl(out_path)
        for iid in (row.get("metadata") or {}).get("item_ids") or []
    }
    counts = {"written": 0, "skipped": 0, "failed": 0, "no_text": 0}
    for item in items:
        if item["item_id"] in done:
            counts["skipped"] += 1
            continue
        text = texts.get(item["article_id"])
        if not text:
            counts["no_text"] += 1
            continue
        gold = (labels.get(item["item_id"]) or {}).get("label")
        session, error = run(step_inputs(item, text, model, gold), False)
        session["name"] = (
            f"citation check: {item['entity_name']} / {item['article_id']}"
        )
        append_jsonl(out_path, session)
        counts["failed" if error else "written"] += 1
    return counts


def investigation_request(
    agent_id: uuid.UUID, name: str, sessions: Sequence[Tuple[uuid.UUID, Dict[str, Any]]]
) -> Any:
    """The investigation: per (Kitaru session id, session inputs), one
    question with the claim highlighted (``/inputs/claims/0/1`` points into
    the session)."""
    from kitaru.api_models.v1.annotation import AnnotationSelector
    from kitaru.api_models.v1.investigation import (
        InvestigationCreateRequest,
        InvestigationSessionHighlight,
        InvestigationSessionInput,
        InvestigationSessionQuestion,
    )

    rows = []
    for sid, inputs in sessions:
        entity, claim = inputs["claims"][0]
        rows.append(
            InvestigationSessionInput(
                session_id=sid,
                questions=[
                    InvestigationSessionQuestion(
                        key=QUESTION_KEY,
                        question=QUESTION.format(entity=entity, claim=claim),
                        highlights=[
                            InvestigationSessionHighlight(
                                selector=AnnotationSelector(path="/inputs/claims/0/1"),
                                description="The sentence to check",
                            ),
                            InvestigationSessionHighlight(
                                selector=AnnotationSelector(
                                    path="/inputs/article_text"
                                ),
                                description="The article it cites",
                            ),
                        ],
                    )
                ],
            )
        )
    return InvestigationCreateRequest(
        agent_id=agent_id,
        name=name,
        description="Hand labels for the citation checker's test set: does the "
        "article state everything in the sentence?",
        sessions=rows,
    )


def answer_label(value: Any) -> Optional[str]:
    """'yes'/'no'/'unsure' from an annotation value (text, bool, or an
    object holding one), None when it isn't one of those."""
    if isinstance(value, dict):
        value = value.get("value", value.get("answer", value.get("label")))
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, str):
        word = value.strip().lower().rstrip(".")
        word = {"y": "yes", "n": "no", "u": "unsure", "true": "yes", "false": "no"}.get(
            word, word
        )
        return word if word in LABELS else None
    return None


def with_gold(inputs: Dict[str, Any], gold: List[Optional[str]]) -> Dict[str, Any]:
    labels = dict(inputs.get("labels") or {})
    labels["gold"] = gold
    return {**inputs, "labels": labels}


def aggregate_table(aggregates: List[Any]) -> List[Dict[str, Any]]:
    """Baseline against replay mean per evaluation, as plain rows."""
    rows = []
    for a in aggregates:
        rows.append(
            {
                "name": a.name,
                "baseline_mean": a.baseline.mean,
                "baseline_count": a.baseline.count,
                "result_mean": a.result.mean,
                "result_count": a.result.count,
            }
        )
    return sorted(rows, key=lambda r: r["name"])


def describe_table(rows: List[Dict[str, Any]], model: str) -> str:
    def f(v: Any) -> str:
        return "n/a" if v is None else f"{v:.3f}"

    lines = [f"{'evaluation':<16} {'baseline':>10} {model[:28]:>28}"]
    for r in rows:
        lines.append(
            f"{r['name']:<16} {f(r['baseline_mean']):>10} {f(r['result_mean']):>28}"
            f"   (n={r['baseline_count']}/{r['result_count']})"
        )
    return "\n".join(lines)


# ---- Kitaru calls -------------------------------------------------------


async def _agent(client: Any, name: str) -> Any:
    async for agent in client.agents.iter():
        if agent.name == name:
            return agent
    raise RuntimeError(f"No Kitaru agent named {name!r}: register it first")


async def _agent_sessions(client: Any, agent_id: uuid.UUID) -> List[Any]:
    from kitaru.api_models.v1.session import SessionListParams

    params = SessionListParams(
        filter={"field": "agent_id", "op": "eq", "value": str(agent_id)}
    )
    return [s async for s in client.sessions.iter(params) if s.agent_id == agent_id]


async def _labelled_sessions(
    client: Any, agent_id: uuid.UUID, item_ids: Optional[set] = None
) -> List[Tuple[Any, Dict[str, Any]]]:
    """(session, its inputs) for the agent's recorded citation sessions
    (replays left out), optionally only those for ``item_ids``."""
    out = []
    for s in await _agent_sessions(client, agent_id):
        ids = (s.metadata or {}).get("item_ids") or []
        if str(getattr(s.origin, "value", s.origin)) == "replay" or not ids:
            continue
        if item_ids is not None and not set(ids) & item_ids:
            continue
        detail = await client.sessions.get(s.id)
        out.append((s, detail.inputs or {}))
    return out


async def investigate(name: str, item_ids: Optional[set] = None, agent: str = AGENT):
    from kitaru.client import KitaruAPIClient

    async with KitaruAPIClient() as client:
        a = await _agent(client, agent)
        sessions = await _labelled_sessions(client, a.id, item_ids)
        request = investigation_request(a.id, name, [(s.id, i) for s, i in sessions])
        return await client.investigations.create(request), len(sessions)


async def pull(
    investigation_id: str, labels_path: str, who: str = "kitaru"
) -> Dict[str, int]:
    """Write the investigation's answers to ``labels_path`` and into each
    session's ``labels.gold`` and metadata."""
    from kitaru.api_models.v1.annotation import AnnotationListParams
    from kitaru.api_models.v1.session import SessionUpdateRequest
    from kitaru.client import KitaruAPIClient

    counts = {"answered": 0, "unreadable": 0, "unanswered": 0}
    async with KitaruAPIClient() as client:
        iid = uuid.UUID(investigation_id)
        async for inv_s in client.investigations.iter_sessions(iid):
            params = AnnotationListParams(
                filter={
                    "field": "investigation_session_id",
                    "op": "eq",
                    "value": str(inv_s.id),
                }
            )
            answers = [
                a
                async for a in client.annotations.iter(params)
                if a.investigation_session_id == inv_s.id
                and a.question_key == QUESTION_KEY
            ]
            if not answers:
                counts["unanswered"] += 1
                continue
            label = answer_label(answers[-1].value)
            if label is None:
                counts["unreadable"] += 1
                continue
            detail = await client.sessions.get(inv_s.session_id)
            inputs = detail.inputs or {}
            ids = (inputs.get("labels") or {}).get("item_ids") or []
            for item in ids:
                append_jsonl(labels_path, label_row(item, label, "kitaru", who))
            gold = [label] * len(ids)
            meta = {**(detail.metadata or {}), "gold": gold}
            await client.sessions.update(
                inv_s.session_id,
                SessionUpdateRequest(inputs=with_gold(inputs, gold), metadata=meta),
            )
            counts["answered"] += 1
    return counts


async def experiment(
    name: str,
    model: str,
    prices: Optional[Dict[str, List[float]]] = None,
    item_ids: Optional[set] = None,
    agent: str = AGENT,
    wait: bool = True,
    poll_seconds: float = 20.0,
) -> Dict[str, Any]:
    """Cohort of the labelled sessions, experiment with ``model`` as the
    override, one run; with ``wait`` its aggregates once it ends."""
    from kitaru.api_models.v1.cohort import CohortCreateRequest
    from kitaru.api_models.v1.cohort_version import CohortVersionCreateRequest
    from kitaru.api_models.v1.experiment import ExperimentCreateRequest
    from kitaru.api_models.v1.experiment_run import ExperimentRunCreateRequest
    from kitaru.api_models.v1.plugin import EvaluatorConfig
    from kitaru.api_models.v1.replay_config import ReplayOverride
    from kitaru.client import KitaruAPIClient

    async with KitaruAPIClient() as client:
        a = await _agent(client, agent)
        sessions = await _labelled_sessions(client, a.id, item_ids)
        cohort = await client.cohorts.create(
            CohortCreateRequest(name=f"{name} cohort", agent_id=a.id)
        )
        version = await client.cohorts.create_version(
            cohort.id,
            CohortVersionCreateRequest(add_session_ids=[s.id for s, _ in sessions]),
        )
        exp = await client.experiments.create(
            ExperimentCreateRequest(
                name=name,
                agent_id=a.id,
                override=ReplayOverride(model=model),
                evaluators=[
                    EvaluatorConfig(
                        evaluator=EVALUATOR, params={"prices": prices or {}}
                    )
                ],
            )
        )
        versions = (await client.agents.list_versions(a.id)).items
        latest = max(versions, key=lambda v: v.version)
        run = await client.experiments.start_run(
            exp.id,
            ExperimentRunCreateRequest(
                cohort_version_id=version.id,
                agent_version_id=latest.id,
                evaluate_baselines=True,
            ),
        )
        out: Dict[str, Any] = {
            "experiment_id": str(exp.id),
            "run_id": str(run.id),
            "sessions": len(sessions),
            "model": model,
        }
        if not wait:
            return out
        started = time.monotonic()
        while str(getattr(run.status, "value", run.status)) in ("running", "canceling"):
            await asyncio.sleep(poll_seconds)
            run = await client.experiment_runs.get(run.id)
        out["status"] = str(getattr(run.status, "value", run.status))
        out["minutes"] = round((time.monotonic() - started) / 60, 1)
        out["table"] = aggregate_table(
            await client.experiment_runs.get_evaluation_aggregates(run.id)
        )
        return out


def save_json(path: str, value: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, default=str)
