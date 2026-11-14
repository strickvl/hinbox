"""Turn Hinbox trace files into Kitaru sessions, in parts under its upload limit.

Each trace (one article, one eval item, the grounding pass) becomes one
session line in Kitaru's own JSONL format, which the built-in
``kitaru/kitaru-jsonl`` importer reads. We write that format rather than
letting ``kitaru/phoenix`` read the raw spans, because Kitaru's viewer only
shows a model call as a conversation when its messages are ``{"role",
"content"}`` and the node says where the system prompt, the prompt and the
reply are (``*_selector``); the Phoenix importer sets neither
(https://github.com/zenml-io/kitaru/issues/1517). A step span carries no
``input.value``, only our ``hinbox.*`` labels, so those become its inputs and
outputs here, and merge decisions join the outputs of their ``merge.<type>``
step.

    uv run python -m scripts.kitaru_traces data/guantanamo/entities/traces/run-*.jsonl \\
        --out data/guantanamo/entities/kitaru

then, for each part:

    kitaru session import <part> --importer kitaru/kitaru-jsonl@latest \\
        --agent hinbox@latest --media-type application/x-ndjson --wait

The parts hold article text: keep them off GitHub (the repo is public).
"""

import argparse
import json
import os
import re
from contextlib import ExitStack
from typing import Any, Dict, Iterable, List, Optional, Tuple

# A server's upload cap is ``max_blob_size_bytes`` in ``kitaru status``: 100 MiB
# by default, 50 MiB on a managed cloud workspace; stay under both.
DEFAULT_MAX_BYTES = 45 * 1024 * 1024

# ``hinbox.*`` labels that say what went into a step; every other label is
# about how it ended
INPUT_LABELS = {
    "article_id",
    "entity_name",
    "entity_type",
    "new_name",
    "existing_name",
    "approach",
    "chars_before",
    "profile_chars_shown",
    "prompt_files",
    "extraction_attempt",
    "attempt_count",
    "article.id",
    "article.title",
    "article.url",
    "article.published_date",
    "article.source_type",
    "article.chars",
}
UNFINISHED = (
    "The run stopped before this trace finished, so its top span was never written."
)
_MESSAGE_KEY = re.compile(r"llm\.(input|output)_messages\.(\d+)\.message\.(.+)$")
_TOOL_CALL_KEY = re.compile(
    r"tool_calls\.(\d+)\.tool_call\.(id|function\.name|function\.arguments)$"
)


def _decode(value: Any) -> Any:
    """A JSON string as the value it holds; anything else unchanged."""
    if isinstance(value, str) and value[:1] in "[{":
        try:
            return json.loads(value)
        except ValueError:
            pass
    return value


def messages(attrs: Dict[str, Any], direction: str) -> List[Dict[str, Any]]:
    """A model call's OpenInference messages as ``{"role", "content", ...}`` dicts."""
    found: Dict[int, Dict[str, Any]] = {}
    for key, value in attrs.items():
        m = _MESSAGE_KEY.match(key)
        if not m or m.group(1) != direction:
            continue
        msg = found.setdefault(int(m.group(2)), {})
        call = _TOOL_CALL_KEY.match(m.group(3))
        if call:
            calls = msg.setdefault("tool_calls", {})
            entry = calls.setdefault(int(call.group(1)), {"function": {}})
            field = call.group(2)
            if field == "id":
                entry["id"] = value
            else:
                entry["function"][field.split(".")[1]] = value
        else:
            msg[m.group(3)] = value
    out = []
    for i in sorted(found):
        msg = found[i]
        if "tool_calls" in msg:
            msg["tool_calls"] = [
                msg["tool_calls"][j] for j in sorted(msg["tool_calls"])
            ]
        out.append(msg)
    return out


def _labels(attrs: Dict[str, Any]) -> Dict[str, Any]:
    return {
        k[len("hinbox.") :]: v
        for k, v in attrs.items()
        if k.startswith("hinbox.")
        and k not in {"hinbox.step", "hinbox.system_prompt_sha"}
    }


def _error(row: Dict[str, Any]) -> Optional[str]:
    status = row.get("status") or {}
    if status.get("status_code") != "ERROR":
        return None
    if status.get("description"):
        return status["description"]
    for event in row.get("events") or []:
        message = (event.get("attributes") or {}).get("exception.message")
        if message:
            return message
    return "failed"


def _llm_fields(attrs: Dict[str, Any]) -> Dict[str, Any]:
    """Inputs, outputs, selectors and model fields of a model-call span."""
    ins, outs = messages(attrs, "input"), messages(attrs, "output")
    users = [i for i, m in enumerate(ins) if m.get("role") == "user"]
    reply = outs[0] if outs else {}
    if reply.get("content") is not None:
        output_selector: Optional[str] = "/0/content"
    elif reply.get("tool_calls"):  # structured output sent as a tool call
        output_selector = "/0/tool_calls/0/function/arguments"
    else:
        output_selector = None
    reasoning = []
    if attrs.get("hinbox.llm.reasoning"):
        outs.append({"role": "reasoning", "content": attrs["hinbox.llm.reasoning"]})
        reasoning = [f"/{len(outs) - 1}/content"]
    params = _decode(attrs.get("llm.invocation_parameters"))
    if isinstance(params, dict):
        params.pop("response_format", None)  # the whole schema; kept in attributes
    else:
        params = None
    return {
        "node_type": "llm_call",
        "inputs": ins,
        "outputs": outs,
        "system_prompt_selector": "/0/content"
        if ins and ins[0].get("role") == "system" and "content" in ins[0]
        else None,
        "input_text_selector": f"/{users[-1]}/content"
        if users and "content" in ins[users[-1]]
        else None,
        "output_text_selector": output_selector,
        "reasoning_selectors": reasoning,
        "requested_model": attrs.get("gen_ai.request.model")
        or attrs.get("llm.model_name"),
        "model": attrs.get("gen_ai.response.model") or attrs.get("llm.model_name"),
        "model_provider": attrs.get("llm.provider"),
        "model_params": params,
        "tokens": {
            "input_tokens": attrs.get("llm.token_count.prompt"),
            "output_tokens": attrs.get("llm.token_count.completion"),
        },
    }


def _span_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    """Inputs and outputs of a step span, from its labels and merge decisions."""
    attrs = row.get("attributes") or {}
    labels = _labels(attrs)
    inputs: Any = {k: v for k, v in labels.items() if k in INPUT_LABELS}
    outputs: Any = {k: v for k, v in labels.items() if k not in INPUT_LABELS}
    decisions = [
        e["attributes"]
        for e in row.get("events") or []
        if "kind" in (e.get("attributes") or {})
    ]
    if decisions:
        outputs["decisions"] = decisions
    return {
        "node_type": "span",
        "inputs": inputs or _decode(attrs.get("input.value")),
        "outputs": outputs or _decode(attrs.get("output.value")),
    }


def node(
    row: Dict[str, Any], index: int, parent_index: Optional[int]
) -> Dict[str, Any]:
    """One span as a Kitaru node."""
    attrs = row.get("attributes") or {}
    is_llm = attrs.get("openinference.span.kind") == "LLM"
    error = _error(row)
    kept = {  # the messages and values are already in inputs/outputs
        k: v
        for k, v in attrs.items()
        if not _MESSAGE_KEY.match(k) and k not in {"input.value", "output.value"}
    }
    return {
        "index": index,
        "parent_index": parent_index,
        "links": [],
        "external_id": row["context"]["span_id"],
        "trace_id": row["context"]["trace_id"],
        "name": row["name"],
        "status": "failed" if error else "completed",
        "error": error,
        "started_at": row.get("start_time"),
        "ended_at": row.get("end_time"),
        **(_llm_fields(attrs) if is_llm else _span_fields(row)),
        "attributes": kept,
        "metadata": {},
    }


def session(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """One trace's spans as a Kitaru session; its top span gives the name and status.

    A trace whose top span is missing (the run stopped while the article was
    between extraction and merging) is kept as a failed session, so its
    model calls still show.
    """
    rows = sorted(rows, key=lambda r: r.get("start_time") or "")
    position = {r["context"]["span_id"]: i for i, r in enumerate(rows)}
    nodes = [node(r, i, position.get(r.get("parent_id"))) for i, r in enumerate(rows)]
    roots = [n for n, r in zip(nodes, rows) if not r.get("parent_id")]
    resource = (rows[0].get("resource") or {}).get("attributes") or {}
    metadata = {
        k[len("hinbox.") :]: v
        for k, v in resource.items()
        if k.startswith("hinbox.") and not k.startswith("hinbox.env.")
    }
    metadata["complete"] = bool(roots)
    if roots:
        root = roots[0]
        ended = root["outputs"] if isinstance(root["outputs"], dict) else {}
        outcome = str(ended.get("article.outcome", ""))
        error = root["error"] or (
            outcome
            if outcome.startswith(("incomplete", "Relevance check failed"))
            else None
        )
        name, inputs, outputs = root["name"], root["inputs"], root["outputs"]
    else:
        error, name = UNFINISHED, f"unfinished ({nodes[0]['name']})"
        inputs = outputs = None
    return {
        "status": "failed" if error else "completed",
        "name": name,
        "inputs": inputs,
        "outputs": outputs,
        "error": error,
        "started_at": rows[0].get("start_time"),
        "ended_at": max((r.get("end_time") or "" for r in rows), default=None) or None,
        "external_id": f"hinbox:{rows[0]['context']['trace_id']}",
        "metadata": metadata,
        "framework": None,
        "nodes": nodes,
    }


def index_traces(paths: Iterable[str]) -> Dict[str, List[Tuple[str, int, int]]]:
    """Each trace's lines as (file, offset, length), traces in order of first appearance.

    A run writes the spans of several articles interleaved, so one trace's
    lines are not contiguous.
    """
    traces: Dict[str, List[Tuple[str, int, int]]] = {}
    for path in paths:
        with open(path, "rb") as f:
            offset = 0
            for raw in f:
                if raw.strip():
                    trace_id = json.loads(raw)["context"]["trace_id"]
                    traces.setdefault(trace_id, []).append((path, offset, len(raw)))
                offset += len(raw)
    return traces


def write_parts(paths: List[str], out_dir: str, max_bytes: int) -> List[str]:
    """Write one session line per trace into ``part-NNN.jsonl`` files of at most
    ``max_bytes`` (a bigger session gets a part of its own); returns their paths."""
    traces = index_traces(paths)
    os.makedirs(out_dir, exist_ok=True)
    written: List[str] = []
    out, size = None, 0
    with ExitStack() as stack:
        files = {p: stack.enter_context(open(p, "rb")) for p in set(paths)}
        for lines in traces.values():
            rows = []
            for path, offset, length in lines:
                files[path].seek(offset)
                rows.append(json.loads(files[path].read(length)))
            line = (json.dumps(session(rows)) + "\n").encode()
            if out is None or (size and size + len(line) > max_bytes):
                if out is not None:
                    out.close()
                written.append(
                    os.path.join(out_dir, f"part-{len(written) + 1:03d}.jsonl")
                )
                out, size = open(written[-1], "wb"), 0
            out.write(line)
            size += len(line)
        if out is not None:
            out.close()
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("traces", nargs="+", help="trace JSONL files")
    parser.add_argument("--out", required=True, help="folder for the parts")
    parser.add_argument(
        "--max-mb",
        type=float,
        default=DEFAULT_MAX_BYTES / 1024 / 1024,
        help="largest part in MiB (default %(default)s); a bigger session gets a part of its own",
    )
    args = parser.parse_args()
    max_bytes = int(args.max_mb * 1024 * 1024)
    for part in write_parts(args.traces, args.out, max_bytes):
        size = os.path.getsize(part)
        note = "  (one session, over --max-mb)" if size > max_bytes else ""
        print(f"{part}  {size / 1024 / 1024:.1f} MiB{note}")


if __name__ == "__main__":
    main()
