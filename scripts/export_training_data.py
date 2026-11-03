"""Turn trace files into chat-format training examples, one file per task.

Every model call in ``{output_dir}/traces/*.jsonl`` becomes one JSON line
that TRL, Unsloth and Axolotl read as-is::

    {"messages": [system, user, ..., assistant], "tools": [...],
     "metadata": {"task": "match_check", "model": ..., "entity_name": ...,
                  "eval_correct": true, "attempt": 1, "final_attempt": true, ...}}

``metadata`` gathers the ``hinbox.*`` notes from the call's step, article
and eval-item spans (nearest wins), so a training set can be filtered on how
the call ended: only eval items the model got right, only extractions that
passed QC, only the last of Instructor's retries. Calls that raised or hit
the token limit are left out unless ``--keep-failed`` is given.

    uv run python -m scripts.export_training_data data/guantanamo/entities/traces/*.jsonl \\
        --out data/guantanamo/entities/training

``attempt``/``final_attempt`` count Instructor's retries within one request.
An extraction's QC repair retry is a separate request: its calls carry
``extraction_attempt`` (1 or 2), that attempt's own ``attempt_count`` and
``attempt_qc_flags``, and ``attempt_selected`` for the one the pipeline kept;
the step's ``final_count``/``qc_flags`` describe the kept result.

Anthropic tool declarations are converted to the OpenAI ``function`` form;
an Anthropic retry turn (``tool_use``/``tool_result`` content blocks) is kept
as sent. Prompt provenance is ``prompt_files``: every prompt file the step
loaded, with a hash of its text.

The output holds article text: keep it off GitHub (the repo is public).
"""

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional

_MSG = re.compile(r"^llm\.(input|output)_messages\.(\d+)\.message\.(.+)$")
_TOOL_CALL = re.compile(
    r"^tool_calls\.(\d+)\.tool_call\.(id|function\.name|function\.arguments)$"
)
_TOOL = re.compile(r"^llm\.tools\.(\d+)\.tool\.json_schema$")


def load_spans(paths: Iterable[str]) -> List[Dict[str, Any]]:
    spans = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            spans.extend(json.loads(line) for line in f if line.strip())
    return spans


def _json_or_text(value: Any) -> Any:
    try:
        return json.loads(value) if isinstance(value, str) else value
    except ValueError:
        return value


def _messages(attrs: Dict[str, Any], side: str) -> List[Dict[str, Any]]:
    """Rebuild the OpenAI-style message list from flattened span attributes."""
    found: Dict[int, Dict[str, Any]] = defaultdict(dict)
    for key, value in attrs.items():
        m = _MSG.match(key)
        if not m or m.group(1) != side:
            continue
        msg, field = found[int(m.group(2))], m.group(3)
        call = _TOOL_CALL.match(field)
        if call:
            calls = msg.setdefault("_calls", defaultdict(dict))
            calls[int(call.group(1))][call.group(2).removeprefix("function.")] = value
        else:
            msg[field.removeprefix("message.")] = value
    out = []
    for i in sorted(found):
        msg = found[i]
        calls = msg.pop("_calls", {})
        if calls:
            msg["tool_calls"] = [
                {
                    **({"id": c["id"]} if c.get("id") else {}),
                    "type": "function",
                    "function": {
                        "name": c.get("name", ""),
                        "arguments": _json_or_text(c.get("arguments", "")),
                    },
                }
                for _, c in sorted(calls.items())
            ]
        msg.setdefault("content", None)
        out.append(msg)
    return out


def _input_messages(attrs: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The request's messages as sent (``input.value``), else the flattened ones.

    ``input.value`` is the whole request as JSON, so retry turns keep their
    tool-call ids and ``tool_call_id`` links.
    """
    sent = _json_or_text(attrs.get("input.value"))
    if isinstance(sent, dict) and isinstance(sent.get("messages"), list):
        return sent["messages"]
    return _messages(attrs, "input")


def _openai_tool(tool: Any) -> Any:
    """An Anthropic tool declaration (``name``/``input_schema``) in OpenAI form."""
    if isinstance(tool, dict) and "input_schema" in tool and "function" not in tool:
        fn = {"name": tool.get("name", ""), "parameters": tool["input_schema"]}
        if tool.get("description"):
            fn["description"] = tool["description"]
        return {"type": "function", "function": fn}
    return tool


def _tools(attrs: Dict[str, Any]) -> List[Any]:
    tools = {int(m.group(1)): v for k, v in attrs.items() if (m := _TOOL.match(k))}
    return [_openai_tool(_json_or_text(tools[i])) for i in sorted(tools)]


def _context(span: Dict[str, Any], by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """``hinbox.*`` notes from the span and its ancestors; the nearest wins."""
    chain, current = [], span
    while current is not None:
        chain.append(current)
        current = by_id.get(current.get("parent_id") or "")
    meta: Dict[str, Any] = {}
    for s in reversed(chain):  # root first, so nearer spans overwrite
        for key, value in (s.get("attributes") or {}).items():
            if key.startswith("hinbox.") and not key.startswith("hinbox.llm."):
                meta[key.removeprefix("hinbox.")] = value
    return meta


def _is_failed(span: Dict[str, Any]) -> bool:
    attrs = span.get("attributes") or {}
    finish = attrs.get("gen_ai.response.finish_reasons") or []
    errored = (span.get("status") or {}).get("status_code") == "ERROR"
    return errored or "length" in finish or "max_tokens" in finish


def task_of(step: str) -> str:
    """``merge.people/profile.update`` -> ``profile.update``."""
    return step.rsplit("/", 1)[-1] if step else "unknown"


def examples(spans: List[Dict[str, Any]], keep_failed: bool = False) -> List[Dict]:
    """One training example per model call, in the order the calls started."""
    by_id = {s["context"]["span_id"]: s for s in spans}
    calls = [
        s
        for s in spans
        if (s.get("attributes") or {}).get("openinference.span.kind") == "LLM"
    ]
    calls.sort(key=lambda s: s.get("start_time") or "")
    siblings: Dict[Optional[str], List[str]] = defaultdict(list)
    for s in calls:
        siblings[s.get("parent_id")].append(s["context"]["span_id"])

    rows = []
    for s in calls:
        if _is_failed(s) and not keep_failed:
            continue
        attrs = s.get("attributes") or {}
        reply = _messages(attrs, "output")
        if not reply:
            continue
        group = siblings[s.get("parent_id")]
        span_id = s["context"]["span_id"]
        meta = {
            **(s.get("resource") or {}).get("attributes", {}),
            **_context(s, by_id),
            "task": task_of(str(attrs.get("hinbox.step") or "")),
            "model": attrs.get("gen_ai.response.model") or attrs.get("llm.model_name"),
            "invocation_parameters": _json_or_text(
                attrs.get("llm.invocation_parameters")
            ),
            "reasoning": attrs.get("hinbox.llm.reasoning"),
            "finish_reasons": list(attrs.get("gen_ai.response.finish_reasons") or []),
            "failed": _is_failed(s),
            "attempt": group.index(span_id) + 1,
            "final_attempt": group[-1] == span_id,
            "trace_id": s["context"]["trace_id"],
            "span_id": span_id,
        }
        meta = {
            k: v
            for k, v in meta.items()
            if v is not None and not k.startswith(("telemetry.", "service."))
        }
        row: Dict[str, Any] = {
            "messages": _input_messages(attrs) + reply,
            "metadata": meta,
        }
        tools = _tools(attrs)
        if tools:
            row["tools"] = tools
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("traces", nargs="+", help="trace JSONL files")
    parser.add_argument("--out", required=True, help="directory for <task>.jsonl")
    parser.add_argument("--task", action="append", help="only these tasks")
    parser.add_argument(
        "--keep-failed",
        action="store_true",
        help="also keep calls that raised or hit the token limit",
    )
    args = parser.parse_args()

    rows = examples(load_spans(args.traces), keep_failed=args.keep_failed)
    by_task: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        task = row["metadata"]["task"]
        if not args.task or task in args.task:
            by_task[task].append(row)
    os.makedirs(args.out, exist_ok=True)
    for task, task_rows in sorted(by_task.items()):
        path = os.path.join(args.out, f"{task}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for row in task_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{task}: {len(task_rows)} examples -> {path}")


if __name__ == "__main__":
    main()
