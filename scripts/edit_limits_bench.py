"""Replay a finished local run's edit replies with the length limits on.

    uv run python -m scripts.edit_limits_bench <run>/traces/run-*.jsonl \
        --out <dir> --model ollama/<name> [--runaway 30] [--long 30] [--workers 2]

Reads the run's traces (no model call), finds every profile-edit request, and
picks two groups of distinct prompts:

- ``runaway``: a reply to it ran to the token limit;
- ``long``: its reply ended normally but has a value at or over a limit of
  ``src/engine/edit_limits.py``, so the limits would cut it.

Each picked prompt is sent once more, exactly as recorded (same messages and
settings), with only the reply schema swapped for the limited one. For the
runaway group the question is whether the reply now ends and what is left of
it; for the long group, how many of the old reply's cited sentences the new
one still has. ``--limits off`` sends the plain schema (a control: at
temperature 0 the old reply should come back).

Writes ``results.jsonl`` (one row per prompt), ``review.md`` (the new edits,
for reading) and prints the counts. Both files hold article-derived text:
keep them off GitHub.
"""

import argparse
import hashlib
import json
import os
import random
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterable, List, Optional, Tuple

from scripts.export_training_data import (
    _input_messages,
    _json_or_text,
    _without_retry_turns,
)
from scripts.export_training_data import _messages as _span_messages
from src.constants import OLLAMA_MODEL
from src.engine import edit_limits as el
from src.engine.edit_guards import _claim as claim_of
from src.engine.edit_guards import guard_edits
from src.engine.edit_limits import LimitedEditList, without_cut_off
from src.engine.profile_edits import ProfileEditList, apply_edits
from src.engine.profile_notes import is_profile_note
from src.evals.profile_cases import _sentences as split_sentences

SCHEMA_NAMES = ("ProfileEditList", "LimitedEditList")
_NOTE_KEYS = ("hinbox.entity_name", "hinbox.article_id")
_CITATION = re.compile(r"\^\[[^\]]*\]")
_PROFILE = re.compile(
    r"Existing profile:\n-{16}\n(.*)\n\nNew article \(ID: ", re.DOTALL
)


def edit_call(span: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The request and reply of a profile-edit span, or None for another span."""
    attrs = span.get("attributes") or {}
    params = _json_or_text(attrs.get("llm.invocation_parameters"))
    if not isinstance(params, dict):
        return None
    fmt = params.get("response_format") or {}
    if (fmt.get("json_schema") or {}).get("name") not in SCHEMA_NAMES:
        return None
    # The prompt as first sent: a retry turn would hold a whole cut reply
    messages = _without_retry_turns(_input_messages(attrs))
    reply = _span_messages(attrs, "output")
    return {
        "key": hashlib.sha256(
            json.dumps(messages, sort_keys=True).encode()
        ).hexdigest()[:16],
        "messages": messages,
        "params": params,
        "finish": list(attrs.get("gen_ai.response.finish_reasons") or []),
        "reply": (reply[0].get("content") if reply else "") or "",
        "step": attrs.get("hinbox.step"),
    }


def _group(call: Dict[str, Any]) -> Optional[str]:
    if "length" in call["finish"]:
        return "runaway"
    old = parse_reply(call["reply"])
    return "long" if old is not None and over_limits(old) else None


def scan(paths: Iterable[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """(edit calls in a group, notes) from trace files, one line at a time.

    A full run's traces are too big to hold in memory, so a call keeps only
    its prompt's key, its group and where its line is; ``notes`` has each
    span's parent and ``hinbox.*`` values, to name the entity and article."""
    calls: List[Dict[str, Any]] = []
    notes: Dict[str, Any] = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for number, line in enumerate(f):
                if not line.strip():
                    continue
                span = json.loads(line)
                attrs = span.get("attributes") or {}
                kept = {k: attrs[k] for k in _NOTE_KEYS if k in attrs}
                span_id = span["context"]["span_id"]
                notes[span_id] = (span.get("parent_id"), kept)
                call = edit_call(span)
                group = _group(call) if call else None
                if call and group:
                    where = (path, number)
                    calls.append(
                        {
                            "key": call["key"],
                            "group": group,
                            "span": span_id,
                            "where": where,
                        }  # fmt: skip
                    )
    return calls, notes


def _note(notes: Dict[str, Any], span_id: Optional[str], key: str) -> Any:
    """``key`` from the span or its nearest ancestor that has it."""
    while span_id in notes:
        span_id, kept = notes[span_id]
        if key in kept:
            return kept[key]
    return None


def load(picked: List[Dict[str, Any]], notes: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The picked calls in full, read again from their lines."""
    wanted = {c["where"]: c for c in picked}
    full: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for path in sorted({where[0] for where in wanted}):
        with open(path, encoding="utf-8") as f:
            for number, line in enumerate(f):
                slim = wanted.get((path, number))
                if slim is None:
                    continue
                full[(path, number)] = {
                    **(edit_call(json.loads(line)) or {}),
                    "group": slim["group"],
                    "of": slim["of"],
                    "entity": _note(notes, slim["span"], "hinbox.entity_name"),
                    "article": _note(notes, slim["span"], "hinbox.article_id"),
                }
    return [full[c["where"]] for c in picked]


def parse_reply(text: str) -> Optional[ProfileEditList]:
    try:
        return ProfileEditList.model_validate_json(text)
    except ValueError:
        return None


def over_limits(reply: ProfileEditList) -> bool:
    """Would the limits cut this reply?"""
    return (
        len(reply.disagreements) > el.MAX_DISAGREEMENTS
        or len(reply.new_facts) > el.MAX_NEW_FACTS
        or len(reply.edits) > el.MAX_EDITS
        or any(len(f) >= el.NEW_FACT_CHARS for f in reply.new_facts)
        or any(
            len(d.profile_sentence) >= el.PROFILE_SENTENCE_CHARS
            or len(d.article_says) >= el.ARTICLE_SAYS_CHARS
            for d in reply.disagreements
        )
        or any(
            len(e.anchor) >= el.ANCHOR_CHARS or len(e.text) >= el.TEXT_CHARS
            for e in reply.edits
        )
    )


def pick(
    calls: List[Dict[str, Any]], runaway: int, long: int, seed: int = 0
) -> List[Dict[str, Any]]:
    """Up to ``runaway`` and ``long`` distinct prompts, the same for a seed."""
    picked = []
    for group, size in (("runaway", runaway), ("long", long)):
        first: Dict[str, Dict[str, Any]] = {}
        for call in calls:
            if call["group"] == group:
                first.setdefault(call["key"], call)
        found = [first[k] for k in sorted(first)]
        chosen = random.Random(seed).sample(found, min(size, len(found)))
        picked += [{**c, "of": len(found)} for c in chosen]
    return picked


def _sentences(reply: Optional[ProfileEditList]) -> List[str]:
    """The cited pieces of a reply's edit texts, spaces collapsed."""
    if reply is None:
        return []
    pieces = [p for e in reply.edits for p in _CITATION.split(e.text)]
    return [s for s in (" ".join(p.split()).strip(" .") for p in pieces) if s]


def _applied(profile: str, edits: List[Any]) -> Dict[str, int]:
    report = apply_edits(profile, edits)[1]
    return {
        "applied": report["applied"],
        "appended": report["appended"],
        "skipped": len(report["skipped"]),
    }


def _notes(edits: List[Any]) -> int:
    """New sentences that talk about the profile (``profile_notes.py``)."""
    return sum(
        is_profile_note(claim_of(s))
        for e in edits
        if e.op == "add"
        for s in split_sentences(e.text)
    )


def compare(call: Dict[str, Any], new_text: str) -> Dict[str, Any]:
    """What the limits left of a reply, beside the recorded one."""
    old, new = parse_reply(call["reply"]), parse_reply(new_text)
    row: Dict[str, Any] = {"parsed": new is not None, "old_parsed": old is not None}
    row["reply_text"] = new_text
    if new is None:
        return row
    edits, cut = without_cut_off(new)
    found = _PROFILE.search(call["messages"][-1]["content"])
    profile = found.group(1) if found else ""
    old_sentences, new_sentences = _sentences(old), set(_sentences(new))
    # What the pipeline's code guards then leave (no model check, no dates)
    guarded = guard_edits(profile, edits)[0]
    return {
        **row,
        "cut": cut,
        "edits_after_guards": len(guarded),
        "notes": _notes(edits),
        "notes_after_guards": _notes(guarded),
        "edits": len(new.edits),
        "edits_kept": len(edits),
        "edits_dropped": len(new.edits) - len(edits),
        "placed": _applied(profile, edits),
        "old_edits": len(old.edits) if old else None,
        "old_sentences": len(old_sentences),
        "old_sentences_kept": sum(s in new_sentences for s in old_sentences),
        "new_edits": [e.model_dump() for e in edits],
    }


def replay(call: Dict[str, Any], model: str, limits: bool) -> Dict[str, Any]:
    from src.utils.llm import _create_raw_client
    from src.utils.provider_routing import resolve_chat_target

    target = resolve_chat_target(model)
    schema = LimitedEditList if limits else ProfileEditList
    params = {
        **call["params"],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "schema": schema.model_json_schema(),
            },
        },
    }
    row = {k: call[k] for k in ("key", "group", "step", "entity", "article")}
    row["old_finish"] = call["finish"]
    started = time.monotonic()
    try:
        response = _create_raw_client(target).chat.completions.create(
            model=target.api_model, messages=call["messages"], **params
        )
    except Exception as e:  # a refused schema must show, not stop the bench
        return {**row, "error": f"{type(e).__name__}: {e}"[:500]}
    choice = response.choices[0]
    return {
        **row,
        "seconds": round(time.monotonic() - started, 1),
        "finish": choice.finish_reason,
        "tokens": getattr(response.usage, "completion_tokens", None),
        **compare(call, choice.message.content or ""),
    }


def summary(rows: List[Dict[str, Any]]) -> List[str]:
    lines = []
    for group in ("runaway", "long"):
        got = [r for r in rows if r["group"] == group]
        if not got:
            continue
        ok = [r for r in got if r.get("finish") == "stop" and r.get("parsed")]
        cut = [r for r in ok if r.get("cut")]
        secs = [r["seconds"] for r in got if "seconds" in r]
        lines += [
            f"{group}: {len(got)} prompts replayed",
            f"  ended normally and readable: {len(ok)}",
            f"  still ran to the token limit: "
            f"{sum(r.get('finish') == 'length' for r in got)}",
            f"  request refused or failed: {sum('error' in r for r in got)}",
            f"  replies with a value cut at a limit: {len(cut)}",
            f"  edits kept / dropped as cut: "
            f"{sum(r['edits_kept'] for r in ok)} / "
            f"{sum(r['edits_dropped'] for r in ok)}",
            "  kept edits placed / put at the end / skipped: "
            + " / ".join(
                str(sum(r["placed"][k] for r in ok))
                for k in ("applied", "appended", "skipped")
            ),
            f"  replies with no edit left: {sum(not r['edits_kept'] for r in ok)}",
            f"  edits left by the code guards: "
            f"{sum(r['edits_after_guards'] for r in ok)}",
            f"  new sentences about the profile, before / after the guards: "
            f"{sum(r['notes'] for r in ok)} / "
            f"{sum(r['notes_after_guards'] for r in ok)}",
        ]
        if secs:
            lines.append(
                f"  seconds per reply: median {statistics.median(secs):.0f}, "
                f"longest {max(secs):.0f}"
            )
        if group == "long":
            lines.append(
                f"  the old reply's cited sentences still there: "
                f"{sum(r['old_sentences_kept'] for r in ok)} of "
                f"{sum(r['old_sentences'] for r in ok)}; replies that lost one: "
                f"{sum(r['old_sentences_kept'] < r['old_sentences'] for r in ok)}"
            )
    return lines


def review(rows: List[Dict[str, Any]]) -> str:
    out = ["# Edit replies written under the length limits\n"]
    for r in rows:
        out.append(f"## {r['group']}: {r.get('entity')} / article {r.get('article')}\n")
        out.append(
            f"finish {r.get('finish')}, {r.get('tokens')} tokens, cut: "
            f"{json.dumps(r.get('cut'))}, error: {r.get('error')}\n"
        )
        for e in r.get("new_edits") or []:
            out.append(
                f"- **{e['op']}** after `{e['anchor'][:120]}`\n\n  {e['text']}\n"
            )
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("traces", nargs="+", help="trace JSONL files of a local run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=OLLAMA_MODEL)
    ap.add_argument("--runaway", type=int, default=30, help="prompts to replay")
    ap.add_argument("--long", type=int, default=30, help="prompts to replay")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1, help="requests at once")
    ap.add_argument("--limits", choices=("on", "off"), default="on")
    ap.add_argument("--list", action="store_true", help="count the groups, no calls")
    args = ap.parse_args()

    calls, notes = scan(args.traces)
    for group in ("runaway", "long"):
        keys = {c["key"] for c in calls if c["group"] == group}
        print(f"{group}: {len(keys)} distinct prompts")
    if args.list:
        return 0
    picked = load(pick(calls, args.runaway, args.long, args.seed), notes)
    os.makedirs(args.out, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        rows = list(
            pool.map(lambda c: replay(c, args.model, args.limits == "on"), picked)
        )
    with open(os.path.join(args.out, "results.jsonl"), "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    with open(os.path.join(args.out, "review.md"), "w", encoding="utf-8") as f:
        f.write(review(rows))
    print("\n".join(summary(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
