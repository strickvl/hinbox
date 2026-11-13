"""Try the guard against notes about "the profile" on a finished run's edit replies.

    uv run python -m scripts.profile_notes_bench <run>/traces/run-*.jsonl \
        --out <dir> [--articles <parquet>] [--sample 60] \
        [--ask 40 --model ollama/<name>]

Reads the run's traces and takes every profile-edit reply that ended normally
and has a new sentence about the profile ("..., conflicting with the
profile's count of 93"; ``src/engine/profile_notes.py``). Each reply's edits
go through ``guard_edits()`` against the profile its prompt showed, with the
articles' published dates when ``--articles`` is given:

- without a model (the code path): how many notes become a dated later
  state, the article's half alone, a reworded dispute, or are dropped,
  beside what the guards did with the same sentences before they knew
  notes about the profile;
- with ``--ask N``: for N of those replies, also with the dispute check's
  model call (the pipeline's default path), and what it answered.

Writes ``rewrites.md`` (and ``checked.md``): each note beside what took its
place, for reading. They hold article-derived text: keep them off GitHub.
"""

import argparse
import json
import os
import random
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

from scripts.edit_limits_bench import _PROFILE, edit_call, parse_reply
from src.constants import OLLAMA_MODEL
from src.engine.edit_guards import _claim, guard_edits
from src.engine.profile_notes import is_profile_note
from src.evals.profile_cases import _sentences

_ARTICLE_ID = re.compile(r"\n\nNew article \(ID: ([^)]+)\):\n")
_ENTITY = re.compile(r'profile of the ([\w ]+?) "(.+?)"\. ')
Row = List[str]  # [the note, what happened to it, what took its place]


def _notes_in(reply: Any) -> List[str]:
    return [
        s
        for e in reply.edits
        if e.op == "add"
        for s in _sentences(e.text)
        if is_profile_note(_claim(s))
    ]


def scan(paths: Iterable[str]) -> Tuple[List[Tuple[str, int]], Counter]:
    """(where each reply with a note is, totals), one trace line at a time."""
    places: List[Tuple[str, int]] = []
    totals: Counter = Counter()
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for number, line in enumerate(f):
                call = edit_call(json.loads(line)) if line.strip() else None
                reply = parse_reply(call["reply"]) if call else None
                if not call or reply is None or "length" in call["finish"]:
                    continue
                found = _notes_in(reply)
                totals["replies"] += 1
                totals["new sentences"] += sum(
                    len(_sentences(e.text)) for e in reply.edits if e.op == "add"
                )
                totals["notes"] += len(found)
                if found:
                    totals["replies with a note"] += 1
                    places.append((path, number))
    return places, totals


def load(places: List[Tuple[str, int]]) -> List[Dict[str, Any]]:
    """Those calls in full, read again from their lines."""
    wanted = set(places)
    full: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for path in sorted({path for path, _ in wanted}):
        with open(path, encoding="utf-8") as f:
            for number, line in enumerate(f):
                if (path, number) in wanted:
                    full[(path, number)] = edit_call(json.loads(line)) or {}
    return [full[place] for place in places]


def published_dates(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["id", "published_date"]).to_pydict()
    return dict(zip(table["id"], table["published_date"]))


def _report(call: Dict[str, Any], dates: Dict[str, Any], ask: Any) -> Tuple:
    """(the reply, its guarded edits, the guard report) for a recorded call."""
    user = call["messages"][-1]["content"]
    found = _PROFILE.search(user)
    article = _ARTICLE_ID.search(user)
    reply = parse_reply(call["reply"])
    if not found or reply is None:
        return None, [], {}
    when = dates.get(article.group(1)) if article else None
    kept, report = guard_edits(found.group(1), reply.edits, when, dates, ask=ask)
    return reply, kept, report


def guarded(call: Dict[str, Any], dates: Dict[str, Any], ask: Any = None) -> List[Row]:
    """A row for each note of a reply, after ``guard_edits()``."""
    reply, kept, report = _report(call, dates, ask)
    if reply is None:
        return []
    rows = [[old, "later state", new] for old, new in report["later_updates"]]
    rows += [[old, f"note, {how}", new] for old, how, new in report["profile_notes"]]
    rows += [[old, "repeat, dropped", ""] for old in report["repeats"]]
    rows += [[old, "invented dispute, dropped", ""] for old in report["disputes"]]
    done = {r[0] for r in rows}
    for row in report.get("dispute_check") or []:
        if row[0] not in done:  # a conflict or failed check is under "note,"
            rows.append([row[0], f"model: {row[1]}", row[2] if len(row) > 2 else ""])
    # The report keeps 200 characters of each sentence; show them whole
    notes = {n[:200]: n for n in _notes_in(reply)}
    after = {s[:200]: s for e in kept if e.op == "add" for s in _sentences(e.text)}
    return [
        [notes[old], how, after.get(new, new)] for old, how, new in rows if old in notes
    ]


def before(call: Dict[str, Any], dates: Dict[str, Any]) -> Dict[str, str]:
    """What the guards did with each note before they knew notes about the
    profile (no model): only "sources disagree" wordings were looked at."""
    import src.engine.edit_guards as guards

    knows, guards.is_profile_note = guards.is_profile_note, lambda claim: False
    try:
        reply, _, report = _report(call, dates, None)
    finally:
        guards.is_profile_note = knows
    if reply is None:
        return {}
    how = {old: "later state" for old, _ in report["later_updates"]}
    how.update({old: "repeat, dropped" for old in report["repeats"]})
    how.update({old: "invented dispute, dropped" for old in report["disputes"]})
    return {n: how.get(n[:200], "kept as written") for n in _notes_in(reply)}


def _asker(call: Dict[str, Any], model: str) -> Any:
    from src.engine.profile_edits import _dispute_asker
    from src.utils.llm import local_generation

    named = _ENTITY.search(call["messages"][0]["content"])
    kind, name = named.groups() if named else ("entity", "")
    return _dispute_asker(local_generation, model, kind, name)


def write(path: str, title: str, rows: List[Row]) -> None:
    out = [f"# {title}\n"]
    for old, how, new in rows:
        out.append(f"- **{how}**\n  - was: {old}\n  - now: {new or '(nothing)'}\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))


def tally(rows: List[Row]) -> str:
    counts = Counter(how for _, how, _ in rows)
    left = sum(bool(new) and is_profile_note(_claim(new)) for _, _, new in rows)
    lines = [f"  {how}: {n}" for how, n in counts.most_common()]
    return "\n".join(lines + [f"  still naming the profile afterwards: {left}"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("traces", nargs="+", help="trace JSONL files of a run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--articles", help="articles Parquet, for published dates")
    ap.add_argument("--sample", type=int, default=60, help="rewrites to write out")
    ap.add_argument("--ask", type=int, default=0, help="replies to check by model")
    ap.add_argument("--model", default=OLLAMA_MODEL)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    places, totals = scan(args.traces)
    print("\n".join(f"{k}: {v}" for k, v in totals.items()))
    dates = published_dates(args.articles)
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    rows: List[Row] = []
    moved: Counter = Counter()  # (before this guard, with it)
    # In batches, so the prompts of a whole run are never all in memory
    for start in range(0, len(places), 200):
        for call in load(places[start : start + 200]):
            got, was = guarded(call, dates), before(call, dates)
            rows += got
            moved.update((was.get(old, "?"), how) for old, how, _ in got)
    print(f"without a model, {len(rows)} notes:\n{tally(rows)}")
    print("before this guard -> with it:")
    for (was_how, how), n in sorted(moved.items(), key=lambda kv: (kv[0][0], -kv[1])):
        print(f"  {was_how} -> {how}: {n}")
    write(
        os.path.join(args.out, "rewrites.md"),
        "Notes about the profile, without a model",
        rng.sample(rows, min(args.sample, len(rows))),
    )
    if args.ask:
        picked = load(rng.sample(places, min(args.ask, len(places))))
        checked = [r for c in picked for r in guarded(c, dates, _asker(c, args.model))]
        print(f"with the model check, {len(checked)} notes:\n{tally(checked)}")
        write(
            os.path.join(args.out, "checked.md"),
            "Notes about the profile, with the dispute check",
            checked,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
