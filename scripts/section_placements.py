"""Where new sentences land in replayed profiles once they have starter headings.

Reads the snapshots ``just eval-replay`` writes (one ``step_NNN.json`` per
article) and, for every update after a profile was reorganised
(``profile["reorganised"] == "done"``), finds each sentence the update added
and the heading it sits under. Writes:

- ``review.md``: per entity and step, the article title and each added
  sentence with its section ("Military commission or legal case › Plea
  talks"), for a person to mark right or wrong;
- ``summary.json``: per entity, added sentences per top-level section, how
  many went to the last section (where an unplaced add ends up), headings
  that updates created, the longest section's own text at the end and at any
  step, and the deepest heading level below the top one.

    uv run python -m scripts.section_placements <eval>/profile_replay/<model> \\
        --out <dir> [--only <slug> ...]

Outputs hold profile text: keep them off GitHub.
"""

import argparse
import json
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


def _steps(entity_dir: str) -> List[Dict[str, Any]]:
    names = sorted(f for f in os.listdir(entity_dir) if re.match(r"step_\d+\.json$", f))
    out = []
    for name in names:
        with open(os.path.join(entity_dir, name), encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def outline(text: str) -> List[Tuple[int, int, str, int]]:
    """(line, level, title, own-text chars) of each heading."""
    from src.engine.section_split import _headings

    lines = text.splitlines()
    heads = _headings(lines)
    out = []
    for k, (i, level, title) in enumerate(heads):
        end = heads[k + 1][0] if k + 1 < len(heads) else len(lines)
        out.append((i, level, title, len("\n".join(lines[i + 1 : end]).strip())))
    return out


def section_of(text: str, sentence: str) -> str:
    """The heading path above ``sentence`` ("Top › Sub"), "(opening)" before
    any heading, or "?" when it can't be found."""
    from src.engine.profile_edits import _find

    span = _find(text, sentence)
    if span is None:
        return "?"
    line = text[: span[0]].count("\n")
    path: List[Tuple[int, str]] = []
    for i, level, title, _ in outline(text):
        if i > line:
            break
        path = [p for p in path if p[0] < level] + [(level, title)]
    return " › ".join(t for _, t in path) or "(opening)"


def added_sentences(old: str, new: str) -> List[str]:
    from src.engine.section_split import _sentences

    before = Counter(_sentences(old))
    out = []
    for s in _sentences(new):
        if before[s]:
            before[s] -= 1
        else:
            out.append(s)
    return out


def entity_report(steps: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[str]]:
    by_section: Counter = Counter()
    review: List[str] = []
    created: List[str] = []
    added = last = 0
    longest = depth = 0
    reorganised_at: Optional[int] = None
    for prev, cur in zip(steps, steps[1:]):
        old, new = prev["profile"].get("text") or "", cur["profile"].get("text") or ""
        if prev["profile"].get("reorganised") != "done":
            if cur["profile"].get("reorganised") == "done":
                reorganised_at = cur["step"]
            continue
        heads = outline(new)
        if heads:
            top = min(h[1] for h in heads)
            longest = max(longest, max(h[3] for h in heads))
            depth = max(depth, max(h[1] for h in heads) - top)
            last_top = [h[2] for h in heads if h[1] == top][-1]
        old_titles = {h[2] for h in outline(old)}
        created += [h[2] for h in heads if h[2] not in old_titles]
        fresh = added_sentences(old, new)
        if not fresh:
            continue
        review += [f"### Step {cur['step']}: {cur['article'].get('title', '')}", ""]
        for s in fresh:
            where = section_of(new, s)
            by_section[where.split(" › ")[0]] += 1
            added += 1
            last += bool(heads) and where.split(" › ")[0] == last_top
            review.append(f"- [ ] **{where}**: {s[:220]}")
        review.append("")
    final = outline(steps[-1]["profile"].get("text") or "") if steps else []
    summary = {
        "steps": len(steps),
        "reorganised_at_step": reorganised_at,
        "added_sentences": added,
        "added_to_last_top_section": last,
        "added_by_section": dict(by_section.most_common()),
        "headings_created_by_updates": created,
        "longest_own_section_chars_any_step": longest,
        "longest_own_section_chars_final": max((h[3] for h in final), default=0),
        "deepest_level_below_top": depth,
        "final_outline": [f"{'  ' * (h[1] - 3)}{h[2]} ({h[3]})" for h in final],
    }
    return summary, review


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("replay_dir", help="<eval>/profile_replay/<model>")
    parser.add_argument("--out", required=True)
    parser.add_argument("--only", nargs="*", help="entity folder names")
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    summaries: Dict[str, Any] = {}
    review: List[str] = []
    for slug in sorted(os.listdir(args.replay_dir)):
        path = os.path.join(args.replay_dir, slug)
        if not os.path.isdir(path) or (args.only and slug not in args.only):
            continue
        summary, lines = entity_report(_steps(path))
        summaries[slug] = summary
        review += [f"## {slug}", ""] + lines
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=1, ensure_ascii=False)
    with open(os.path.join(args.out, "review.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(review) + "\n")
    print(json.dumps(summaries, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
