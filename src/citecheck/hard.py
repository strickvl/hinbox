"""Training examples for the claims a self-distilled checker never gets right.

Self-distillation (``scripts/tinker_citecheck.py distill``) asks untuned
Inkling each training example 4 times and keeps an answer only when all its
verdicts match the labels. An example with one claim the model always gets
wrong is never kept, so the model never learns that claim: in the 27 Nov v3
run, 74 claims the teachers called "partial" were passed in all 4 tries, and
the tuned checker went on passing the same 26 wrong test claims as untuned
Inkling.

``hard_claims()`` reads the distill log (every answer's verdicts) and finds
each claim that every readable answer put on the wrong side of yes/no
(supported against the rest). ``hard_prompts()`` makes one new example per
such claim, from the example's own system message and article text, so the
claim can be asked again on its own, more times and at a higher temperature
(``distill --samples 16 --temperature 1.0``), and any answer that lands on
the label is kept with the model's own reasoning. ``mixed()`` puts the kept
answers beside the earlier kept rows, repeated, for training.

``teacher_rows()`` is the optional step for the hard claims still never
right: a cloud judge (``--teacher``) answers them under the same rules, and an
answer that agrees with the label becomes a row whose reasoning is the
judge's. Only judges whose terms allow it belong here.
"""

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.citecheck.export import reply_json
from src.utils.corrections import sentence_key

CLAIMS_HEADER = "\n\n## CLAIMS TO VERIFY\n"
_ARTICLE_ID = re.compile(r"^## SOURCE ARTICLE \(ID: (.*?)\)\n")
_ENTITY = re.compile(r"^\[(.*?)\] ")


def row_key(messages: List[Dict[str, Any]]) -> str:
    """The key the distill log gives an example's prompt (``row_key()`` in
    ``scripts/tinker_citecheck.py``, which can't import Hinbox)."""
    raw = json.dumps(messages, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def _content(message: Dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content or [] if p.get("type") == "text")


def parse_prompt(
    messages: List[Dict[str, Any]],
) -> Tuple[str, str, List[Tuple[str, str]]]:
    """(article id, the user message up to the claims, [(entity, claim)]) of a
    judge prompt built by ``grounding_messages()``. A claim runs over several
    lines until the line that starts with the next number."""
    user = next(_content(m) for m in messages if m["role"] == "user")
    head, sep, tail = user.rpartition(CLAIMS_HEADER)
    match = _ARTICLE_ID.match(head)
    if not sep or not match:
        raise ValueError("not a judge prompt: no article header or claims section")
    claims: List[List[str]] = []
    for line in tail.split("\n"):
        prefix = f"{len(claims) + 1}. "
        if line.startswith(prefix):
            rest = line[len(prefix) :]
            entity = _ENTITY.match(rest)
            claims.append(
                [entity.group(1), rest[entity.end() :]] if entity else ["", rest]
            )
        elif claims:
            claims[-1][1] += "\n" + line
    return match.group(1), head, [(e, c) for e, c in claims]


def _yes(level: Optional[str]) -> bool:
    return level == "supported"


def hard_claims(
    examples: List[Dict[str, Any]], log: List[Dict[str, Any]], min_answers: int = 3
) -> List[Dict[str, Any]]:
    """Claims of ``examples`` (export rows) that every readable answer in the
    distill ``log`` got wrong on yes/no, with at least ``min_answers``
    answers giving that claim a verdict. Kept examples are passed over."""
    by_key = {r["key"]: r for r in log}
    out = []
    for ex in examples:
        prompt = ex["messages"][:-1]
        entry = by_key.get(row_key(prompt))
        if entry is None or entry.get("kept") is not None:
            continue
        aid, head, claims = parse_prompt(prompt)
        system = [m for m in prompt if m["role"] == "system"]
        for n, (entity, claim) in enumerate(claims, 1):
            gold = entry["gold"].get(str(n))
            said = [a[str(n)] for a in entry.get("answers", []) if str(n) in a]
            if gold is None or len(said) < min_answers:
                continue
            if all(_yes(s) != _yes(gold) for s in said):
                out.append(
                    {
                        "article_id": aid,
                        "entity_name": entity,
                        "claim": claim,
                        "level": gold,
                        "answers": said,
                        "system": system,
                        "head": head,
                    }
                )
    return out


def claim_prompt(
    system: List[Dict[str, Any]], head: str, claims: List[Tuple[str, str]]
) -> List[Dict[str, Any]]:
    """A judge prompt for ``claims`` over the same article text and system
    message, numbered from 1 as ``grounding_messages()`` numbers them."""
    from src.utils.grounding import _claim_line

    lines = "\n".join(_claim_line(i + 1, e, c) for i, (e, c) in enumerate(claims))
    return system + [{"role": "user", "content": head + CLAIMS_HEADER + lines}]


def label_kinds(labels: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, str], str]:
    """(article id, sentence key) -> where a label came from: ``planted:<kind>``
    for a planted claim, else its ``source``."""
    out = {}
    for r in labels:
        source = r.get("source") or "planted"
        kind = f"{source}:{r['kind']}" if r.get("kind") else source
        out[(r["article_id"], sentence_key(r["claim"]))] = kind
    return out


def hard_prompts(
    hard: List[Dict[str, Any]], kinds: Optional[Dict[Tuple[str, str], str]] = None
) -> List[Dict[str, Any]]:
    """One example per hard claim: its prompt, the label as the reply (which
    ``distill`` reads as the answer to match), and where the label came from."""
    kinds = kinds or {}
    rows, seen = [], set()
    for h in hard:
        key = (h["article_id"], sentence_key(h["claim"]))
        if key in seen:
            continue
        seen.add(key)
        prompt = claim_prompt(h["system"], h["head"], [(h["entity_name"], h["claim"])])
        reply = {"role": "assistant", "content": reply_json([(h["level"], "")])}
        rows.append(
            {
                "messages": prompt + [reply],
                "hard": {
                    "article_id": h["article_id"],
                    "level": h["level"],
                    "answers": h["answers"],
                    "kind": kinds.get(key, "teachers"),
                },
            }
        )
    return rows


def hard_summary(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    """Counts of hard claims by label kind and label (no text)."""
    counts: Counter = Counter()
    for r in rows:
        counts[f"{r['hard']['kind']} {r['hard']['level']}"] += 1
    return dict(sorted(counts.items()))


def _all_planted_of(
    row: Dict[str, Any], kinds: Dict[Tuple[str, str], str], drop: Tuple[str, ...]
) -> bool:
    aid, _, claims = parse_prompt(row["messages"][:-1])
    got = [kinds.get((aid, sentence_key(c))) for _, c in claims]
    return bool(got) and all(k in drop for k in got)


def mixed(
    base: List[Dict[str, Any]],
    extra: List[Dict[str, Any]],
    repeat: int = 4,
    seed: int = 4,
    kinds: Optional[Dict[Tuple[str, str], str]] = None,
    drop_kinds: Tuple[str, ...] = (),
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """``base`` rows (less any whose claims are all planted of ``drop_kinds``,
    e.g. ``planted:year``, which the model already gets right) plus each
    ``extra`` row ``repeat`` times, shuffled with a fixed seed."""
    drop = tuple(drop_kinds)
    kept = [r for r in base if not (drop and _all_planted_of(r, kinds or {}, drop))]
    rows = kept + [r for r in extra for _ in range(max(1, repeat))]
    random.Random(seed).shuffle(rows)
    counts = {
        "base": len(base),
        "base_dropped": len(base) - len(kept),
        "extra": len(extra),
        "rows": len(rows),
    }
    return rows, counts


def missed(
    prompts: List[Dict[str, Any]], log: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Hard prompts the distill ``log`` tried and never kept."""
    kept = {r["key"] for r in log if r.get("kept") is not None}
    tried = {r["key"] for r in log}
    out = []
    for r in prompts:
        key = row_key(r["messages"][:-1])
        if key in tried and key not in kept:
            out.append(r)
    return out


def estimate_dollars(
    prompts: List[Dict[str, Any]],
    price_in: float,
    price_out: float,
    out_tokens: int = 600,
) -> float:
    """Rough cost of one judge call per prompt: 4 characters per token in,
    ``out_tokens`` out; prices in dollars per million tokens."""
    chars = sum(len(_content(m)) for r in prompts for m in r["messages"][:-1])
    return (chars / 4 * price_in + len(prompts) * out_tokens * price_out) / 1e6


def _same_level(said: Optional[str], gold: str, loose: bool) -> bool:
    if said == gold:
        return True
    off = ("partial", "not_supported")
    return loose and said in off and gold in off


def teacher_row(
    prompt_row: Dict[str, Any],
    verdicts: List[Optional[Tuple[Any, Optional[str], Optional[str]]]],
    teacher: str,
    loose: bool = True,
) -> Optional[Dict[str, Any]]:
    """A training row from a judge's ``verdicts`` (``judge_claims()``) on a
    hard prompt, or None when a verdict is missing, has no reasoning or
    disagrees with the label. The reasoning goes in the reply's thinking
    part, as in a distilled row, and the JSON answer follows."""
    from src.citecheck.items import _level

    golds = [
        it["support_level"]
        for it in json.loads(_content(prompt_row["messages"][-1]))["items"]
    ]
    if len(verdicts) != len(golds):
        return None
    pairs = []
    for v, gold in zip(verdicts, golds):
        if v is None or not (v[1] or "").strip():
            return None
        level = _level(v[0])
        if not _same_level(level, gold, loose):
            return None
        pairs.append((level, v[1].strip()))
    thinking = "\n\n".join(f"Claim {i + 1}: {r}" for i, (_, r) in enumerate(pairs))
    content = [
        {"type": "thinking", "thinking": thinking},
        {"type": "text", "text": reply_json(pairs)},
    ]
    return {
        "messages": prompt_row["messages"][:-1]
        + [{"role": "assistant", "content": content}],
        "teacher": teacher,
        "hard": prompt_row.get("hard", {}),
    }


def teacher_rows(
    prompts: List[Dict[str, Any]], teacher: str, loose: bool = True
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Ask ``teacher`` about each hard prompt (one call each) and keep the rows
    whose verdicts agree with the labels. A prompt asked under other judge
    rules than ``HINBOX_GROUNDING_RULES`` now gives is refused."""
    from src.utils.grounding import grounding_system_prompt, judge_claims

    counts: Dict[str, int] = defaultdict(int)
    out = []
    for r in prompts:
        prompt = r["messages"][:-1]
        system = next(_content(m) for m in prompt if m["role"] == "system")
        if system != grounding_system_prompt():
            raise SystemExit(
                "the hard prompts were asked under other judge rules: set "
                "HINBOX_GROUNDING_RULES as for the export"
            )
        aid, head, claims = parse_prompt(prompt)
        text = head[_ARTICLE_ID.match(head).end() :]
        try:
            verdicts = judge_claims(
                article_id=aid,
                article_text=text,
                claims=claims,
                model_type="cloud",
                ask_passage=False,
                model=teacher,
            )
        except Exception:  # noqa: BLE001 - a failed call is counted, not fatal
            counts["failed"] += 1
            continue
        row = teacher_row(r, verdicts, teacher, loose)
        counts["kept" if row else "disagreed"] += 1
        if row:
            out.append(row)
    return out, dict(counts)
