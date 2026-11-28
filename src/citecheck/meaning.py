"""A second, small step after the "list the differences" checker: is each
listed difference only a rewording, or does it change the meaning?

On the 400 dev sentences (28 Nov) the differences checker
(``HINBOX_GROUNDING_DECIDE=differences``) listed a difference for many
claims it would have passed, and code then doubted every one: over 7
answer sets, 326 answers turned from "supported" to "partial", 53 of them
on wrong claims and 273 on good ones (majority vote: 2 of 180 wrong passed,
96 of 220 good doubted). Most listed differences were harmless rewording. This step
asks the model once more, per such claim: here is the claim, the article,
and the differences you listed; for each, is it "rewording" (the same
facts, certainty and attribution in other words) or "changes_meaning"?
The claim stays "supported" only when every difference is rewording.

``reply_details()`` reads a differences reply (JSON or text) into the
model's own level and its listed differences per claim; ``meaning_prompts()``
builds one prompt per claim and answer set the model passed with
differences listed (``just citecheck meaning-prompts``);
``meaning_predictions()`` turns the second answers into verdicts, scored
beside the model's own level and code's rule (``just citecheck
meaning-score``). Develop on the dev set only (``citecheck dev-set``).
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from src.utils.grounding_differences import plain_differences, real_differences

LEVELS = ("supported", "partial", "not_supported", "unclear")
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")
_PLAIN_LEVEL = re.compile(
    r"support[ _]level\W{0,4}(supported|partial|not[ _]supported|unclear)", re.I
)
_ANSWER = re.compile(r"answer\W{0,4}(rewording|changes[ _]meaning)", re.I)

Detail = Optional[Tuple[str, List[str]]]

SYSTEM = (
    "You check one claim against a news article. Another check listed the "
    "ways the claim differs from what the article says. For each listed "
    "difference, decide whether it is only rewording or changes the meaning.\n"
    '- "rewording": the claim says the same thing as the article in other '
    "words: the same facts, numbers, dates, names and people, the same "
    "certainty, and the same attribution.\n"
    '- "changes_meaning": the claim states something the article does not: '
    "another fact, number, date, name, place or person; a cause, motive, "
    'consequence, "first" or judgment the article does not make; more '
    'certainty than the article has (a dropped "allegedly" or '
    '"expected", a plan written as done); a dropped attribution; or '
    "another person's detail.\n"
    "Read the article passage before deciding. Answer as JSON: "
    '{"items": [{"difference_number": 1, "reasoning": "<at most 30 words>", '
    '"answer": "rewording" or "changes_meaning"}]}, one item per difference.'
)


def _payload(text: str) -> Optional[Any]:
    body = _FENCE.sub("", (text or "").strip())
    start, end = body.find("{"), body.rfind("}")
    if not 0 <= start < end:
        return None
    try:
        return json.loads(body[start : end + 1])
    except json.JSONDecodeError:
        return None


def reply_details(text: str, count: int) -> List[Detail]:
    """(the model's own level, its listed differences) per claim of a
    differences reply, before code decides; None where a claim has no
    readable verdict. Read like ``read_reply()`` in ``score.py``."""
    out: List[Detail] = [None] * count
    payload = _payload(text)
    if payload is None:
        body = _FENCE.sub("", (text or "").strip())
        found, prev = [], 0
        for m in _PLAIN_LEVEL.finditer(body):
            level = m.group(1).lower().replace(" ", "_")
            found.append((level, plain_differences(body[prev : m.start()])))
            prev = m.end()
        return list(found) if len(found) == count else out
    items = payload.get("items") if isinstance(payload, dict) else None
    for pos, item in enumerate(items or []):
        if not isinstance(item, dict):
            continue
        n = item.get("claim_number")
        i = n - 1 if isinstance(n, int) and 1 <= n <= count else pos
        level = item.get("support_level")
        if i < count and level in LEVELS:
            out[i] = (level, real_differences(item.get("differences")))
    return out


def details_by_sample(replies: List[Dict[str, Any]]) -> Dict[int, Dict[str, Detail]]:
    """Answer set (``sample``) -> item id -> detail, from reply rows."""
    out: Dict[int, Dict[str, Detail]] = {}
    for row in replies:
        ids = row.get("item_ids") or []
        got = out.setdefault(int(row.get("sample") or 0), {})
        for iid, d in zip(ids, reply_details(row.get("reply") or "", len(ids))):
            got[iid] = d
    return out


def asked_id(item_id: str, sample: int) -> str:
    """The id a second-step prompt carries: the item and its answer set."""
    return f"{item_id}#s{sample}"


def needs_second_look(detail: Detail) -> bool:
    """The model said supported and listed a difference: code alone would
    doubt it."""
    return bool(detail and detail[0] == "supported" and detail[1])


def meaning_prompts(
    items: List[Dict[str, Any]],
    texts: Dict[str, str],
    details: Dict[int, Dict[str, Detail]],
    max_chars: int = 80000,
) -> List[Dict[str, Any]]:
    """One prompt ``{"item_ids": [asked_id()], "messages"}`` per item and
    answer set that needs a second look, for ``scripts/tinker_citecheck.py
    sample`` (one answer each)."""
    rows = []
    for sample, got in sorted(details.items()):
        for r in items:
            d = got.get(r["item_id"])
            if not needs_second_look(d):
                continue
            listed = "\n".join(f"{n}. {x}" for n, x in enumerate(d[1], 1))
            user = (
                f"## SOURCE ARTICLE (ID: {r['article_id']})\n"
                f"{(texts.get(r['article_id']) or '')[:max_chars]}\n\n"
                f"## CLAIM\n[{r['entity_name']}] {r['claim']}\n\n"
                f"## DIFFERENCES LISTED\n{listed}"
            )
            rows.append(
                {
                    "item_ids": [asked_id(r["item_id"], sample)],
                    "messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": user},
                    ],
                }
            )
    return rows


def meaning_answers(text: str) -> List[str]:
    """The "rewording" / "changes_meaning" answers in a reply, in order."""
    payload = _payload(text)
    items = payload.get("items") if isinstance(payload, dict) else None
    if isinstance(items, list):
        got = [
            str(i.get("answer", "")).strip().lower().replace(" ", "_")
            for i in items
            if isinstance(i, dict)
        ]
        return [a for a in got if a in ("rewording", "changes_meaning")]
    return [m.group(1).lower().replace(" ", "_") for m in _ANSWER.finditer(text or "")]


Predicted = Dict[str, Optional[str]]


def meaning_predictions(
    details: Dict[int, Dict[str, Detail]], replies: List[Dict[str, Any]]
) -> Tuple[Dict[str, Dict[int, Predicted]], Dict[str, int]]:
    """Three readings of each answer set, as ``{"model", "code", "meaning"}``
    -> sample -> item id -> verdict:

    - ``model``: the model's own level, listed differences ignored;
    - ``code``: "supported" with any difference listed is "partial"
      (``HINBOX_GROUNDING_DECIDE=differences`` as it is);
    - ``meaning``: as ``code``, except that the claim stays "supported" when
      the second step answered "rewording" for every listed difference (a
      missing or short second answer counts as doubt)."""
    second = {
        r["item_ids"][0]: r.get("reply") or "" for r in replies if r.get("item_ids")
    }
    out: Dict[str, Dict[int, Predicted]] = {"model": {}, "code": {}, "meaning": {}}
    counts = {"asked": 0, "answered": 0, "kept_supported": 0, "doubted": 0, "unread": 0}
    for sample, got in details.items():
        model, code, meaning = (out[k].setdefault(sample, {}) for k in out)
        for iid, d in got.items():
            level = d[0] if d else None
            model[iid] = code[iid] = meaning[iid] = level
            if not needs_second_look(d):
                continue
            code[iid] = meaning[iid] = "partial"
            counts["asked"] += 1
            key = asked_id(iid, sample)
            counts["answered"] += key in second
            answers = meaning_answers(second.get(key, ""))
            if len(answers) < len(d[1]):
                counts["unread"] += 1
            elif all(x == "rewording" for x in answers[: len(d[1])]):
                counts["kept_supported"] += 1
                meaning[iid] = "supported"
            else:
                counts["doubted"] += 1
    return out, counts
