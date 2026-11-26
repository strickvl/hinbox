"""Chat-format training files for a tuned citation checker.

Each example is one judge call exactly as the pipeline makes it: the same
system message, the article, the numbered claims (``grounding_messages()``),
then the reply the tuned model should give, in the JSON the pipeline reads
from a local model (``{"items": [{"claim_number", "reasoning",
"support_level"}]}``). Because the question and the answer format are the
pipeline's own, a tuned model is used by setting ``HINBOX_GROUNDING_MODEL``,
with no code change.

The reasoning in each answer is short: the article words the verdict rests
on, and for a wrong claim what differs. Teaching that short answer is the
point: untuned Inkling 2 Mini writes about 4,500 tokens of reasoning per call
before answering.

The number of claims per example varies (1 to ``max_claims``), so the model
sees both a single claim and a full call. Files hold ``{"messages": [...]}``
rows, the format of the Tinker cookbook's ``FromConversationFileBuilder``.
A label from a test article is refused: those articles are only for the
hand-labelled score.
"""

import json
import random
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

from src.citecheck.items import article_split
from src.utils.grounding import GROUNDING_MAX_ARTICLE_CHARS, grounding_messages

LEVELS = ("supported", "partial", "not_supported", "unclear")


def reply_json(verdicts: List[Tuple[str, str]]) -> str:
    """The reply for (level, reasoning) per claim, numbered from 1."""
    items = [
        {"claim_number": i + 1, "reasoning": reasoning, "support_level": level}
        for i, (level, reasoning) in enumerate(verdicts)
    ]
    return json.dumps({"items": items}, ensure_ascii=False)


def build_examples(
    labels: List[Dict[str, Any]],
    texts: Dict[str, str],
    max_claims: int = 20,
    seed: int = 3,
    max_article_chars: int = GROUNDING_MAX_ARTICLE_CHARS,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Training examples from labels (teacher and planted rows: item_id,
    article_id, entity_name, claim, level, reasoning) and counts.

    Claims are grouped by article, shuffled, and cut into calls of random
    size; an article with no text is skipped."""
    rng = random.Random(seed)
    by_article: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    seen = set()
    for row in labels:
        if article_split(row["article_id"]) == "test":
            raise ValueError(
                f"label {row['item_id']} comes from test article {row['article_id']}"
            )
        if row.get("level") not in LEVELS or row["item_id"] in seen:
            continue
        seen.add(row["item_id"])
        by_article[row["article_id"]].append(row)
    examples: List[Dict[str, Any]] = []
    counts: Counter = Counter()
    for aid in sorted(by_article):
        text = (texts.get(aid) or "")[:max_article_chars]
        if not text:
            counts["no_text"] += len(by_article[aid])
            continue
        group = by_article[aid][:]
        rng.shuffle(group)
        while group:
            size = rng.randint(1, max_claims)
            chunk, group = group[:size], group[size:]
            messages = grounding_messages(
                aid, text, [(c["entity_name"], c["claim"]) for c in chunk]
            )
            reply = reply_json(
                [(c["level"], (c.get("reasoning") or "").strip()) for c in chunk]
            )
            messages.append({"role": "assistant", "content": reply})
            examples.append({"messages": messages})
            for c in chunk:
                counts[f"{c.get('source', '?')}:{c['level']}"] += 1
    rng.shuffle(examples)
    return examples, {"examples": len(examples), "claims": dict(counts)}


def cap_planted(
    labels: List[Dict[str, Any]], ratio: float, seed: int = 3
) -> List[Dict[str, Any]]:
    """Keep every label that isn't planted, and at most ``ratio`` planted
    labels per other label, picked at random.

    The 26 Nov strict export held 84,720 planted claims beside 1,413 teacher
    claims, so the teachers' errors (dropped attributions and hedges, added
    causes, sentences about "the profile"), which no planted kind makes, were
    under 2% of what the model would learn from."""
    planted = [r for r in labels if r.get("source") == "planted"]
    other = [r for r in labels if r.get("source") != "planted"]
    keep = int(ratio * len(other))
    if len(planted) <= keep:
        return labels
    return other + random.Random(seed).sample(planted, keep)


def split_labels(
    labels: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """(train, tune) by the article's split."""
    train = [r for r in labels if article_split(r["article_id"]) == "train"]
    tune = [r for r in labels if article_split(r["article_id"]) == "tune"]
    return train, tune
