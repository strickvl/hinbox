"""Hand-checked lists of what a profile should say after reading one article.

The profile-update eval used to pass a ``new_fact`` case as soon as the new
profile cited the article. A profile that cited it on one vague sentence
("Camp 7 is described as a secretive prison camp") passed the same as one
that added the ruling the article was about. These labels close that gap.

One JSON row per (entity, article) in ``{eval_dir}/claim_labels.jsonl``::

    {"entity_name": "Camp 7", "entity_type": "location",
     "article_id": "9ed95f1e-...", "use": "update" | "create",
     "case_id": "new_fact:Camp 7:10:9ed95f1e-..."   # update rows only
     "status": "draft" | "reviewed", "note": "...",
     "disputes_ok": false,   # true when the article itself reports a conflict
     "facts": [{"id": "routine", "claim": "...", "passage": "<short quote>",
                "said_by": "who states or alleges it", "core": true,
                "match": [["routine"], ["search"]]}],
     "must_not": [{"kind": "wrong_actor" | "negation" | "other", "note": "...",
                   "match": [["ruiz"], ["prison lawyer"]]}]}

``match`` is a list of word groups. A sentence matches when, for every group,
it holds a word that starts with one of the group's entries (case-insensitive,
citations ignored), so ``[["routine"], ["search"]]`` matches "said searches
at Camp 7 were routine". Attribution goes into the groups: a fact the article
only reports as an allegation gets a group like ``["ruiz", "lawyer", "said",
"alleg"]``, so stating it as plain fact doesn't count.

A fact counts as stated when a sentence backed by the article matches it:
the sentence cites it, or the paragraph's next citation does ("A. B.^[a9]"
backs A too, since profiles cite once per paragraph). A
``must_not`` reading counts when any new or changed sentence matches it. The
matcher can't read negation ("did not clear him of wrongdoing" holds both
"clear" and "wrongdoing"), so ``must_not`` groups use wording a correct
sentence wouldn't contain.

``status`` starts as ``draft`` (drafted by a model from the article and the
saved runs); only ``reviewed`` rows are scored unless asked otherwise.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from src.evals.profile_cases import _CITATION, _sentences

LABELS_FILE = "claim_labels.jsonl"
STATUSES = ("reviewed", "draft", "off")
_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})


def _norm(text: str) -> str:
    return " ".join(_CITATION.sub("", text).translate(_QUOTES).lower().split())


def _pattern(term: str) -> str:
    # A number must match whole: "55" is not the start of "Detainee 552"
    end = r"(?![0-9])" if term[-1:].isdigit() else ""
    return rf"(?<![a-z0-9]){re.escape(_norm(term))}{end}"


def matches(sentence: str, groups: List[List[str]]) -> bool:
    """True when every group has an entry that starts a word in ``sentence``."""
    text = _norm(sentence)
    return bool(groups) and all(
        any(re.search(_pattern(t), text) for t in group) for group in groups
    )


def _cites(sentence: str, article_id: str) -> bool:
    return any(
        article_id == i.strip()
        for group in _CITATION.findall(sentence)
        for i in group.split(",")
    )


def _citing_sentences(text: str, article_id: str) -> List[str]:
    """Sentences backed by a citation of ``article_id``: one that carries it,
    or one whose paragraph's next citation does. Profiles are asked to cite
    once per paragraph, so "A. B. C.^[a9]" backs all three sentences with a9;
    a sentence after the paragraph's last citation is backed by nothing."""
    citing: List[str] = []
    for paragraph in text.split("\n"):
        waiting: List[str] = []
        for sentence in _sentences(paragraph):
            waiting.append(sentence)
            if _CITATION.search(sentence):
                if _cites(sentence, article_id):
                    citing.extend(waiting)
                waiting = []
    return citing


def load_labels(
    eval_dir: str, status: str = "reviewed"
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Labels by (entity_name, article_id). ``status="draft"`` also loads
    unreviewed rows; ``"off"`` loads none. The last row for a key wins."""
    path = os.path.join(eval_dir, LABELS_FILE)
    if status == "off" or not os.path.exists(path):
        return {}
    allowed = {"reviewed"} if status == "reviewed" else {"reviewed", "draft"}
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {
        (r["entity_name"], r["article_id"]): r
        for r in rows
        if r.get("status", "draft") in allowed
    }


def score_claims(
    label: Dict[str, Any], old_text: str, new_text: str, article_id: str
) -> Dict[str, Any]:
    """Which labelled facts the new profile states and which banned readings
    it adds. Pure, no model calls. For a new profile pass ``old_text=""``.

    Facts whose words already match an old sentence are listed under
    ``old_matches`` for whoever reviews the label (the fact may not be new, or
    the match words may be too loose); they are still required.
    """
    old = _sentences(old_text)
    new = _sentences(new_text)
    old_set: Set[str] = {_norm(s) for s in old}
    citing = _citing_sentences(new_text, article_id)
    changed = [s for s in new if _norm(s) not in old_set]

    found: List[str] = []
    missing_core: List[str] = []
    for fact in label.get("facts", []):
        if any(matches(s, fact["match"]) for s in citing):
            found.append(fact["id"])
        elif fact.get("core", True):
            missing_core.append(fact["id"])
    hits = [
        {"kind": n["kind"], "note": n.get("note", ""), "sentence": s[:300]}
        for n in label.get("must_not", [])
        for s in changed
        if matches(s, n["match"])
    ]
    facts = label.get("facts", [])
    return {
        "claims_expected": len(facts),
        "claims_found": found,
        "core_missing": missing_core,
        "old_matches": [
            f["id"] for f in facts if any(matches(s, f["match"]) for s in old)
        ],
        "must_not_hits": hits,
        "claims_ok": not missing_core and not hits,
    }


def label_for(
    labels: Dict[Tuple[str, str], Dict[str, Any]], case: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """The label for a profile-update case, if one was written for it."""
    return labels.get((case["entity_name"], case["article_id"]))
