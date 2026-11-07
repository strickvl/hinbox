"""Score profile updates on the cases ``just eval-cases`` built.

Each case is an existing profile plus one article. The model under test
produces an updated profile, and these checks score it without a model:

- ``length_ratio``: new length / old length. Below ``COLLAPSE_RATIO`` counts
  as a collapse. The full replay showed single updates cutting a 31k-character
  profile to 6k with every QC check passing.
- ``fact_retention``: the share of the old profile's cited sentences that are
  still there, still with a citation. A sentence counts as kept when one
  cited new sentence contains at least ``KEPT_OVERLAP`` of its content words,
  so rewording still counts.
  Heavier rewrites that fall below that need the model judge (later).
- ``unchanged_share``: the share of old sentences kept word for word. Low
  means the update rewrote text it had no reason to touch.
- ``added_sentences``: new sentences with no close match in the old profile.
- ``cites_article``: whether the case's article is cited.
- ``invented_citations``: cited ids that are neither old sources nor the
  case's article. On the replay these were all garbled copies of real
  36-character ids (cut short, a digit changed), which still break the link.
- ``dispute_claims``: new or changed sentences saying the sources disagree or differ.
  Reported only; on cases with no planted dispute most are invented.
- ``unsupported_numbers``: new or changed sentences citing the case's article
  that contain numbers, none of which appear in the article. Qwen wrote "A
  later report gives the year as 2007 instead" where the article had no 2007.
  Reported only.
- With a reviewed claim label (``src/evals/claim_labels.py``) for the case's
  entity and article, ``claims_found``/``core_missing``/``must_not_hits``:
  which expected facts the new profile states (in a sentence citing the
  article) and which wrong readings (wrong actor, reversed meaning) it adds.
- For ``disputed`` cases, ``keeps_both``: the original year is still stated
  with one of its original sources, and the changed year is stated with the
  case's article, in a sentence about the claim or right after the kept one.
  A profile that silently swaps one year for the other fails.

A case passes when nothing collapsed, retention is at least
``MIN_RETENTION``, no citation was invented, and the kind's own check holds.
For a labelled ``new_fact`` case that check is every core fact stated, no
wrong reading added and, unless the article itself reports a conflict, no
"sources disagree" sentence; an unlabelled one only has to cite the article.
"""

import functools
import hashlib
import json
import os
import re
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, Set

from src.evals.claim_labels import label_for, load_labels, score_claims
from src.evals.profile_cases import _CITATION, _content_words, _sentences

COLLAPSE_RATIO = 0.8
# Hand-checked on the Luna replay: reworded facts kept 0.45-0.67 of their
# content words, facts really dropped mostly under 0.35
KEPT_OVERLAP = 0.45
MIN_RETENTION = 0.9
# The sentence stating the new year must share at least this many content
# words with the claim, and this share of the shorter of the two word sets.
# Measured against the claim alone, a short correct sentence ("the article
# instead says the camps opened in 2003") held 2 of 22 words and failed.
DISPUTED_MIN_SHARED = 2
DISPUTED_OVERLAP = 0.15
# Wording that tells the reader two sources conflict. Whole phrases, so
# "a different lawyer" and "the armed conflict" don't count
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_DISPUTE_WORDS = re.compile(
    r"\b(disagree\w*|differs?\b|contradict\w*|discrepanc\w*"
    r"|inconsistent with|conflicts? with|conflicting (?:accounts|reports|dates"
    r"|figures|sources)|gives? different)",
    re.I,
)
# ...and that the conflict is between sources, not one reported inside the
# article ("a claim contradicted by the prison spokeswoman" is just a fact)
_SOURCE_WORDS = re.compile(
    r"\b(sources?|reports?|articles?|accounts?|records?)\b", re.I
)
MIN_CONTENT_WORDS = 3  # shorter lines (headers, "Tags:") aren't facts

# Returns the new text, or {"text", "report"} to keep a report in the run file
Updater = Callable[[Dict[str, Any]], Any]


def _cited_ids(text: str) -> Set[str]:
    return {i.strip() for group in _CITATION.findall(text) for i in group.split(",")}


def _facts(text: str) -> List[str]:
    """Sentences with enough content words to state something."""
    return [s for s in _sentences(text) if len(_content_words(s)) >= MIN_CONTENT_WORDS]


def _best_overlap(words: Set[str], candidates: List[Set[str]]) -> float:
    if not words:
        return 0.0
    return max((len(words & c) / len(words) for c in candidates), default=0.0)


def _numbers(text: str) -> Set[str]:
    return {n.replace(",", "") for n in _NUMBER.findall(_CITATION.sub("", text))}


def _shares_claim(about: Set[str], words: Set[str]) -> bool:
    """Loose test that a freshly worded sentence is about the claim.

    Words are compared by their first 4 letters, so "opened" matches "opening".
    """
    about, words = {w[:4] for w in about}, {w[:4] for w in words}
    shared = len(about & words)
    smaller = min(len(about), len(words)) or 1
    return shared >= DISPUTED_MIN_SHARED and shared / smaller >= DISPUTED_OVERLAP


def _stating(
    sentences: List[str],
    value: str,
    sources: Set[str],
    about: Optional[Callable[[Set[str]], bool]] = None,
) -> List[int]:
    """Indexes of sentences that state ``value``, cite one of ``sources`` and,
    given ``about``, are about the same claim (``about`` gets their content
    words). Without that last test, an unrelated sentence that happens to share
    the year and source ("his lawyer joined in 2006") would count."""
    pattern = re.compile(rf"\b{re.escape(value)}\b")
    return [
        i
        for i, s in enumerate(sentences)
        if pattern.search(_CITATION.sub("", s))
        and _cited_ids(s) & sources
        and (about is None or about(_content_words(s)))
    ]


def _keeps_both(case: Dict[str, Any], new_text: str) -> bool:
    expected, article = case["expected"], {case["article_id"]}
    sentences = _sentences(new_text)
    claim = _content_words(expected.get("profile_sentence", ""))
    # Older case files stored a grouped citation as one "a1, a2" id
    sources = {i.strip() for g in expected["original_sources"] for i in g.split(",")}
    # The old year must stay in a sentence that still makes the original claim
    kept = _stating(
        sentences,
        expected["original_value"],
        sources,
        lambda words: _best_overlap(claim, [words]) >= KEPT_OVERLAP,
    )
    if not kept:
        return False
    # The new year may be worded freshly ("another report says 2007"), so it
    # only has to share a little with either version, or sit right after the
    # kept claim, where the prompt asks for it: Haiku's "A later report gives
    # the bombing date as Oct. 12, 2001 instead." shares one word with it
    either = claim | _content_words(expected.get("article_sentence", ""))
    if _stating(
        sentences,
        expected["disputed_value"],
        article,
        lambda words: _shares_claim(either, words),
    ):
        return True
    stated = set(_stating(sentences, expected["disputed_value"], article))
    return any(i + 1 in stated for i in kept)


def score_update(case: Dict[str, Any], new_text: str) -> Dict[str, Any]:
    """Score one updated profile against its case. Pure, no model calls."""
    old_text = case["profile"].get("text", "")
    old_facts = _facts(old_text)
    new_facts = _facts(new_text)
    old_words = [_content_words(s) for s in old_facts]
    new_words = [_content_words(s) for s in new_facts]

    cited_old = [w for s, w in zip(old_facts, old_words) if _CITATION.search(s)]
    # A cited fact only counts as kept if it is still cited: the same words
    # with the source link stripped off aren't grounded any more
    cited_new = [w for s, w in zip(new_facts, new_words) if _CITATION.search(s)]
    kept = sum(1 for w in cited_old if _best_overlap(w, cited_new) >= KEPT_OVERLAP)
    new_set = {" ".join(s.split()) for s in new_facts}
    unchanged = sum(1 for s in old_facts if " ".join(s.split()) in new_set)
    added = sum(1 for w in new_words if _best_overlap(w, old_words) < KEPT_OVERLAP)
    # Any new or changed sentence counts here, even one that repeats an old
    # claim with a different value ("...in 2007 with others; sources disagree")
    # All sentences, not just facts: "Sources differ." is too short to be one
    old_set = {" ".join(s.split()) for s in _sentences(old_text)}
    changed = [s for s in _sentences(new_text) if " ".join(s.split()) not in old_set]
    disputes = sum(
        1 for s in changed if _DISPUTE_WORDS.search(s) and _SOURCE_WORDS.search(s)
    )
    # Any one of the sentence's numbers in the article is enough, since a
    # sentence that states both sides ("2003, rather than 2002") also repeats
    # the profile's own number
    in_article = _numbers(case.get("article_text", ""))
    unsupported = sum(
        1
        for s in changed
        if case["article_id"] in _cited_ids(s)
        and _numbers(s)
        and not _numbers(s) & in_article
    )

    ratio = len(new_text) / max(len(old_text), 1)
    retention = kept / len(cited_old) if cited_old else 1.0
    article_id = case["article_id"]
    known = _cited_ids(old_text) | set(case["profile"].get("sources") or [])
    new_ids = _cited_ids(new_text)
    scores: Dict[str, Any] = {
        "old_chars": len(old_text),
        "new_chars": len(new_text),
        "length_ratio": round(ratio, 3),
        # Tested before rounding: 0.79990 rounds to 0.8 and would slip through
        "collapsed": ratio < COLLAPSE_RATIO,
        "fact_retention": round(retention, 3),
        # Also tested before rounding: 188/209 = 0.8995 would round to 0.9
        "low_retention": retention < MIN_RETENTION,
        "facts_lost": len(cited_old) - kept,
        "unchanged_share": round(unchanged / len(old_facts), 3) if old_facts else 1.0,
        "added_sentences": added,
        # Added sentences saying sources disagree. Outside disputed cases most
        # are invented: Luna called "approved Sept. 28" vs "charges sworn in
        # April" a conflict. Reported, not failed, since some are real.
        "dispute_claims": disputes,
        "unsupported_numbers": unsupported,
        "cites_article": article_id in new_ids,
        "invented_citations": sorted(new_ids - known - {article_id}),
    }
    if case["kind"] == "disputed":
        scores["keeps_both"] = _keeps_both(case, new_text)
    label = case.get("claim_label")
    if label:
        scores.update(score_claims(label, old_text, new_text, article_id))
        scores["disputes_ok"] = bool(label.get("disputes_ok"))
    scores["passed"] = case_passed(case, scores)
    return scores


def case_passed(case: Dict[str, Any], scores: Dict[str, Any]) -> bool:
    expected = case.get("expected", {})
    if scores["collapsed"] or scores["low_retention"] or scores["invented_citations"]:
        return False
    if case["kind"] == "irrelevant":
        return scores["added_sentences"] <= expected.get("max_new_sentences", 2)
    if case["kind"] == "new_fact":
        if "claims_ok" in scores:  # labelled: the right facts, read the right way
            return (
                scores["cites_article"]
                and scores["claims_ok"]
                and (scores["disputes_ok"] or not scores["dispute_claims"])
            )
        return scores["cites_article"]
    if case["kind"] == "disputed":
        return scores["keeps_both"]
    return True


REVIEWS_FILE = "profile_update_reviews.jsonl"


def load_reviews(eval_dir: str) -> Dict[str, Dict[str, Any]]:
    """Hand checks of case labels, by case id (the last row for an id wins).

    One JSON row per checked case: ``{"case_id", "verdict", "note"}``, where
    ``keep`` confirms the label (a disputed case then gets scored) and ``drop``
    leaves the case out for good (two different events, a wrong label). The
    file sits apart from the cases, so ``just eval-cases`` can rebuild them.
    An optional ``pairing`` (``case_pairing()``) ties the verdict to the two
    sentences that were checked; a rebuild that pairs others voids it.
    """
    path = os.path.join(eval_dir, REVIEWS_FILE)
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {r["case_id"]: r for r in rows}


def case_pairing(case: Dict[str, Any]) -> str:
    """Short hash of the sentences a disputed label rests on, years masked.

    A rebuild can flip the planted year (2001 instead of 1999) and the verdict
    still holds, but pairing a different profile sentence needs a new check.
    """
    expected = case.get("expected", {})
    text = (
        expected.get("profile_sentence", "")
        + "\n"
        + expected.get("article_sentence", "")
    )
    return hashlib.sha256(re.sub(r"\d", "#", text).encode()).hexdigest()[:12]


def load_cases(
    eval_dir: str,
    split: str = "all",
    include_unreviewed: bool = False,
    claim_labels: str = "reviewed",
) -> List[Dict[str, Any]]:
    """Read ``profile_update_cases.jsonl``, leaving out unconfirmed labels.

    Each case with a claim label (``claim_labels``: ``reviewed`` rows only,
    ``draft`` for unreviewed ones too, ``off`` for none) gets it under
    ``claim_label``, which ``score_update()`` then checks.
    """
    path = os.path.join(eval_dir, "profile_update_cases.jsonl")
    with open(path, encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    reviews = load_reviews(eval_dir)
    kept = []
    for c in cases:
        review = reviews.get(c["case_id"], {})
        verdict = review.get("verdict")
        if review.get("pairing") not in (None, case_pairing(c)):
            verdict = None  # checked against sentences this case no longer has
        if verdict == "drop" or (split != "all" and c["split"] != split):
            continue
        if verdict == "keep":
            c = {**c, "needs_review": False}
        if include_unreviewed or not c.get("needs_review"):
            kept.append(c)
    labels = load_labels(eval_dir, claim_labels)
    for c in kept:
        label = label_for(labels, c) if c["kind"] == "new_fact" else None
        if label:
            c["claim_label"] = label
    return kept


def make_rewrite_updater(model: str, domain: str) -> Updater:
    """Run ``update_profile``'s whole-profile rewrite (with reflection) on a case."""
    from src.engine.profiles import VersionedProfile, update_profile

    def update(case: Dict[str, Any]) -> str:
        profile, _, _ = update_profile(
            case["entity_type"],
            case["entity_name"],
            case["profile"],
            VersionedProfile(),
            case["article_text"],
            case["article_id"],
            domain=domain,
            model=model,
            approach="rewrite",
        )
        # update_profile hands back the old profile when the model's answer
        # can't be parsed; scoring that would reward a failed call. A real
        # update always adds the article to ``sources`` (even when the text is
        # rightly left word for word), and the fallback never does.
        if case["article_id"] not in (profile.get("sources") or []):
            raise RuntimeError("update failed: the old profile came back")
        return profile.get("text", "")

    return update


@contextmanager
def _update_step(case: Dict[str, Any], approach: str) -> Iterator[None]:
    """The ``profile.update`` step the pipeline would open around this call."""
    from src.utils.step_timing import timed_step
    from src.utils.tracing import annotate

    with timed_step("profile.update"):
        annotate(
            entity_type=case["entity_type"],
            entity_name=case["entity_name"],
            article_id=case["article_id"],
            approach=approach,
        )
        yield


@functools.lru_cache(maxsize=4)
def _published_dates(domain: str) -> Dict[str, Any]:
    """Published date by article id, from the domain's articles Parquet ({}
    when it isn't there, which turns the later-update rule off)."""
    import pyarrow.parquet as pq

    from src.config_loader import DomainConfig

    path = DomainConfig(domain).get_data_path()
    if not os.path.exists(path):
        return {}
    table = pq.read_table(path, columns=["id", "published_date"]).to_pydict()
    return dict(zip(table["id"], table["published_date"]))


def case_dates(case: Dict[str, Any], domain: str) -> Dict[str, Any]:
    """Published dates of the case's article and the profile's sources. A
    disputed case's article is a copy of a real one with a year changed, so
    it has that article's date."""
    dates = _published_dates(domain)
    ids = list(case["profile"].get("sources") or []) + [case["article_id"]]
    found = {i: dates.get(i.removesuffix("-disputed")) for i in ids}
    return {i: d for i, d in found.items() if d is not None}


def make_edit_updater(model: str, domain: str) -> Updater:
    """Ask for a list of edits, check and apply them (``profile_edits.py``,
    ``edit_guards.py``); a profile too long for the local model goes to the
    fact list, as in the pipeline."""
    from src.engine.profile_edits import edit_profile
    from src.utils.ollama_context import PromptTooLongError

    def update(case: Dict[str, Any]) -> str:
        args = (
            case["entity_type"],
            case["entity_name"],
            case["profile"],
            case["article_text"],
            case["article_id"],
        )
        try:
            with _update_step(case, "edit"):
                profile = edit_profile(
                    *args,
                    model=model,
                    domain=domain,
                    article_dates=case_dates(case, domain),
                    kind=case.get("entity_kind") or None,
                )
        except PromptTooLongError:
            result = make_fact_updater(model, domain)(case)
            result["report"]["fallback"] = "facts"
            return result
        # The raw edits too, so a changed guard can be replayed without the model
        report = {**(profile.get("edit_report") or {}), "edits": profile.get("edits")}
        return {"text": profile["text"], "report": report}

    return update


def make_fact_updater(model: str, domain: str) -> Updater:
    """Prototype: update from a list of sourced facts (``src/engine/fact_list.py``)."""
    from src.engine.fact_list import update_from_facts

    def update(case: Dict[str, Any]) -> str:
        with _update_step(case, "facts"):
            profile = update_from_facts(
                case["entity_type"],
                case["entity_name"],
                case["profile"],
                case["article_text"],
                case["article_id"],
                model=model,
                domain=domain,
            )
        report = {**profile["fact_report"], "rows": profile["fact_rows"][-12:]}
        return {"text": profile["text"], "report": report}

    return update


UPDATERS: Dict[str, Callable[[str, str], Updater]] = {
    "rewrite": make_rewrite_updater,
    "edit": make_edit_updater,
    "facts": make_fact_updater,
}


def _median(values: List[float]) -> Optional[float]:
    values = sorted(values)
    return values[len(values) // 2] if values else None


def summarize_updates(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pass rate overall and per kind, plus the numbers behind failures.

    A failed call counts as a failed case, so a model can't score well by
    erroring on the hard ones.
    """
    scored = [r for r in records if r.get("scores")]
    summary: Dict[str, Any] = {
        "n": len(records),
        "errors": len(records) - len(scored),
        "pass_rate": round(
            sum(r["scores"]["passed"] for r in scored) / max(len(records), 1), 3
        ),
        "collapses": sum(r["scores"]["collapsed"] for r in scored),
        "median_retention": _median([r["scores"]["fact_retention"] for r in scored]),
        "median_unchanged": _median([r["scores"]["unchanged_share"] for r in scored]),
        "invented_citations": sum(
            bool(r["scores"]["invented_citations"]) for r in scored
        ),
        # Only where no dispute was planted, so these are likely invented
        "dispute_claims": sum(
            r["scores"].get("dispute_claims", 0)
            for r in scored
            if r["kind"] != "disputed"
        ),
        "unsupported_numbers": sum(
            r["scores"].get("unsupported_numbers", 0) for r in scored
        ),
        # Cases with a claim label, the expected facts they missed and the
        # wrong readings (wrong actor, reversed meaning) they added
        "labelled": sum("claims_ok" in r["scores"] for r in scored),
        "core_missing": sum(len(r["scores"].get("core_missing", [])) for r in scored),
        "wrong_readings": sum(
            len(r["scores"].get("must_not_hits", [])) for r in scored
        ),
        "median_seconds": _median([r["seconds"] for r in records]),
    }
    for kind in sorted({r["kind"] for r in records}):
        of_kind = [r for r in records if r["kind"] == kind]
        passed = sum(1 for r in of_kind if r.get("scores", {}).get("passed"))
        summary[f"pass_{kind}"] = f"{passed}/{len(of_kind)}"
    return summary


def run_updates(
    cases: List[Dict[str, Any]], updater: Updater, workers: int = 4
) -> List[Dict[str, Any]]:
    """Update and score every case in parallel, keeping the input order.

    Each record keeps the new profile text, so a later judge pass (or a
    person) can look at exactly what was scored.
    """
    from concurrent.futures import ThreadPoolExecutor
    from time import perf_counter

    from src.utils.tracing import annotate, item_span

    def one(case: Dict[str, Any]) -> Dict[str, Any]:
        with item_span(
            f"eval {case['case_id']}",
            eval_item_id=case["case_id"],
            eval_kind=case["kind"],
            eval_split=case["split"],
        ):
            record = score_one(case)
            scores = record.get("scores") or {}
            annotate(
                eval_passed=scores.get("passed"),
                eval_scores=scores or None,
                eval_error=record.get("error"),
            )
            return record

    def score_one(case: Dict[str, Any]) -> Dict[str, Any]:
        start = perf_counter()
        record: Dict[str, Any] = {
            "item_id": case["case_id"],
            "kind": case["kind"],
            "split": case["split"],
        }
        try:
            result = updater(case)
            # An updater may also hand back a report of what it did
            new_text, report = (
                (result["text"], result.get("report"))
                if isinstance(result, dict)
                else (result, None)
            )
            record.update(new_text=new_text, scores=score_update(case, new_text))
            if report:
                record["report"] = report
        except Exception as e:  # noqa: BLE001 — one failed call must not stop the run
            record["error"] = f"{type(e).__name__}: {e}"
        record["seconds"] = round(perf_counter() - start, 3)
        return record

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(one, cases))


def rescore_run(
    records: List[Dict[str, Any]], cases: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Score a saved run's profiles again with today's checks, no model calls.

    Records whose case is no longer loaded (dropped, other split) are left out;
    failed calls stay failed.
    """
    by_id = {c["case_id"]: c for c in cases}
    out = []
    for r in records:
        case = by_id.get(r.get("item_id"))
        if case is None:
            continue
        if "new_text" in r:
            r = {**r, "scores": score_update(case, r["new_text"])}
        out.append(r)
    return out
