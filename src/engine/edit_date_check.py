"""The fact-list updater's date check, run after the edit-list updater's edits.

Why: in a profile update the model is asked to list where the article and the
profile give the same thing different dates, and it misses a date the article
gives in passing. On the profile-update eval's hard cases ("the Pentagon
prison system that opened in January 2003" against a profile's January 2002;
"the October 2001 bombing" against October 2000), Luna listed no disagreement
in any run, with both sentences in the prompt. Asking harder in the prompt
didn't help and cost new facts elsewhere.

``date_check_edits()`` does it the other way round, with the code the
fact-list updater already has (``date_disputes.py``):

1. **Pair** (code): each article sentence with a year, beside the profile's
   cited sentences that give that thing another year (``date_pairs()``),
   leaving out the sentences the model itself listed as disagreements;
2. **Judge** (one model call, ``date_check.md``): the same past event given
   two dates (``conflict``), two events, or something that changed over time;
3. **Write** (code): for each ``conflict``, an ``add`` edit right after the
   old sentence, "Another report differs: ...^[article]", so both dates stay
   with their own sources.

On by default for every model (``HINBOX_EDIT_DATE_CHECK``: ``on`` by default,
``cloud`` for cloud models only, ``off``). Over 276 eval cases on
Luna (dev, train and test splits and two Kitaru cohorts, 15 Nov) it wrote
one wrong dispute (Spath: "Nashiri was moved in 2004", out of the CIA site,
paired with "began his military custody at Guantánamo in September 2006"),
found two real errors in stored profiles (Khadr's plea year, Abbasi's
release) and passed 5 of 8 planted disputes on train and test against 2.
The fact-list updater's version had written 1-3 wrong disputes per 160
cases. On local Qwen (the same 242 dev, train and test cases, 15 Nov) it
first wrote 2 wrong disputes of 9: the USS Cole bombing paired with the
earlier attack on The Sullivans, and an article's "him" (another detainee)
rewritten as Nashiri, a false claim in his profile. With the pronoun rule and
the name guard in ``date_disputes.py`` it passed 8 of 13 planted disputes
against 5 with the check off and wrote 6 disputes, all right (Qwen called the
Cole pair ``different`` this time), so it is on for local models too.
"""

import os
import re
from typing import Any, Dict, Iterable, List, Tuple

EDIT_DATE_CHECK_ENV = "HINBOX_EDIT_DATE_CHECK"
_CITATION = re.compile(r"\^\[[^\]]+\]")


def _is_dispute(sentence: str) -> bool:
    """This check's own dispute ("Another report differs: ..."), or the edit
    model's: words saying sources differ beside a word naming a source ("A
    later report gives the year as 2005 instead", "the sources disagree").
    "His lawyers instead sought a civilian trial" is no dispute."""
    from src.engine.date_disputes import DISPUTE_PREFIX
    from src.engine.edit_guards import _DISPUTE, _SOURCES

    if sentence.startswith(DISPUTE_PREFIX.strip()):
        return True
    differs = _DISPUTE.search(sentence) or re.search(r"\binstead\b", sentence, re.I)
    return bool(differs and _SOURCES.search(sentence))


def edit_date_check_enabled(local: bool) -> bool:
    """On unless ``HINBOX_EDIT_DATE_CHECK=off``; with ``cloud``, on for cloud
    models only."""
    value = os.getenv(EDIT_DATE_CHECK_ENV, "on").strip().lower()
    if value == "cloud":
        return not local
    return value not in {"off", "0", "no"}


def _norm(sentence: str) -> str:
    return " ".join(_CITATION.sub("", sentence).split())


def date_check_edits(
    generate: Any,
    model: str,
    domain: str,
    entity_type: str,
    entity_name: str,
    old_text: str,
    article_text: str,
    article_id: str,
    disputed: Iterable[str] = (),
) -> Tuple[List[Any], Dict[str, Any]]:
    """The ``add`` edits for the date conflicts the model's own list missed,
    and a report (pairs found, verdicts, the sentences written).

    ``disputed``: profile sentences the model already listed as disagreements.
    Sentences already followed by a dispute sentence are skipped too.
    """
    from src.engine.date_disputes import find_date_disputes
    from src.engine.fact_list import _call, _prompt
    from src.engine.profile_edits import ProfileEdit
    from src.evals.profile_cases import _sentences
    from src.utils.step_timing import timed_step

    sentences = _sentences(old_text)
    rows = [
        {"id": str(i), "claim": _norm(s), "text": s}
        for i, s in enumerate(sentences)
        if _CITATION.search(s)
    ]
    done = {_norm(s) for s in disputed}
    # A sentence disputed before is left alone, so a later article giving the
    # same other date doesn't add a second "Another report differs"
    already = {
        str(i)
        for i in range(len(sentences) - 1)
        if _is_dispute(_norm(sentences[i + 1]))
    }
    skip = {r["id"] for r in rows if r["claim"] in done or r["id"] in already}

    def ask(name: str, user: str, schema: Any) -> Any:
        with timed_step(f"profile.{name}"):
            system = _prompt(
                domain, name, model, entity_type=entity_type, entity_name=entity_name
            )
            return _call(generate, model, system, user, schema)

    disputes, report = find_date_disputes(ask, rows, article_text, article_id, skip)
    report["date_dispute_texts"] = [d["text"] for d in disputes]
    edits = [
        ProfileEdit(op="add", anchor=d["row"]["text"], text=d["text"]) for d in disputes
    ]
    return edits, report
