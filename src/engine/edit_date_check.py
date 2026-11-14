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

Off by default, as in the fact-list updater, where it wrote 1-3 wrong disputes
per 160 eval cases on Luna: ``HINBOX_EDIT_DATE_CHECK=on`` turns it on.
"""

import os
import re
from typing import Any, Dict, Iterable, List, Tuple

EDIT_DATE_CHECK_ENV = "HINBOX_EDIT_DATE_CHECK"
_CITATION = re.compile(r"\^\[[^\]]+\]")


def edit_date_check_enabled() -> bool:
    return os.getenv(EDIT_DATE_CHECK_ENV, "off").strip().lower() in {"on", "1", "yes"}


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
    """
    from src.engine.date_disputes import find_date_disputes
    from src.engine.fact_list import _call, _prompt
    from src.engine.profile_edits import ProfileEdit
    from src.evals.profile_cases import _sentences
    from src.utils.step_timing import timed_step

    rows = [
        {"id": str(i), "claim": _norm(s), "text": s}
        for i, s in enumerate(_sentences(old_text))
        if _CITATION.search(s)
    ]
    done = {_norm(s) for s in disputed}
    skip = {r["id"] for r in rows if r["claim"] in done}

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
