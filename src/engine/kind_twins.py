"""Join places and organizations a run saved twice under two kind spellings.

Before ``plain_kind()``, a type reached the merger as an Enum, as
"LocationType.CITY" (extraction cache) or as "city", so the same place could be
created twice. In the 12 Nov full run that left 112 pairs, "Pentagon" among
them (251 and 156 sources). A pair is joined the way the pipeline would have
built one page: the row with more sources is kept, and each article only the
other row has is fed to it in date order with the pipeline's own profile update
(``update_or_fill()``; a country or region is only linked, no model call). If
any update fails, both rows stay as they were.
"""

import copy
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.constants import ENTITY_SINGULAR
from src.engine.corrections_pass import correction_scope
from src.engine.later_updates import article_dates
from src.engine.profile_stubs import is_stub, update_or_fill
from src.engine.profiles import VersionedProfile
from src.utils.entity_keys import plain_kind
from src.utils.task_models import task_model

TWIN_TABLES = ("locations", "organizations")


def _sources(row: Dict[str, Any]) -> int:
    return len({a.get("article_id") for a in row.get("articles") or []})


def _rank(row: Dict[str, Any]) -> Tuple[int, int, int]:
    text = (row.get("profile") or {}).get("text") or ""
    return (_sources(row), int(not is_stub(row.get("profile") or {})), len(text))


def find_twins(rows: List[Dict[str, Any]]) -> List[Tuple[int, List[int]]]:
    """(keeper index, donor indexes) for each name and kind saved 2+ times.

    The keeper has the most sources, then is not a stub, then is longest.
    """
    groups: Dict[Tuple[str, Any], List[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        groups[(row.get("name") or "", plain_kind(row.get("type")))].append(i)
    twins = []
    for idx in groups.values():
        if len(idx) > 1:
            order = sorted(idx, key=lambda i: _rank(rows[i]), reverse=True)
            twins.append((order[0], order[1:]))
    return twins


def donor_articles(keeper: Dict[str, Any], donor: Dict[str, Any]) -> List[Dict]:
    """The donor's linked articles the keeper lacks, oldest first."""
    have = {a.get("article_id") for a in keeper.get("articles") or []}
    new = [a for a in donor.get("articles") or [] if a.get("article_id") not in have]
    return sorted(new, key=lambda a: str(a.get("article_published_date") or ""))


def _alternatives(keeper: Dict[str, Any], donor: Dict[str, Any]) -> List[Any]:
    names = list(keeper.get("alternative_names") or [])
    for entry in donor.get("alternative_names") or []:
        if entry not in names:
            names.append(entry)
    return names


def join_pair(
    table: str,
    keeper: Dict[str, Any],
    donor: Dict[str, Any],
    article_texts: Dict[str, str],
    model_type: str,
    domain: str,
    update: Callable = update_or_fill,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """The keeper with the donor's articles fed in, or None if a step failed.

    The keeper passed in is never changed.
    """
    work = copy.deepcopy(keeper)
    work["type"] = plain_kind(work.get("type"))
    singular, name = ENTITY_SINGULAR[table], work.get("name") or ""
    report: Dict[str, Any] = {"name": name, "type": work["type"], "updated": 0}
    for entry in donor_articles(keeper, donor):
        article_id = entry.get("article_id")
        text = article_texts.get(article_id)
        if not text:
            return None, {**report, "failed": f"no text for article {article_id}"}
        versions = work.get("profile_versions")
        versioned = VersionedProfile(**versions) if versions else VersionedProfile()
        dates = article_dates(work, article_id, entry.get("article_published_date"))
        with correction_scope(table, work):
            profile, versioned, history = update(
                singular,
                name,
                work["profile"],
                versioned,
                text,
                article_id,
                model_type,
                domain,
                article_texts=article_texts,
                model=task_model("profile", model_type),
                article_dates=dates,
                entity=work,
            )
        if article_id not in (profile.get("sources") or []):
            return None, {**report, "failed": f"profile update for {article_id}"}
        work["profile"], work["profile_versions"] = profile, versioned.model_dump()
        work.setdefault("articles", []).append(entry)
        work.setdefault("reflection_history", []).extend(history or [])
        report["updated"] += 1
    work["alternative_names"] = _alternatives(work, donor)
    return work, report
