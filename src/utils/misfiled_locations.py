"""Drop extracted locations and organizations that are something else.

Small local models put agencies and people into the locations list (typed
"other" or "building"): CIA, FBI, ACLU, Donald Trump. The four entity types
are extracted separately, so a location whose name also came back as a person
or organization from the same article, or that names a configured
organization, is dropped here, and so is a nationality word, news outlet, law
or date (``non_place_names``). An organization whose name is a person from
the same article ("President Donald Trump") is dropped too. No LLM calls.
"""

from typing import Any, Dict, FrozenSet, List, Set, Tuple

from src.utils.name_variants import normalize_for_match, person_name_tokens
from src.utils.non_place_names import non_place_reason

# A name shared with a one-word person or with an organization only counts
# against types like these: "Fort Lauderdale" is a city and a school team,
# "Washington" is a city and a surname, "Pentagon" is a building and the DoD.
_LOOSE_TYPES: FrozenSet[str] = frozenset({"other", ""})
# Words that make a name an institution ("Department of Justice") when it is
# typed "other" or "building". Areas keep them ("Bureau County"), and so do
# names ending in a structure word ("Administration Building").
_ORG_WORDS: FrozenSet[str] = frozenset(
    {"department", "agency", "bureau", "enforcement", "administration"}
    | {"commission", "committee", "division", "association", "sheriff"}
)
_STRUCTURE_WORDS: FrozenSet[str] = frozenset(
    {"building", "center", "centre", "hall", "house", "tower", "courthouse"}
)


def _names(items: List[Dict[str, Any]]) -> Set[str]:
    out: Set[str] = set()
    for item in items:
        for name in [item.get("name", ""), *item.get("aliases", [])]:
            if isinstance(name, str) and name.strip():
                out.add(normalize_for_match(name))
    return out


def misfiled_reason(
    location: Dict[str, Any], people: Set[str], organizations: Set[str]
) -> str:
    """Why this location is really a person or organization ("" if it isn't)."""
    name = normalize_for_match(location.get("name", ""))
    if not name:
        return ""
    non_place = non_place_reason(name)
    if non_place:
        return non_place
    loc_type = str(location.get("type") or "").split(".")[-1].lower()
    words = name.split()
    if name in people and (len(words) > 1 or loc_type in _LOOSE_TYPES):
        return "person"
    if (
        _ORG_WORDS & set(words)
        and loc_type in _LOOSE_TYPES | {"building"}
        and words[-1] not in _STRUCTURE_WORDS
    ):
        return "organization name"
    if name in organizations and loc_type in _LOOSE_TYPES:
        return "organization"
    return ""


def drop_misfiled_locations(
    entities: Dict[str, List[Dict[str, Any]]],
    org_equivalence_groups: List[List[str]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    """Return (kept locations, [{"name", "reason"}] for each one dropped)."""
    people = _names(entities.get("people", []))
    orgs = _names(entities.get("organizations", []))
    orgs |= {normalize_for_match(n) for group in org_equivalence_groups for n in group}
    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, str]] = []
    for loc in entities.get("locations", []):
        reason = misfiled_reason(loc, people, orgs)
        if reason:
            dropped.append({"name": str(loc.get("name", "")), "reason": reason})
        else:
            kept.append(loc)
    return kept, dropped


def drop_misfiled_organizations(
    entities: Dict[str, List[Dict[str, Any]]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    """Drop organizations that are a person from the same article.

    "President Donald Trump" goes when "Donald Trump" came back as a person
    (titles and initials ignored). A one-word name ("Trump") only goes when it
    is typed "other"; "Trump administration" always stays.
    """
    people = {
        tuple(person_name_tokens(name))
        for person in entities.get("people", [])
        for name in [person.get("name", ""), *person.get("aliases", [])]
        if isinstance(name, str) and name.strip()
    }
    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, str]] = []
    for org in entities.get("organizations", []):
        tokens = tuple(person_name_tokens(str(org.get("name", ""))))
        org_type = str(org.get("type") or "").split(".")[-1].lower()
        if tokens in people and (len(tokens) > 1 or org_type in _LOOSE_TYPES):
            dropped.append({"name": str(org.get("name", "")), "reason": "person"})
        else:
            kept.append(org)
    return kept, dropped
