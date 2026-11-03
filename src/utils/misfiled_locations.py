"""Drop extracted locations and organizations that are something else.

Small local models put agencies and people into the locations list (typed
"other" or "building"): CIA, FBI, ACLU, Donald Trump. The four entity types
are extracted separately, so a location whose name also came back as a person
or organization from the same article, or that names a configured
organization, is dropped here, and so is a nationality word, news outlet, law
or date (``non_place_names``). An organization whose name is a person from
the same article ("President Donald Trump"), a country that came back as a
country location from the same article ("France", typed "government" when an
article lists countries), or a prison camp or base ("Camp 7", "Guantanamo Bay
Navy Base") is dropped too. No LLM calls.
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
    {"building", "center", "centre", "hall", "house", "tower", "courthouse", "room"}
)
# Location types where a law word is part of a real place name ("ACT" the
# Australian Capital Territory, "Treaty Room").
_PLACE_TYPES: FrozenSet[str] = frozenset({"city", "region", "country"})


# Organizations named like a facility: "Camp 7", "Guantanamo Bay Navy Base",
# "Guantánamo detention center". The camp or base belongs in locations; the
# people running it are "Joint Task Force Guantanamo" or "U.S. Southern Command".
_FACILITY_FIRST: FrozenSet[str] = frozenset({"camp"})
_FACILITY_LAST: FrozenSet[str] = frozenset({"base", "prison"})
_FACILITY_PHRASES: Tuple[str, ...] = (
    "detention center",
    "detention facility",
    "naval station",
)


def _is_facility_name(name: str) -> bool:
    """Whether a ``normalize_for_match`` name ("camp 7") names a camp or base."""
    words = name.split()
    return bool(words) and (
        (words[0] in _FACILITY_FIRST and len(words) > 1)
        or words[-1] in _FACILITY_LAST
        or any(p in name for p in _FACILITY_PHRASES)
    )


# Types a facility org gets ("military" for Camp 7); a "legal" or "media" org
# with one of these words is left alone.
_FACILITY_ORG_TYPES: FrozenSet[str] = (
    frozenset({"military", "government"}) | _LOOSE_TYPES
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
    loc_type = str(location.get("type") or "").split(".")[-1].lower()
    words = name.split()
    non_place = non_place_reason(name)
    law_in_place_name = non_place == "law" and (
        loc_type in _PLACE_TYPES or words[-1] in _STRUCTURE_WORDS
    )
    if non_place and not law_in_place_name:
        return non_place
    if name in people and (len(words) > 1 or loc_type in _LOOSE_TYPES):
        return "person"
    if (
        _ORG_WORDS & set(words)
        and loc_type in _LOOSE_TYPES | {"building"}
        and words[-1] not in _STRUCTURE_WORDS
    ):
        return "organization name"
    # "Camp 7" also extracted as a military org stays a location: the org copy
    # is the one drop_misfiled_organizations() removes
    if (
        name in organizations
        and loc_type in _LOOSE_TYPES
        and not _is_facility_name(name)
    ):
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


def _place_reason(
    org: Dict[str, Any], org_type: str, location_types: Dict[str, str]
) -> str:
    """Why this organization is really a place ("" if it isn't)."""
    name = normalize_for_match(str(org.get("name", "")))
    words = name.split()
    if not words:
        return ""
    if location_types.get(name) == "country":
        return "country"
    if _is_facility_name(name) and org_type in _FACILITY_ORG_TYPES:
        return "facility"
    return ""


def drop_misfiled_organizations(
    entities: Dict[str, List[Dict[str, Any]]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    """Drop organizations that are a person or a place.

    "President Donald Trump" goes when "Donald Trump" came back as a person
    (titles and initials ignored). A one-word name ("Trump") only goes when it
    is typed "other"; "Trump administration" always stays. "France" goes when
    the same article's locations hold "France" typed "country" ("French
    government" stays), and "Camp 7" or "Guantanamo Bay Navy Base" goes when
    typed "military", "government" or "other".
    """
    location_types = {
        normalize_for_match(str(loc.get("name", ""))): str(loc.get("type") or "")
        .split(".")[-1]
        .lower()
        for loc in entities.get("locations", [])
    }
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
        elif reason := _place_reason(org, org_type, location_types):
            dropped.append({"name": str(org.get("name", "")), "reason": reason})
        else:
            kept.append(org)
    return kept, dropped
