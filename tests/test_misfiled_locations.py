"""Locations that are really people or organizations are dropped."""

from src.utils.misfiled_locations import (
    drop_misfiled_locations,
    drop_misfiled_organizations,
)

ORG_GROUPS = [
    ["Central Intelligence Agency", "CIA"],
    ["Federal Bureau of Investigation", "FBI"],
]


def _drop(locations, people=(), orgs=()):
    entities = {
        "people": [{"name": n} for n in people],
        "organizations": [{"name": n} for n in orgs],
        "locations": locations,
    }
    kept, dropped = drop_misfiled_locations(entities, ORG_GROUPS)
    return [k["name"] for k in kept], {d["name"]: d["reason"] for d in dropped}


def test_agencies_people_and_offices_are_dropped():
    kept, dropped = _drop(
        [
            {"name": "CIA", "type": "other"},
            {"name": "American Civil Liberties Union", "type": "other"},
            {"name": "Donald Trump", "type": "other"},
            {"name": "Department of Homeland Security", "type": "building"},
            {"name": "Leon County Sheriff's Office", "type": "other"},
            {"name": "Miami", "type": "city"},
        ],
        people=["Donald Trump"],
        orgs=["American Civil Liberties Union"],
    )
    assert kept == ["Miami"]
    assert dropped["Donald Trump"] == "person"
    assert dropped["CIA"] == "organization"
    assert dropped["Department of Homeland Security"] == "organization name"


def test_real_places_sharing_a_name_are_kept():
    kept, _ = _drop(
        [
            {"name": "Pentagon", "type": "building"},  # also the DoD
            {"name": "Fort Lauderdale", "type": "city"},  # also a school team
            {"name": "Washington", "type": "city"},  # also a surname
            {
                "name": "Florida International University",
                "type": "LocationType.BUILDING",
            },
        ],
        people=["Washington"],
        orgs=["Pentagon", "Fort Lauderdale", "Florida International University"],
    )
    assert kept == [
        "Pentagon",
        "Fort Lauderdale",
        "Washington",
        "Florida International University",
    ]


def test_org_word_in_a_camp_name_is_kept():
    kept, _ = _drop(
        [{"name": "Camp Justice Administration Area", "type": "detention_facility"}]
    )
    assert kept == ["Camp Justice Administration Area"]


def test_institution_words_in_areas_and_structures_are_kept():
    kept, _ = _drop(
        [
            {"name": "Bureau County", "type": "region"},
            {"name": "Administration Building", "type": "building"},
            {"name": "Camp Justice Administration Area", "type": "detention_facility"},
        ]
    )
    assert len(kept) == 3


def test_nationalities_media_laws_and_dates_are_dropped():
    names = {
        "Yemeni": "country",
        "Algerians": "other",
        "Miami Herald": "building",
        "The New York Times": "other",
        "Voice of America": "other",
        "Intercept": "other",
        "NDAA": "other",
        "2001 Authorization for the Use of Military Force": "other",
        "March 2008": "other",
    }
    kept, dropped = _drop([{"name": n, "type": t} for n, t in names.items()])
    assert kept == []
    assert dropped["Yemeni"] == "nationality"
    assert dropped["Miami Herald"] == "media"
    assert dropped["NDAA"] == "law"
    assert dropped["March 2008"] == "date"


def test_places_that_look_like_non_places_are_kept():
    names = [
        "Indian Ocean",
        "Dominican Republic",
        "Times Square",
        "Afghanistan",
        "New York Times Building",
        "Camp 5",
        "Jordan",
    ]
    kept, _ = _drop([{"name": n, "type": "other"} for n in names])
    assert kept == names


def test_organizations_that_are_people_are_dropped():
    entities = {
        "people": [{"name": "Donald Trump", "aliases": ["Trump"]}],
        "organizations": [
            {"name": "President Donald Trump", "type": "government"},
            {"name": "Donald J. Trump", "type": "other"},
            {"name": "Trump", "type": "other"},
            {"name": "Trump", "type": "government"},  # one word, typed: kept
            {"name": "Trump administration", "type": "government"},
            {"name": "ACLU", "type": "legal"},
        ],
    }
    kept, dropped = drop_misfiled_organizations(entities)
    assert [o["name"] for o in kept] == ["Trump", "Trump administration", "ACLU"]
    assert [d["reason"] for d in dropped] == ["person"] * 3


def test_organizations_kept_when_no_people_came_back():
    entities = {"organizations": [{"name": "Donald Trump", "type": "other"}]}
    kept, dropped = drop_misfiled_organizations(entities)
    assert len(kept) == 1 and not dropped


def test_law_words_inside_real_place_names_are_kept():
    kept, _ = _drop(
        [
            {"name": "Treaty Room", "type": "building"},
            {"name": "ACT", "type": "region"},
        ]
    )
    assert kept == ["Treaty Room", "ACT"]


def test_organizations_that_are_countries_or_camps_are_dropped():
    entities = {
        "locations": [
            {"name": "France", "type": "country"},
            {"name": "Camp 7", "type": "detention_facility"},
        ],
        "organizations": [
            {"name": "France", "type": "government"},
            {"name": "Israel", "type": "government"},  # no country location: kept
            {"name": "French government", "type": "government"},
            {"name": "Camp 7", "type": "military"},
            {"name": "Camp Justice", "type": "military"},
            {"name": "Guantanamo Bay Navy Base", "type": "military"},
            {"name": "Guantánamo detention center", "type": "military"},
            {"name": "Joint Task Force-GTMO", "type": "military"},
            {"name": "U.S. Southern Command", "type": "military"},
            {"name": "Prison Legal News", "type": "media"},
            {"name": "Camp", "type": "other"},  # one word: not a named camp
        ],
    }
    kept, dropped = drop_misfiled_organizations(entities)
    assert [o["name"] for o in kept] == [
        "Israel",
        "French government",
        "Joint Task Force-GTMO",
        "U.S. Southern Command",
        "Prison Legal News",
        "Camp",
    ]
    assert {d["name"]: d["reason"] for d in dropped} == {
        "France": "country",
        "Camp 7": "facility",
        "Camp Justice": "facility",
        "Guantanamo Bay Navy Base": "facility",
        "Guantánamo detention center": "facility",
    }
