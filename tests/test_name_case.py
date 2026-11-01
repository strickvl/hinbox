"""ALL CAPS names (copied from datelines) get the article's own case."""

from src.utils.name_case import (
    recase_entity_names,
    recase_from_article,
    title_case_place,
)

DATELINE = "GUANTÁNAMO BAY NAVY BASE, Cuba — "


def test_uses_the_articles_normal_case_spelling():
    text = DATELINE + "Guards at the Guantánamo Bay Navy Base said..."
    assert recase_from_article("GUANTANAMO BAY NAVY BASE", text) == (
        "Guantánamo Bay Navy Base"
    )


def test_keeps_the_name_when_the_article_only_shouts_it():
    assert recase_from_article("GUANTANAMO BAY NAVY BASE", DATELINE + "x") == (
        "GUANTANAMO BAY NAVY BASE"
    )


def test_acronyms_and_normal_names_are_untouched():
    assert recase_from_article("FBI", "the Fbi said") == "FBI"
    assert recase_from_article("JTF GTMO", "Jtf Gtmo") == "JTF GTMO"
    assert recase_from_article("Camp Delta", "CAMP DELTA") == "Camp Delta"


def test_title_case_place():
    assert title_case_place("GUANTANAMO BAY NAVY BASE") == "Guantanamo Bay Navy Base"
    assert title_case_place("U.S. NAVAL STATION GUANTANAMO BAY") == (
        "U.S. Naval Station Guantanamo Bay"
    )
    assert title_case_place("Miami") == "Miami"


def test_only_locations_fall_back_to_title_case():
    entities = {
        "locations": [{"name": "GUANTANAMO BAY NAVY BASE"}],
        "organizations": [{"name": "AMERICAN CIVIL LIBERTIES UNION"}],
        "events": [{"title": "HUNGER STRIKE BEGINS"}],
    }
    changed = recase_entity_names(entities, DATELINE)
    assert entities["locations"][0]["name"] == "Guantanamo Bay Navy Base"
    assert entities["organizations"][0]["name"] == "AMERICAN CIVIL LIBERTIES UNION"
    assert changed == [
        {"from": "GUANTANAMO BAY NAVY BASE", "to": "Guantanamo Bay Navy Base"}
    ]
