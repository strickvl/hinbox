"""Near-name duplicates left after the 74-article run (1 Nov 2026), and
garbled citation IDs in profiles."""

import json

import pytest

from src.engine import EntityMerger
from src.engine.merge_candidates import name_variant_candidate
from src.utils.citation_ids import citation_repairs, repair_profile_citations
from src.utils.name_variants import lone_name_in_full_name, names_are_variants
from tests.test_entity_merger_merge_smoke import make_empty_entities


@pytest.mark.parametrize(
    "a, b, entity_type",
    [
        ("Bush administration", "George W. Bush administration", "organizations"),
        ("Trump administration", "Donald J. Trump administration", "organizations"),
        ("Hadi al Iraqi", "Abd al Hadi al Iraqi", "people"),
        ("Khalid Sheikh Mohammed", "Khalid Shaikh Muhammad", "people"),
        (
            "CIA waterboarding of Khalid Shaikh Mohammed",
            "Waterboarding of Khalid Sheik Mohammed",
            "events",
        ),
        (
            "Charging of Abd al Hadi al Iraqi",
            "Charging of Hadi al Iraqi",
            "events",
        ),
    ],
)
def test_run_duplicates_are_variants(a, b, entity_type):
    assert names_are_variants(a, b, entity_type=entity_type)


@pytest.mark.parametrize(
    "a, b, entity_type",
    [
        ("Bush administration", "Obama administration", "organizations"),
        ("Bush administration", "Bush campaign", "organizations"),
        ("Navy base", "Guantanamo Bay Navy Base", "organizations"),
        ("al Iraqi", "Abd al Hadi al Iraqi", "people"),  # particle + one word
        ("Lara", "Lara Tur", "people"),  # only a weak match, see below
        (
            "Charging of Abd al Hadi al Iraqi",
            "Arraignment of Abd al Hadi al Iraqi",
            "events",
        ),
        # Only the last word in common
        ("Hearing on torture", "Senate hearing on CIA torture", "events"),
    ],
)
def test_look_alikes_stay_apart(a, b, entity_type):
    assert not names_are_variants(a, b, entity_type=entity_type)


@pytest.mark.parametrize(
    "a, b, expected",
    [
        ("Lara", "Lara Tur", True),
        ("Hadi", "Abd al Hadi al Iraqi", True),
        ("Tur", "Lara Tur", False),  # a surname is already a full variant
        ("Al", "Abd al Hadi al Iraqi", False),
        ("Abd", "Abd al Hadi al Iraqi", False),
        ("Lara Tur", "Lara Smith", False),
    ],
)
def test_lone_name_in_full_name(a, b, expected):
    assert lone_name_in_full_name(a, b) is expected


def _person(vec):
    return {"profile_embedding": vec, "profile": {"text": "p"}}


class TestLoneNameCandidate:
    def test_only_person_with_that_first_name_reaches_match_check(self):
        """'Lara' scores 0.3 against 'Lara Tur', below the 0.82 cutoff, and isn't
        a surname; she's the only Lara, so the match check sees the pair."""
        entities = make_empty_entities()
        entities["people"] = {
            "Lara Tur": _person([0.3, 0.954, 0]),
            "Marcos Duro": _person([0.6, 0.8, 0]),
        }
        key, _ = EntityMerger("people").find_similar_entity(
            "Lara", [1.0, 0.0, 0.0], entities, 0.82
        )
        assert key == "Lara Tur"

    def test_two_people_with_that_first_name_send_neither(self):
        entities = make_empty_entities()
        entities["people"] = {
            "Daniel Harding": _person([0.3, 0.954, 0]),
            "Daniel Ellsberg": _person([0.2, 0.98, 0]),
        }
        key, _ = name_variant_candidate(
            "people", "alternative_names", "Daniel", [1.0, 0, 0], entities
        )
        assert key is None

    def test_full_variant_wins_over_lone_name(self):
        entities = make_empty_entities()
        entities["people"] = {
            "Hadi al Iraqi": _person([0.2, 0.98, 0]),
            "Hadi Saleh": _person([0.9, 0.436, 0]),
        }
        key, _ = name_variant_candidate(
            "people",
            "alternative_names",
            "Abd al Hadi al Iraqi",
            [1.0, 0, 0],
            entities,
        )
        assert key == "Hadi al Iraqi"

    def test_rejected_lone_name_candidate_is_not_retried(self):
        entities = make_empty_entities()
        entities["people"] = {"Lara Tur": _person([0.3, 0.954, 0])}
        key, _ = name_variant_candidate(
            "people",
            "alternative_names",
            "Lara",
            [1.0, 0, 0],
            entities,
            exclude=("Lara Tur",),
        )
        assert key is None


REAL = "62cdb2ad-18d6-4de6-a599-a5858c1fb8ac"
OTHER = "0b7e4f10-9c1d-4a2b-8e3f-5d6c7b8a9f01"


class TestCitationRepair:
    @pytest.mark.parametrize(
        "garbled",
        [
            "62cdb2ad-18d6-4de6-a5858c1fb8ac",  # chunk dropped
            "62cdb2ad-18d6-4de6-a599-a599-a5858c1fb8ac",  # chunk repeated
        ],
    )
    def test_near_miss_snaps_to_the_source(self, garbled):
        assert citation_repairs(f"He was held.^[{garbled}]", [REAL, OTHER]) == {
            garbled: REAL
        }

    def test_known_and_far_ids_are_left_alone(self):
        text = f"A.^[{REAL}] B.^[ffffffff-0000-1111-2222-333333333333]"
        assert citation_repairs(text, [REAL, OTHER]) == {}

    def test_profile_text_and_saved_fact_rows_are_fixed(self):
        bad = "62cdb2ad-18d6-4de6-a5858c1fb8ac"
        sentence = f"He was held at Camp 7.^[{bad}, {OTHER}]"
        profile = {
            "text": f"## Background\n{sentence}",
            "sources": [REAL, OTHER],
            "fact_rows_json": json.dumps(
                [{"text": sentence, "sources": [{"article_id": bad}]}]
            ),
        }
        fixed, count = repair_profile_citations(profile)
        assert count == 1
        assert fixed["text"].endswith(f"^[{REAL}, {OTHER}]")
        rows = json.loads(fixed["fact_rows_json"])
        assert rows[0]["text"] in fixed["text"]
        assert rows[0]["sources"] == [{"article_id": REAL}]

    def test_profile_without_sources_is_unchanged(self):
        profile = {"text": "X.^[abc-123]"}
        assert repair_profile_citations(profile) == (profile, 0)


@pytest.mark.parametrize(
    "a, b",
    [
        # Words added in two places: a different event
        ("Easing of lockdown conditions in Camp 6", "Lockdown of Camp 6"),
        (
            "Scheduled pre-trial hearing for Abd al Rahim al Nashiri",
            "Scheduled trial for Abd al Rahim al Nashiri",
        ),
    ],
)
def test_events_with_words_added_in_two_places_stay_apart(a, b):
    assert not names_are_variants(a, b, entity_type="events")
