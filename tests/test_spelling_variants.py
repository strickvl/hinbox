"""Spelling variants found in the 1 Nov 2026 run, and look-alikes that must
stay apart. A variant only reaches the match check; it never merges alone."""

import pytest

from src.engine.merge_candidates import pick_canonical_key
from src.utils.name_variants import (
    is_shouting_name,
    names_are_variants,
    score_canonical_name,
)

GROUPS = [
    ["International Committee of the Red Cross", "Red Cross", "International Red Cross"]
]


@pytest.mark.parametrize(
    "a, b, entity_type",
    [
        ("Clay Trivett", "Clayton Trivett", "people"),
        ("Rosa Eliades", "Rose Eliades", "people"),
        ("Shelley Schools", "Shelly Schools", "people"),
        ("Evan Rosenberg", "Evan Rosenburg", "people"),
        (
            "U.S. Court of Military Commission Review",
            "U.S. Court of Military Commissions Review",
            "organizations",
        ),
        (
            "September 11 Families for Peaceful Tomorrows",
            "September 11th Families for Peaceful Tomorrows",
            "organizations",
        ),
        ("Counterterror bureau", "Counterterrorism Bureau", "organizations"),
        ("International Red Cross", "Red Cross", "organizations"),
        ("Guantánamo Navy base", "GUANTANAMO BAY NAVY BASE", "locations"),
        ("Camp Justice in Guantánamo", "Camp Justice", "locations"),
        (
            "Firing of Convening Authority Harvey Rishikof",
            "Firing of Harvey Rishikof as convening authority",
            "events",
        ),
    ],
)
def test_spelling_variants_reach_the_match_check(a, b, entity_type):
    assert names_are_variants(a, b, entity_type=entity_type, equivalence_groups=GROUPS)


@pytest.mark.parametrize(
    "a, b, entity_type",
    [
        ("David Rosenberg", "Mark Rosenberg", "people"),
        ("Clay Trivett", "Clayton Smith", "people"),  # only one name matches
        ("Iraq", "Iran", "locations"),  # one-word places are never respelled
        ("Yemen", "Yemeni", "locations"),
        ("Camp 5", "Camp 6", "locations"),
        ("North Korea", "South Korea", "locations"),
        ("Palm Beach", "West Palm Beach", "locations"),  # first word differs
        ("Guantanamo Bay Navy Base", "U.S. Navy base", "locations"),  # gold: no
        ("Guantanamo Bay Navy Base", "Guantanamo Bay", "locations"),  # gold: no
        (
            "Senate Armed Services Committee",
            "House Armed Services Committee",
            "organizations",
        ),
        (
            "Charging of Abd al Hadi al Iraqi",
            "Arraignment of Abd al Hadi al Iraqi",
            "events",
        ),
    ],
)
def test_look_alikes_stay_apart(a, b, entity_type):
    assert not names_are_variants(
        a, b, entity_type=entity_type, equivalence_groups=GROUPS
    )


class TestAllCapsCanonicalName:
    CAPS = ("GUANTANAMO BAY NAVY BASE", "military_base")

    def test_all_caps_scores_below_the_same_name_in_normal_case(self):
        assert score_canonical_name("GUANTANAMO BAY NAVY BASE") < score_canonical_name(
            "Guantanamo Bay Navy Base"
        )

    def test_acronyms_are_not_all_caps_names(self):
        assert not is_shouting_name("JTF GTMO")
        assert not is_shouting_name("FBI")
        assert is_shouting_name("GUANTÁNAMO BAY NAVY BASE")

    @pytest.mark.parametrize(
        "incoming", ["Guantanamo Bay Navy Base", "Guantánamo Bay Navy Base"]
    )
    def test_normal_case_spelling_replaces_all_caps(self, incoming):
        canonical, _, swapped = pick_canonical_key(
            self.CAPS, (incoming, "military_base")
        )
        assert swapped and canonical[0] == incoming

    @pytest.mark.parametrize("incoming", ["Gitmo", "Guantánamo", "Guantánamo Bay"])
    def test_shorter_aliases_do_not_replace_the_base_name(self, incoming):
        canonical, _, swapped = pick_canonical_key(
            self.CAPS, (incoming, "military_base")
        )
        assert not swapped and canonical == self.CAPS

    def test_case_only_difference_keeps_existing_normal_case(self):
        existing = ("Guantanamo Bay Navy Base", "military_base")
        _, _, swapped = pick_canonical_key(existing, self.CAPS)
        assert not swapped
