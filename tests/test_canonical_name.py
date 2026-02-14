"""Tests for canonical name selection during entity merging.

Verifies that when two entity names are merged, the more canonical
(formal/complete) name is kept as the primary key and the less
canonical name is demoted to alternative_names.
"""

from src.engine import EntityMerger


class TestScoreCanonicalName:
    """Tests for EntityMerger._score_canonical_name scoring heuristic."""

    def setup_method(self):
        self.merger = EntityMerger("locations")

    def test_longer_name_scores_higher(self):
        """Longer names should score higher (more specific/formal)."""
        short = self.merger._score_canonical_name("Cuba")
        long = self.merger._score_canonical_name("Republic of Cuba")
        assert long > short

    def test_acronym_penalized(self):
        """Acronym-like names should be penalized."""
        acronym = self.merger._score_canonical_name("ICE")
        full = self.merger._score_canonical_name("Immigration and Customs Enforcement")
        assert full > acronym
        # Acronym score should be negative due to penalty
        assert acronym < 0

    def test_contextual_suffix_penalized(self):
        """Metonymic/contextual suffixes like 'soil' should be penalized."""
        contextual = self.merger._score_canonical_name("U.S. soil")
        proper = self.merger._score_canonical_name("United States")
        assert proper > contextual
        # Contextual score should be negative due to large penalty
        assert contextual < 0

    def test_contextual_suffixes_all_penalized(self):
        """All defined contextual suffixes should trigger the penalty."""
        normal_score = self.merger._score_canonical_name("Normal Place")
        for suffix in [
            "soil",
            "territory",
            "waters",
            "border",
            "grounds",
            "arena",
            "area",
        ]:
            name = f"Some {suffix}"
            score = self.merger._score_canonical_name(name)
            assert score < normal_score, (
                f"'{name}' should score lower than 'Normal Place'"
            )

    def test_normal_name_positive_score(self):
        """A regular proper name should have a positive score."""
        score = self.merger._score_canonical_name("United States")
        assert score > 0

    def test_empty_name_zero_score(self):
        """Empty string should score 0."""
        score = self.merger._score_canonical_name("")
        assert score == 0.0


class TestPickCanonicalKey:
    """Tests for EntityMerger._pick_canonical_key selection logic."""

    def setup_method(self):
        self.loc_merger = EntityMerger("locations")
        self.org_merger = EntityMerger("organizations")
        self.ppl_merger = EntityMerger("people")

    def test_us_soil_vs_united_states(self):
        """'United States' should win over 'U.S. soil' (contextual suffix penalty)."""
        existing = ("U.S. soil", "country")
        incoming = ("United States", "country")
        canonical, demoted, swapped = self.loc_merger._pick_canonical_key(
            existing, incoming
        )
        assert swapped is True
        assert canonical == incoming
        assert demoted == existing

    def test_ice_vs_full_name(self):
        """Full org name should win over its acronym."""
        existing = ("ICE", "government_agency")
        incoming = ("Immigration and Customs Enforcement", "government_agency")
        canonical, demoted, swapped = self.org_merger._pick_canonical_key(
            existing, incoming
        )
        assert swapped is True
        assert canonical == incoming
        assert demoted == existing

    def test_pentagon_vs_department_of_defense(self):
        """Pentagon should stay — both are valid proper names, score gap < threshold."""
        existing = ("Pentagon", "government_agency")
        incoming = ("Department of Defense", "government_agency")
        canonical, demoted, swapped = self.org_merger._pick_canonical_key(
            existing, incoming
        )
        assert swapped is False
        assert canonical == existing

    def test_defense_vs_department_of_defense(self):
        """'Department of Defense' should win via containment bonus."""
        existing = ("Defense", "government_agency")
        incoming = ("Department of Defense", "government_agency")
        canonical, demoted, swapped = self.org_merger._pick_canonical_key(
            existing, incoming
        )
        assert swapped is True
        assert canonical == incoming

    def test_dhs_vs_department_of_homeland_security(self):
        """Full name should win over its acronym (DHS)."""
        existing = ("DHS", "government_agency")
        incoming = ("Department of Homeland Security", "government_agency")
        canonical, demoted, swapped = self.org_merger._pick_canonical_key(
            existing, incoming
        )
        assert swapped is True
        assert canonical == incoming

    def test_guantanamo_bay_vs_guantanamo(self):
        """'Guantánamo Bay' should stay — it's more specific (contains the shorter form)."""
        existing = ("Guantánamo Bay", "detention_facility")
        incoming = ("Guantanamo", "detention_facility")
        canonical, demoted, swapped = self.loc_merger._pick_canonical_key(
            existing, incoming
        )
        assert swapped is False
        assert canonical == existing

    def test_identical_names_no_swap(self):
        """Identical names (case-insensitive) should never swap."""
        existing = ("United States", "country")
        incoming = ("united states", "country")
        canonical, demoted, swapped = self.loc_merger._pick_canonical_key(
            existing, incoming
        )
        assert swapped is False
        assert canonical == existing

    def test_people_string_keys(self):
        """People use string keys (not tuples) — should work correctly."""
        canonical, demoted, swapped = self.ppl_merger._pick_canonical_key(
            "Al-Qahtani", "Mohammed Al-Qahtani"
        )
        # "Al-Qahtani" is contained in "Mohammed Al-Qahtani" → incoming wins
        assert swapped is True
        assert canonical == "Mohammed Al-Qahtani"

    def test_full_name_already_canonical(self):
        """When existing is already the better name, no swap should occur."""
        existing = ("Immigration and Customs Enforcement", "government_agency")
        incoming = ("ICE", "government_agency")
        canonical, demoted, swapped = self.org_merger._pick_canonical_key(
            existing, incoming
        )
        assert swapped is False
        assert canonical == existing


class TestCanonicalNameIntegration:
    """Integration tests verifying re-keying in the entity dict after merge."""

    def test_rekey_on_merge(self):
        """After merge with swap, entity dict should have the canonical key, not the old one."""
        merger = EntityMerger("locations")

        # Simulate existing entity keyed under the worse name
        entities = {
            "people": {},
            "organizations": {},
            "locations": {
                ("U.S. soil", "country"): {
                    "name": "U.S. soil",
                    "type": "country",
                    "profile": {"text": "A location profile."},
                    "profile_embedding": [0.5, 0.5, 0.5],
                    "alternative_names": [],
                    "aliases": [],
                    "articles": [],
                },
            },
            "events": {},
        }

        existing_key = ("U.S. soil", "country")
        incoming_key = ("United States", "country")

        canonical, demoted, swapped = merger._pick_canonical_key(
            existing_key, incoming_key
        )
        assert swapped is True

        # Simulate re-keying as merge_entities would do
        existing_entity = entities["locations"].pop(existing_key)
        existing_entity["name"] = canonical[0]
        existing_entity["type"] = canonical[1]
        merger._add_alternative_name(existing_entity, demoted)
        entities["locations"][canonical] = existing_entity

        # Old key should be gone
        assert existing_key not in entities["locations"]
        # New key should exist
        assert incoming_key in entities["locations"]
        # Entity name updated
        assert entities["locations"][incoming_key]["name"] == "United States"
        # Demoted name in alternatives
        alt_names = [
            a["name"] for a in entities["locations"][incoming_key]["alternative_names"]
        ]
        assert "U.S. soil" in alt_names
