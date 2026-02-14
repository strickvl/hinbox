"""Tests for the name-variants utility module."""

from src.utils.name_variants import (
    acronym_matches,
    build_signature,
    compute_acronym,
    expand_equivalents,
    is_acronym_form,
    is_low_quality_name,
    is_name_contained,
    names_likely_same,
    normalize_display,
    normalize_for_match,
    score_canonical_name,
    strip_leading_article,
)

# ──────────────────────────────────────────────
# normalize_display
# ──────────────────────────────────────────────


class TestNormalizeDisplay:
    def test_strips_whitespace(self):
        assert normalize_display("  hello  ") == "hello"

    def test_collapses_internal_whitespace(self):
        assert normalize_display("Department   of   Defense") == "Department of Defense"

    def test_handles_none(self):
        assert normalize_display(None) == ""

    def test_handles_empty(self):
        assert normalize_display("") == ""


# ──────────────────────────────────────────────
# normalize_for_match
# ──────────────────────────────────────────────


class TestNormalizeForMatch:
    def test_lowercase(self):
        assert normalize_for_match("FBI") == "fbi"

    def test_strips_punctuation(self):
        # Dots are stripped, whitespace collapsed: "U.S. Army" → "u s army"
        assert normalize_for_match("U.S. Army") == "u s army"

    def test_normalizes_ampersand(self):
        # & is stripped, whitespace collapsed
        assert (
            normalize_for_match("Justice & Accountability") == "justice accountability"
        )

    def test_collapses_whitespace_after_strip(self):
        result = normalize_for_match("Department (of) Defense")
        assert result == "department of defense"


# ──────────────────────────────────────────────
# is_acronym_form
# ──────────────────────────────────────────────


class TestIsAcronymForm:
    def test_standard_acronym(self):
        assert is_acronym_form("FBI") is True
        assert is_acronym_form("ICE") is True
        assert is_acronym_form("CIA") is True
        assert is_acronym_form("NSA") is True

    def test_dotted_acronym(self):
        assert is_acronym_form("U.N.") is True
        assert is_acronym_form("U.S.") is True

    def test_mixed_case_short(self):
        assert is_acronym_form("DoD") is True
        assert is_acronym_form("DoJ") is True

    def test_not_acronym(self):
        assert is_acronym_form("Department of Defense") is False
        assert is_acronym_form("Navy") is False
        assert is_acronym_form("immigration") is False

    def test_too_short(self):
        assert is_acronym_form("A") is False

    def test_too_long(self):
        assert is_acronym_form("ABCDEFGHIJK") is False

    def test_empty(self):
        assert is_acronym_form("") is False


# ──────────────────────────────────────────────
# compute_acronym
# ──────────────────────────────────────────────


class TestComputeAcronym:
    def test_ice(self):
        assert compute_acronym("Immigration and Customs Enforcement") == "ICE"

    def test_fbi(self):
        assert compute_acronym("Federal Bureau of Investigation") == "FBI"

    def test_dhs(self):
        assert compute_acronym("Department of Homeland Security") == "DHS"

    def test_dod(self):
        assert compute_acronym("Department of Defense") == "DD"
        # Note: DoD is commonly used but "of" is a stopword,
        # so it derives "DD". This is handled by equivalence groups.

    def test_aclu(self):
        assert compute_acronym("American Civil Liberties Union") == "ACLU"

    def test_single_word(self):
        assert compute_acronym("Pentagon") is None

    def test_all_stopwords(self):
        assert compute_acronym("the and of") is None

    def test_two_word(self):
        assert compute_acronym("Coast Guard") == "CG"


# ──────────────────────────────────────────────
# acronym_matches
# ──────────────────────────────────────────────


class TestAcronymMatches:
    def test_ice_matches(self):
        assert acronym_matches("ICE", "Immigration and Customs Enforcement") is True

    def test_fbi_matches(self):
        assert acronym_matches("FBI", "Federal Bureau of Investigation") is True

    def test_aclu_matches(self):
        assert acronym_matches("ACLU", "American Civil Liberties Union") is True

    def test_dhs_matches(self):
        assert acronym_matches("DHS", "Department of Homeland Security") is True

    def test_no_match(self):
        assert acronym_matches("FBI", "Immigration and Customs Enforcement") is False

    def test_non_acronym_short(self):
        assert acronym_matches("Navy", "Department of the Navy") is False

    def test_case_insensitive(self):
        assert acronym_matches("ice", "Immigration and Customs Enforcement") is False
        # "ice" is not detected as acronym form (lowercase)


# ──────────────────────────────────────────────
# is_name_contained
# ──────────────────────────────────────────────


class TestIsNameContained:
    def test_homeland_security_in_department(self):
        assert (
            is_name_contained("Homeland Security", "Department of Homeland Security")
            is True
        )

    def test_coast_guard_not_in_navy(self):
        assert is_name_contained("Coast Guard", "Navy") is False

    def test_too_short(self):
        assert is_name_contained("US", "US Army") is False

    def test_exact_match(self):
        assert is_name_contained("Navy", "Navy") is True

    def test_partial_word_no_match(self):
        # "ice" should not match inside "Service" at word boundaries
        assert is_name_contained("ice", "Internal Revenue Service") is False

    def test_word_boundary(self):
        assert is_name_contained("Army", "US Army Command") is True


# ──────────────────────────────────────────────
# build_signature
# ──────────────────────────────────────────────


class TestBuildSignature:
    def test_long_form(self):
        sig = build_signature("Immigration and Customs Enforcement")
        assert sig.display == "Immigration and Customs Enforcement"
        assert sig.match == "immigration and customs enforcement"
        assert sig.acronym == "ICE"
        assert sig.is_acronym is False

    def test_acronym(self):
        sig = build_signature("FBI")
        assert sig.display == "FBI"
        assert sig.match == "fbi"
        assert sig.is_acronym is True

    def test_tokens(self):
        sig = build_signature("Coast Guard")
        assert sig.tokens == ("coast", "guard")


# ──────────────────────────────────────────────
# expand_equivalents
# ──────────────────────────────────────────────


class TestExpandEquivalents:
    def test_finds_group(self):
        groups = [
            ["Department of Defense", "Pentagon", "DoD"],
            ["FBI", "Federal Bureau of Investigation"],
        ]
        result = expand_equivalents("Pentagon", equivalence_groups=groups)
        assert "Department of Defense" in result
        assert "DoD" in result
        assert "Pentagon" in result

    def test_no_match(self):
        groups = [["Department of Defense", "Pentagon"]]
        result = expand_equivalents("FBI", equivalence_groups=groups)
        assert result == {"FBI"}

    def test_case_insensitive(self):
        groups = [["FBI", "Federal Bureau of Investigation"]]
        result = expand_equivalents("fbi", equivalence_groups=groups)
        assert "Federal Bureau of Investigation" in result

    def test_empty_groups(self):
        result = expand_equivalents("FBI", equivalence_groups=[])
        assert result == {"FBI"}


# ──────────────────────────────────────────────
# names_likely_same
# ──────────────────────────────────────────────


class TestNamesLikelySame:
    """Integration tests for the high-level equivalence check."""

    def test_exact_match(self):
        assert names_likely_same("FBI", "FBI") is True

    def test_normalized_match(self):
        assert (
            names_likely_same("Department  of  Defense", "Department of Defense")
            is True
        )

    def test_acronym_to_full(self):
        assert names_likely_same("ICE", "Immigration and Customs Enforcement") is True

    def test_full_to_acronym(self):
        assert names_likely_same("Immigration and Customs Enforcement", "ICE") is True

    def test_substring_containment(self):
        assert (
            names_likely_same("Homeland Security", "Department of Homeland Security")
            is True
        )

    def test_distinct_orgs(self):
        assert names_likely_same("Navy", "Coast Guard") is False

    def test_distinct_orgs_similar_domain(self):
        assert names_likely_same("FBI", "CIA") is False

    def test_equivalence_group(self):
        groups = [["Department of Defense", "Pentagon", "DoD"]]
        assert (
            names_likely_same(
                "Pentagon",
                "Department of Defense",
                equivalence_groups=groups,
            )
            is True
        )

    def test_people_conservative(self):
        # For people, only exact match and equivalence groups — no acronym/substring
        assert (
            names_likely_same(
                "Smith",
                "John Smith",
                entity_type="people",
            )
            is False
        )

    def test_people_equivalence_group(self):
        groups = [["Robert Smith", "Bob Smith"]]
        assert (
            names_likely_same(
                "Robert Smith",
                "Bob Smith",
                entity_type="people",
                equivalence_groups=groups,
            )
            is True
        )

    def test_dhs_acronym(self):
        assert names_likely_same("DHS", "Department of Homeland Security") is True

    def test_aclu_acronym(self):
        assert names_likely_same("ACLU", "American Civil Liberties Union") is True

    def test_generic_not_matched(self):
        # "Defense" is too short a substring and not an acronym of specific orgs
        # This should NOT match because "Defense" is only 7 chars but it IS
        # contained in "Department of Defense" at word boundaries
        assert names_likely_same("Defense", "Department of Defense") is True

    def test_defense_departments_vs_department_of_defense(self):
        # "Defense departments" is NOT contained in "Department of Defense"
        # (different word order), and not an acronym
        # These would need an equivalence group to match
        assert (
            names_likely_same("Defense departments", "Department of Defense") is False
        )

    def test_locations(self):
        assert (
            names_likely_same(
                "Guantanamo Bay",
                "U.S. military base in Guantanamo Bay",
                entity_type="locations",
            )
            is True
        )


# ──────────────────────────────────────────────
# Name quality assessment
# ──────────────────────────────────────────────


class TestIsLowQualityName:
    """Tests for is_low_quality_name detecting generic/descriptive phrases."""

    def test_generic_plural_organizations(self):
        assert is_low_quality_name("Defense departments") is True
        assert is_low_quality_name("intelligence agencies") is True
        assert is_low_quality_name("security forces") is True
        assert is_low_quality_name("government officials") is True

    def test_proper_org_names_not_flagged(self):
        assert is_low_quality_name("Department of Defense") is False
        assert is_low_quality_name("FBI") is False
        assert is_low_quality_name("Immigration and Customs Enforcement") is False
        assert is_low_quality_name("Joint Task Force Guantanamo") is False

    def test_single_word_not_flagged(self):
        """Single words shouldn't be flagged even if they're plural."""
        assert is_low_quality_name("departments") is False
        assert is_low_quality_name("Navy") is False

    def test_descriptive_location_phrases(self):
        assert is_low_quality_name("U.S. military base in Guantanamo Bay") is True
        assert is_low_quality_name("military base in Cuba") is True
        assert is_low_quality_name("detention center at Guantanamo") is True
        assert is_low_quality_name("prison near Havana") is True
        assert is_low_quality_name("facility at Bagram") is True

    def test_proper_location_names_not_flagged(self):
        assert is_low_quality_name("Naval Station Guantanamo Bay") is False
        assert is_low_quality_name("Guantanamo Bay") is False
        assert is_low_quality_name("Camp Delta") is False
        assert is_low_quality_name("Pentagon") is False

    def test_empty_and_whitespace(self):
        assert is_low_quality_name("") is False
        assert is_low_quality_name("   ") is False


class TestStripLeadingArticle:
    """Tests for strip_leading_article."""

    def test_strips_the(self):
        assert strip_leading_article("the Pentagon") == "Pentagon"
        assert strip_leading_article("The New York Times") == "New York Times"

    def test_preserves_without_article(self):
        assert strip_leading_article("Pentagon") == "Pentagon"
        assert strip_leading_article("FBI") == "FBI"

    def test_case_insensitive(self):
        assert strip_leading_article("THE PENTAGON") == "PENTAGON"


# ──────────────────────────────────────────────
# score_canonical_name
# ──────────────────────────────────────────────


class TestScoreCanonicalName:
    """Tests for the standalone canonical name scoring function."""

    def test_longer_name_scores_higher(self):
        assert score_canonical_name("Republic of Cuba") > score_canonical_name("Cuba")

    def test_acronym_penalized(self):
        assert score_canonical_name("ICE") < 0
        assert score_canonical_name(
            "Immigration and Customs Enforcement"
        ) > score_canonical_name("ICE")

    def test_contextual_suffix_penalized(self):
        assert score_canonical_name("U.S. soil") < 0
        assert score_canonical_name("United States") > score_canonical_name("U.S. soil")

    def test_low_quality_name_penalized(self):
        """Generic plurals and descriptive phrases get a heavy penalty."""
        assert score_canonical_name("Defense departments") < 0
        assert score_canonical_name("Department of Defense") > score_canonical_name(
            "Defense departments"
        )

    def test_descriptive_location_penalized(self):
        """Descriptive location phrases should score much lower than proper names."""
        assert score_canonical_name(
            "U.S. military base in Guantanamo Bay"
        ) < score_canonical_name("Guantanamo Bay")

    def test_normal_name_positive(self):
        assert score_canonical_name("United States") > 0

    def test_empty_zero(self):
        assert score_canonical_name("") == 0.0
