"""The fact-list updater keeps old text, collapses repeats and keeps both sides."""

import json

from src.engine.fact_list import (
    FACT_ROWS_KEY,
    ArticleClaim,
    ArticleClaims,
    ClaimVerdict,
    ClaimVerdicts,
    Placement,
    SectionPlacements,
    _match_heading,
    adds_values,
    candidates,
    load_rows,
    quote_found,
    rows_from_profile,
    same_subject,
    split_sections,
    update_from_facts,
    values_differ,
)
from src.evals.profile_update import score_update

PROFILE = (
    "# Abd al-Rahim al-Nashiri\n\n"
    "Nashiri is a Saudi detainee held at Guantánamo.^[a1]\n\n"
    "## Capture\n\n"
    "Nashiri was captured in the United Arab Emirates in November 2002.^[a1] "
    "He was held by the CIA at secret sites.^[a2]\n\n"
    "## Legal Proceedings\n\n"
    "Prosecutors charged Nashiri over the bombing of the USS Cole in 2011.^[a2] "
    "The judge set a trial date for February 2015.^[a3]\n"
)
ARTICLE = (
    "Nashiri was captured in the United Arab Emirates in October 2002, a court "
    "filing said. He was held by the CIA at secret sites. On Monday a military "
    "judge ruled that his confession could not be used at trial."
)


def _claims():
    return [
        ArticleClaim(
            claim="Nashiri was held by the CIA at secret sites.",
            section="## Capture",
            quote="He was held by the CIA at secret sites.",
        ),
        ArticleClaim(
            claim="Nashiri was captured in the United Arab Emirates in October 2002.",
            section="## Capture",
            quote="captured in the United Arab Emirates in October 2002",
            source="court filing",
        ),
        ArticleClaim(
            claim="A military judge ruled that Nashiri's confession could not be used.",
            section="## Legal Proceedings",
            quote="a military judge ruled that his confession could not be used at trial",
        ),
        ArticleClaim(
            claim="Nashiri likes chess.",
            section="## Capture",
            quote="Nashiri plays chess every day",  # not in the article: dropped
        ),
    ]


def _fake(answers):
    calls = []

    def generate(messages, response_model, model, max_tokens):
        calls.append((response_model.__name__, messages))
        return answers[response_model.__name__](messages)

    return generate, calls


def _verdicts(messages):
    rows = {
        line.split("]")[0].strip(" [")
        for line in messages[1]["content"].splitlines()
        if line.startswith("  [")
    }
    capture = next(r for r in rows if r == "f2")
    held = next(r for r in rows if r == "f3")
    return ClaimVerdicts(
        verdicts=[
            ClaimVerdict(claim=1, verdict="same", row=held),
            ClaimVerdict(claim=2, verdict="conflict", row=capture),
        ]
    )


def _write(messages):
    return SectionPlacements(
        placements=[
            Placement(
                facts=[3],
                after="The judge set a trial date for February 2015.^[a3]",
                sentence="A military judge ruled the confession inadmissible.^[a9]",
            )
        ]
    )


ANSWERS = {
    "ArticleClaims": lambda m: ArticleClaims(claims=_claims()),
    "ClaimVerdicts": _verdicts,
    "SectionPlacements": _write,
}


def test_sections_rejoin_to_the_same_text():
    assert "".join(split_sections(PROFILE)) == PROFILE
    rows = rows_from_profile(PROFILE)
    assert [r["section"] for r in rows] == [
        "# Abd al-Rahim al-Nashiri",
        "## Capture",
        "## Capture",
        "## Legal Proceedings",
        "## Legal Proceedings",
    ]
    assert rows[1]["sources"] == [{"article_id": "a1"}]


def test_quotes_are_checked_against_the_article():
    assert quote_found("He was held by the CIA at secret sites.", ARTICLE)
    assert quote_found(
        "a military judge ruled that the confession could not be used", ARTICLE
    )
    assert not quote_found("Nashiri plays chess every day", ARTICLE)


def test_lookup_finds_the_row_with_the_same_date():
    rows = rows_from_profile(PROFILE)
    found = candidates("Nashiri was captured in the Emirates in October 2002.", rows)
    assert found[0]["id"] == "f2"


def test_update_keeps_old_text_and_both_sides_and_counts_the_repeat():
    generate, calls = _fake(ANSWERS)
    profile = {"text": PROFILE, "sources": ["a1", "a2", "a3"]}
    new = update_from_facts(
        "people", "Nashiri", profile, ARTICLE, "a9", model="m", generate=generate
    )
    text = new["text"]
    # Every old sentence is still there, word for word
    for row in rows_from_profile(PROFILE):
        assert row["text"] in text
    # The conflict is written right after the old sentence, both cited
    assert (
        "November 2002.^[a1] Another report differs: Nashiri was captured in "
        "the United Arab Emirates in October 2002.^[a9]"
    ) in text
    # The new fact goes where the model placed it; any stray citation is replaced
    assert (
        "February 2015.^[a3] A military judge ruled the confession inadmissible.^[a9]"
    ) in text
    # The repeat is counted on its row, not cited again in the text
    held = next(r for r in new["fact_rows"] if r["id"] == "f3")
    assert held["repeats"] == ["a9"]
    assert "secret sites.^[a2]" in text and "secret sites.^[a2, a9]" not in text
    report = new["fact_report"]
    assert report["dropped_quotes"] == ["Nashiri likes chess."]
    assert (report["same"], report["conflict"], report["new"]) == (1, 1, 1)
    # Extract, compare, one section write; none of them saw the whole profile
    assert [c[0] for c in calls] == [
        "ArticleClaims",
        "ClaimVerdicts",
        "SectionPlacements",
    ]
    assert all("Saudi detainee" not in m[1]["content"] for _, m in calls)
    assert new["sources"][-1] == "a9"


def test_a_new_kind_of_source_is_cited_on_the_old_sentence():
    def verdicts(messages):
        return ClaimVerdicts(
            verdicts=[
                ClaimVerdict(claim=1, verdict="same", row="f3", new_evidence=True)
            ]
        )

    answers = {
        **ANSWERS,
        "ClaimVerdicts": verdicts,
        "ArticleClaims": lambda m: ArticleClaims(claims=_claims()[:1]),
    }
    generate, calls = _fake(answers)
    new = update_from_facts(
        "people",
        "Nashiri",
        {"text": PROFILE},
        ARTICLE,
        "a9",
        model="m",
        generate=generate,
    )
    assert "secret sites.^[a2, a9]" in new["text"]
    assert len(calls) == 2  # nothing new to write


def test_a_new_section_is_added_at_the_end_and_scores_as_kept():
    claim = ArticleClaim(
        claim="Nashiri went on hunger strike in 2013.",
        section="Detention conditions",
        quote="judge ruled",
    )
    answers = {**ANSWERS, "ArticleClaims": lambda m: ArticleClaims(claims=[claim])}
    generate, calls = _fake(answers)
    new = update_from_facts(
        "people",
        "Nashiri",
        {"text": PROFILE},
        ARTICLE,
        "a9",
        model="m",
        generate=generate,
    )
    assert new["text"].endswith(
        "## Detention conditions\nNashiri went on hunger strike in 2013.^[a9]\n"
    )
    assert [c[0] for c in calls] == ["ArticleClaims"]  # no rows to compare, no write
    case = {
        "kind": "new_fact",
        "profile": {"text": PROFILE, "sources": ["a1", "a2", "a3"]},
        "article_id": "a9",
        "article_text": ARTICLE,
        "expected": {},
    }
    scores = score_update(case, new["text"])
    assert scores["passed"] and scores["fact_retention"] == 1.0


def test_a_conflict_needs_a_different_year_or_number():
    assert values_differ(
        "Nashiri was captured in October 2003.", "captured in November 2002"
    )
    # Different kinds of number, or the claim's numbers already in the row
    assert not values_differ("Camp 6 is a 200-cell building.", "Camp 6 opened in 2006.")
    assert not values_differ(
        "The bombing killed 17 sailors in 2000.",
        "The 2000 bombing of the USS Cole killed 17 sailors.",
    )
    assert not values_differ("Nashiri was held at secret sites.", "held in 2002")


def test_a_conflict_without_differing_values_is_written_as_a_new_fact():
    claim = ArticleClaim(
        claim="Nashiri was captured in the United Arab Emirates.",
        section="## Capture",
        quote="captured in the United Arab Emirates",
    )

    def verdicts(messages):
        return ClaimVerdicts(
            verdicts=[ClaimVerdict(claim=1, verdict="conflict", row="f2")]
        )

    answers = {
        "ArticleClaims": lambda m: ArticleClaims(claims=[claim]),
        "ClaimVerdicts": verdicts,
        "SectionPlacements": lambda m: SectionPlacements(),
    }
    generate, _ = _fake(answers)
    new = update_from_facts(
        "people",
        "Nashiri",
        {"text": PROFILE},
        ARTICLE,
        "a9",
        model="m",
        generate=generate,
    )
    assert "Another report differs" not in new["text"]
    assert new["fact_report"]["not_conflict"] == 1
    # The model wrote no sentence for it, so it goes in as extracted
    assert (
        "secret sites.^[a2]\nNashiri was captured in the United Arab Emirates.^[a9]"
        in (new["text"])
    )


def test_passing_mentions_keep_two_claims_and_headings_are_reused():
    claims = [
        ArticleClaim(
            claim=f"Nashiri fact number {n} about the judge.",
            section="### legal proceedings",
            quote="judge ruled",
        )
        for n in range(1, 5)
    ]
    answers = {
        "ArticleClaims": lambda m: ArticleClaims(mainly_about=False, claims=claims),
        "SectionPlacements": lambda m: SectionPlacements(
            placements=[
                Placement(
                    facts=[1, 2],
                    after="END",
                    sentence="He faced facts one and two.^[a1]",
                )
            ]
        ),
    }
    generate, _ = _fake(answers)
    new = update_from_facts(
        "people",
        "Nashiri",
        {"text": PROFILE},
        ARTICLE,
        "a9",
        model="m",
        generate=generate,
    )
    assert len(new["fact_report"]["dropped_passing"]) == 2
    assert new["text"].count("Legal Proceedings") == 1
    assert new["text"].endswith(
        "February 2015.^[a3]\nHe faced facts one and two.^[a9]\n"
    )


def test_counts_only_conflict_for_the_same_year():
    assert not values_differ(
        "As of last week, 12 captives were on hunger strike.",
        "About 100 captives were on hunger strike.",
    )
    assert values_differ(
        "In 2013, 12 captives were on hunger strike.",
        "In 2013, 106 captives were on hunger strike.",
    )


def test_the_passing_cap_counts_only_new_claims():
    # Background the profile has comes first; the new facts after it survive
    claims = _claims()[:1] + [
        ArticleClaim(
            claim=f"Nashiri new point {n}.", section="## Capture", quote="judge ruled"
        )
        for n in (1, 2, 3)
    ]

    def verdicts(messages):
        return ClaimVerdicts(verdicts=[ClaimVerdict(claim=1, verdict="same", row="f3")])

    answers = {
        "ArticleClaims": lambda m: ArticleClaims(mainly_about=False, claims=claims),
        "ClaimVerdicts": verdicts,
        "SectionPlacements": lambda m: SectionPlacements(),
    }
    generate, _ = _fake(answers)
    new = update_from_facts(
        "people",
        "Nashiri",
        {"text": PROFILE},
        ARTICLE,
        "a9",
        model="m",
        generate=generate,
    )
    report = new["fact_report"]
    assert (report["same"], report["new"]) == (1, 2)
    assert report["dropped_passing"] == ["Nashiri new point 3."]
    assert "new point 1" in new["text"] and "new point 2" in new["text"]


def test_article_subheadings_are_not_copied_in_as_sections():
    article = ARTICLE + "\n\nASTROTURF FOR CAMP?\n\nThe camp got $30 million."
    assert (
        _match_heading("## ASTROTURF FOR CAMP?", ["## Capture"], article)
        == "## Capture"
    )
    assert _match_heading("Funding", ["## Capture"], article) == "## Funding"
    assert _match_heading("### capture", ["## Capture"]) == "## Capture"
    assert (
        _match_heading("Shouting Heading NOW", ["## A"], near="## B")
        == "## Shouting Heading NOW"
    )
    assert _match_heading("IF A DEATH OCCURS", ["## A"], near="## B") == "## B"


def test_a_loose_quote_must_match_one_stretch_in_order():
    article = (
        "Nashiri was captured in 2002. His lawyer said he was not charged until 2011."
    )
    assert not quote_found("Nashiri was not captured in 2002.", article)
    assert quote_found("His lawyer said that he was not charged until 2011", article)


def test_a_later_state_is_a_timeline_entry_and_rows_survive_to_the_next_update():
    claim = ArticleClaim(
        claim="Nashiri was moved from the secret sites to Guantánamo in 2006.",
        section="## Other",
        quote="judge ruled",
        source="Pentagon statement",
    )

    def verdicts(messages):
        return ClaimVerdicts(
            verdicts=[ClaimVerdict(claim=1, verdict="update", row="f3")]
        )

    answers = {
        "ArticleClaims": lambda m: ArticleClaims(claims=[claim]),
        "ClaimVerdicts": verdicts,
        "SectionPlacements": lambda m: SectionPlacements(),
    }
    generate, _ = _fake(answers)
    new = update_from_facts(
        "people",
        "Nashiri",
        {"text": PROFILE},
        ARTICLE,
        "a9",
        model="m",
        generate=generate,
    )
    assert new["fact_report"]["update"] == 1
    assert "Another report differs" not in new["text"]
    # Goes in the section of the state it follows, not a new "## Other"
    assert (
        "secret sites.^[a2]\nNashiri was moved from the secret sites to Guantánamo in 2006.^[a9]"
        in new["text"]
    )
    # The saved rows keep the new row's source kind for the next update
    rows = load_rows({"text": new["text"], FACT_ROWS_KEY: new[FACT_ROWS_KEY]})
    moved = next(r for r in rows if "moved" in r["claim"])
    assert moved["sources"][0]["kind"] == "Pentagon statement"
    assert len({r["id"] for r in rows}) == len(rows) == 6


def test_saved_rows_whose_sentence_is_gone_are_dropped():
    saved = json.dumps(
        [
            {
                "id": "f9",
                "section": "## Capture",
                "claim": "Old",
                "text": "Gone.^[a1]",
                "sources": [{"article_id": "a1", "kind": "court filing"}],
                "repeats": [],
            },
            {
                "id": "f2",
                "section": "## Capture",
                "claim": "He was held by the CIA at secret sites.",
                "text": "He was held by the CIA at secret sites.^[a2]",
                "sources": [{"article_id": "a2", "kind": "court filing"}],
                "repeats": ["a7"],
            },
        ]
    )
    rows = load_rows({"text": PROFILE, FACT_ROWS_KEY: saved})
    assert "Gone.^[a1]" not in [r["text"] for r in rows]
    held = next(r for r in rows if r["id"] == "f2")
    assert held["repeats"] == ["a7"] and held["sources"][0]["kind"] == "court filing"
    assert len(rows) == 5  # the other four sentences are read from the text


def test_a_conflict_must_be_about_the_same_thing():
    assert same_subject(
        "The camps opened in January 2003.", "The prison camps opened in January 2002."
    )
    assert not same_subject(
        "Attorney General Eric Holder decided the defendants should face federal "
        "trials in New York.",
        "The Obama administration was still deciding where to stage the trials of "
        "five alleged co-conspirators.",
    )


def test_update_profile_can_use_the_fact_list(monkeypatch):
    from src.engine import fact_list, profiles

    def fake(*args, **kwargs):
        return {
            "text": PROFILE + "New.^[a9]\n",
            "sources": ["a9"],
            "fact_report": {},
            FACT_ROWS_KEY: "[]",
        }

    monkeypatch.setattr(fact_list, "update_from_facts", fake)
    profile, _, history = profiles.update_profile(
        "people",
        "Nashiri",
        {"text": PROFILE, "sources": []},
        profiles.VersionedProfile(),
        ARTICLE,
        "a9",
        approach="facts",
    )
    assert profile["text"].endswith("New.^[a9]\n") and profile[FACT_ROWS_KEY] == "[]"
    assert history == []


def test_a_rescheduled_date_stays_on_the_timeline():
    claim = ArticleClaim(
        claim="The judge set a trial date for June 2016.",
        section="## Legal Proceedings",
        quote="judge ruled",
    )
    answers = {
        "ArticleClaims": lambda m: ArticleClaims(claims=[claim]),
        "ClaimVerdicts": lambda m: ClaimVerdicts(
            verdicts=[ClaimVerdict(claim=1, verdict="update", row="f5")]
        ),
        "SectionPlacements": lambda m: SectionPlacements(),
    }
    generate, _ = _fake(answers)
    new = update_from_facts(
        "people",
        "Nashiri",
        {"text": PROFILE},
        ARTICLE,
        "a9",
        model="m",
        generate=generate,
    )
    assert new["fact_report"]["update"] == 1
    assert "Another report differs" not in new["text"]
    assert "February 2015.^[a3]" in new["text"] and "June 2016.^[a9]" in new["text"]


def test_placed_facts_keep_their_quote_and_source_kind():
    generate, _ = _fake(ANSWERS)
    new = update_from_facts(
        "people",
        "Nashiri",
        {"text": PROFILE},
        ARTICLE,
        "a9",
        model="m",
        generate=generate,
    )
    saved = json.loads(new[FACT_ROWS_KEY])
    placed = next(r for r in saved if "inadmissible" in r["text"])
    assert placed["sources"][0]["quote"].startswith("a military judge ruled")


def test_a_same_claim_with_a_new_number_is_written_not_dropped():
    claim = ArticleClaim(
        claim="Nashiri was held by the CIA at secret sites for 4 years.",
        section="## Capture",
        quote="He was held by the CIA at secret sites.",
    )

    def verdicts(messages):
        return ClaimVerdicts(verdicts=[ClaimVerdict(claim=1, verdict="same", row="f3")])

    answers = {
        "ArticleClaims": lambda m: ArticleClaims(claims=[claim]),
        "ClaimVerdicts": verdicts,
        "SectionPlacements": lambda m: SectionPlacements(),
    }
    generate, _ = _fake(answers)
    new = update_from_facts(
        "people",
        "Nashiri",
        {"text": PROFILE},
        ARTICLE,
        "a9",
        model="m",
        generate=generate,
    )
    assert new["fact_report"]["same_with_detail"] == 1
    assert "for 4 years.^[a9]" in new["text"]


def test_new_evidence_keeps_rows_that_share_its_sentence():
    shared = "Nashiri was held by the CIA at secret sites in Thailand.^[a2]"
    profile = PROFILE.replace("He was held by the CIA at secret sites.^[a2]", shared)
    rows = [
        {
            "id": "f1",
            "section": "## Capture",
            "claim": "Nashiri was held by the CIA at secret sites.",
            "text": shared,
            "sources": [{"article_id": "a2", "kind": "official"}],
            "repeats": [],
        },
        {
            "id": "f2",
            "section": "## Capture",
            "claim": "Nashiri was held at secret sites in Thailand.",
            "text": shared,
            "sources": [{"article_id": "a2", "kind": "leak"}],
            "repeats": [],
        },
    ]
    claim = ArticleClaim(
        claim="Nashiri was held by the CIA at secret sites.",
        section="## Capture",
        quote="He was held by the CIA at secret sites.",
        source="court filing",
    )
    answers = {
        "ArticleClaims": lambda m: ArticleClaims(claims=[claim]),
        "ClaimVerdicts": lambda m: ClaimVerdicts(
            verdicts=[
                ClaimVerdict(claim=1, verdict="same", row="f1", new_evidence=True)
            ]
        ),
        "SectionPlacements": lambda m: SectionPlacements(),
    }
    generate, _ = _fake(answers)
    new = update_from_facts(
        "people",
        "Nashiri",
        {"text": profile, FACT_ROWS_KEY: json.dumps(rows)},
        ARTICLE,
        "a9",
        model="m",
        generate=generate,
    )
    saved = {r["claim"]: r for r in json.loads(new[FACT_ROWS_KEY])}
    assert "Nashiri was held by the CIA at secret sites." in saved
    assert "Nashiri was held at secret sites in Thailand." in saved
    assert saved["Nashiri was held at secret sites in Thailand."]["sources"] == [
        {"article_id": "a2", "kind": "leak"}
    ]


def test_a_same_claim_is_not_rewritten_for_a_number_the_profile_already_has():
    # The Cole death toll is in another sentence, spelled out: not new detail
    assert not adds_values(
        "The bombing killed 17 U.S. sailors in October 2000.",
        "Nashiri is accused of the October 2000 bombing of the USS Cole.",
        "### Attack\nIt killed seventeen sailors.^[a1]",
    )
    assert adds_values(
        "The closed hearing was held on Feb. 22.",
        "The Pentagon released the transcript of a closed hearing.",
        "### Release\nThe transcript came out in 2013.^[a1]",
    )
    # "one of the 17" is not a count of one
    assert not adds_values(
        "Gunn was one of the 17 Americans killed.",
        "Gunn was among the 17 sailors killed.",
    )
