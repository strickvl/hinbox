"""The fact-list updater keeps old text, collapses repeats and keeps both sides."""

from src.engine.fact_list import (
    ArticleClaim,
    ArticleClaims,
    ClaimVerdict,
    ClaimVerdicts,
    Placement,
    SectionPlacements,
    candidates,
    quote_found,
    rows_from_profile,
    split_sections,
    update_from_facts,
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
                fact=3,
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
