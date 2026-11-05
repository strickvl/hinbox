"""The date check keeps both dates when an article re-dates a past event in passing."""

from src.engine.date_disputes import (
    DateVerdict,
    DateVerdicts,
    date_check_enabled,
    date_pairs,
    dates_differ,
    dates_share_context,
    dispute_text,
)
from src.engine.fact_list import (
    ArticleClaim,
    ArticleClaims,
    ClaimVerdicts,
    SectionPlacements,
    rows_from_profile,
    update_from_facts,
)
from src.evals.profile_update import score_update

OLD = (
    "In July 2014, Smith was assigned to the case against Nashiri, accused of "
    "the October 2000 USS Cole bombing, which killed 17 sailors.^[a1]"
)
PROFILE = (
    f"# Judge Smith\n\n## Career\n\n{OLD} Smith ordered a new hearing in 2015.^[a2]\n"
)
# The hearing is the news; the bombing date is background the extract skips
ARTICLE = (
    "Judge Smith ruled on Tuesday that the hearing would go ahead in May. "
    "Seventeen sailors died in the bombing of the USS Cole off Aden on "
    "Oct. 12, 1999, and prosecutors say Nashiri planned it."
)
HEARING = ArticleClaim(
    claim="Smith ruled that the hearing would go ahead in May.",
    section="## Career",
    quote="Judge Smith ruled on Tuesday that the hearing would go ahead in May.",
)


def _fake(date_answer, claims=(HEARING,)):
    calls = []

    def generate(messages, response_model, model, max_tokens):
        calls.append((response_model.__name__, messages))
        if response_model is DateVerdicts:
            return date_answer(messages)
        if response_model is ArticleClaims:
            return ArticleClaims(claims=list(claims))
        if response_model is ClaimVerdicts:
            return ClaimVerdicts(verdicts=[])
        return SectionPlacements(placements=[])

    return generate, calls


def _conflict(sentence="The USS Cole bombing happened on Oct. 12, 1999."):
    return lambda m: DateVerdicts(
        verdicts=[
            DateVerdict(
                pair=1,
                event="the USS Cole bombing",
                verdict="conflict",
                sentence=sentence,
            )
        ]
    )


def _update(generate, profile=None, **kwargs):
    kwargs.setdefault("date_check", True)  # off by default in the pipeline
    return update_from_facts(
        "people",
        "Smith",
        profile or {"text": PROFILE},
        ARTICLE,
        "a9",
        model="m",
        generate=generate,
        **kwargs,
    )


def test_dates_differ_needs_a_value_on_each_side():
    assert dates_differ("moved in September 2005", "transferred in September 2006")
    assert dates_differ("captured in October 2002", "captured in November 2002")
    assert dates_differ("on Oct. 12, 1999", "the October 2000 bombing")
    assert dates_differ("captured in Mar. 2002", "captured in April 2002")
    # An extra year on one side is more detail, not another date
    assert not dates_differ("opened in 2002 and grew in 2006", "opened in 2002")
    assert not dates_differ("opened in 2002", "it opened in January 2002")


def test_pairs_need_shared_words_and_skip_unrelated_dates():
    rows = rows_from_profile(PROFILE)
    pairs = date_pairs(rows, ARTICLE, skip=set())
    assert [p["row"]["claim"] for p in pairs] == [rows[0]["claim"]]
    assert "1999" in pairs[0]["sentence"]
    # A year in an article about something else makes no pair
    assert not date_pairs(rows, "Officials said the new road opened in 2019.", set())
    assert not date_pairs(rows, ARTICLE, skip={rows[0]["id"]})


def test_a_conflict_is_written_after_the_old_sentence_and_scores_both():
    generate, calls = _fake(_conflict())
    new = _update(generate)
    added = (
        "Another report differs: The USS Cole bombing happened on Oct. 12, 1999.^[a9]"
    )
    assert f"{OLD} {added}" in new["text"]
    assert new["fact_report"]["date_disputes"] == 1
    # The pair is shown with the sentence before it, for who "it" is
    prompt = next(m for name, m in calls if name == "DateVerdicts")[1]["content"]
    assert "Known fact: In July 2014" in prompt and "Before it: Judge Smith" in prompt
    row = next(r for r in new["fact_rows"] if r.get("disputes"))
    assert row["disputes"] == "f1" and row["sources"][0]["quote"].startswith(
        "Seventeen"
    )
    case = {
        "kind": "disputed",
        "profile": {"text": PROFILE, "sources": ["a1", "a2"]},
        "article_id": "a9",
        "article_text": ARTICLE,
        "expected": {
            "original_value": "2000",
            "disputed_value": "1999",
            "original_sources": ["a1"],
            "profile_sentence": OLD,
            "article_sentence": ARTICLE.split(". ", 1)[1],
        },
    }
    assert score_update(case, new["text"])["keeps_both"]


def test_different_or_later_changes_nothing():
    for verdict in ("different", "later"):
        answer = lambda m, v=verdict: DateVerdicts(  # noqa: E731
            verdicts=[DateVerdict(pair=1, verdict=v)]
        )
        new = _update(_fake(answer)[0])
        assert "Another report" not in new["text"]
        assert new["fact_report"]["date_verdicts"] == [verdict]


def test_a_sentence_without_the_new_date_falls_back_to_the_articles():
    pair = date_pairs(rows_from_profile(PROFILE), ARTICLE, set())[0]
    # The model copied the profile's year: code writes the article's sentence
    text = dispute_text("The bombing was in October 2000.", pair, "a9")
    assert text.startswith("Another report differs: Seventeen sailors died")
    assert text.endswith("Oct. 12, 1999, and prosecutors say Nashiri planned it.^[a9]")


def test_the_same_sentence_is_not_also_written_as_a_new_fact():
    background = ArticleClaim(
        claim="The USS Cole bombing killed 17 sailors on Oct. 12, 1999.",
        section="## Career",
        quote="Seventeen sailors died in the bombing of the USS Cole off Aden",
    )
    generate, _ = _fake(_conflict(), claims=(HEARING, background))
    new = _update(generate)
    assert new["text"].count("1999") == 1
    assert new["fact_report"]["date_dropped_new"] == 1
    assert "hearing would go ahead in May" in new["text"]


def test_the_next_article_with_the_same_other_date_adds_nothing():
    first = _update(_fake(_conflict())[0])
    generate, calls = _fake(_conflict())
    second = _update(generate, profile=first)
    assert second["text"].count("Another report differs") == 1
    assert "DateVerdicts" not in [name for name, _ in calls]


def test_a_failed_check_fails_the_update_so_the_article_is_retried():
    import pytest

    def broken(messages):
        raise RuntimeError("timeout")

    with pytest.raises(RuntimeError, match="timeout"):
        _update(_fake(broken)[0])


def test_another_fact_from_the_disputed_sentence_is_kept():
    article = ARTICLE.replace(
        "and prosecutors say Nashiri planned it.",
        "and Smith dismissed the charges against Nashiri.",
    )
    dismissal = ArticleClaim(
        claim="Smith dismissed the charges against Nashiri.",
        section="## Career",
        quote="Smith dismissed the charges against Nashiri",
    )
    generate, _ = _fake(_conflict(), claims=(HEARING, dismissal))
    new = update_from_facts(
        "people",
        "Smith",
        {"text": PROFILE},
        article,
        "a9",
        model="m",
        generate=generate,
        date_check=True,
    )
    assert "Another report differs" in new["text"]
    assert "Smith dismissed the charges against Nashiri.^[a9]" in new["text"]


def test_a_made_up_third_date_is_not_written():
    pair = date_pairs(rows_from_profile(PROFILE), ARTICLE, set())[0]
    text = dispute_text("The USS Cole bombing happened in 1998.", pair, "a9")
    assert "1998" not in text and "Oct. 12, 1999" in text


def test_the_check_is_off_unless_turned_on(monkeypatch):
    generate, calls = _fake(_conflict())
    _update(generate, date_check=False)
    assert "DateVerdicts" not in [name for name, _ in calls]
    monkeypatch.delenv("HINBOX_FACT_DATE_CHECK", raising=False)
    assert not date_check_enabled()
    monkeypatch.setenv("HINBOX_FACT_DATE_CHECK", "off")
    assert not date_check_enabled()
    monkeypatch.setenv("HINBOX_FACT_DATE_CHECK", "on")
    assert date_check_enabled() and not date_check_enabled(False)


def test_publication_dates_are_not_event_dates():
    # When the source was written says nothing about when the plea happened
    assert not dates_differ(
        "Khadr pleaded guilty in October 2010.",
        "The February 2011 article said Khadr pleaded guilty in October 2010.",
    )
    assert not dates_differ(
        "Pohl met Mohammed in May 2012.",
        "The article, published April 28, 2012, said Pohl met Mohammed in May 2012.",
    )
    assert dates_differ(
        "Khadr pleaded guilty in October 2011.",
        "The May 1, 2010 article said Khadr pleaded guilty in October 2010.",
    )


def test_a_re_cited_row_is_still_date_checked():
    from src.engine.fact_list import ClaimVerdict

    assignment = ArticleClaim(
        claim="In July 2014, Smith was assigned to the case against Nashiri.",
        section="## Career",
        quote="Judge Smith ruled on Tuesday",
        source="court record",
    )

    def generate(messages, response_model, model, max_tokens):
        if response_model is ArticleClaims:
            return ArticleClaims(claims=[assignment])
        if response_model is ClaimVerdicts:
            return ClaimVerdicts(
                verdicts=[
                    ClaimVerdict(claim=1, verdict="same", row="f1", new_evidence=True)
                ]
            )
        if response_model is DateVerdicts:
            return _conflict()(messages)
        return SectionPlacements(placements=[])

    new = _update(generate)
    recited = OLD.replace("^[a1]", "^[a1, a9]")
    assert f"{recited} Another report differs: The USS Cole" in new["text"]


def test_a_conflict_must_name_its_event():
    answer = lambda m: DateVerdicts(  # noqa: E731
        verdicts=[DateVerdict(pair=1, verdict="conflict", sentence="It was in 1999.")]
    )
    new = _update(_fake(answer)[0])
    assert "Another report" not in new["text"]


def test_two_shared_words_need_the_same_subject():
    profile = (
        "# Nashiri\n\n## Detention\n\n"
        "In November 2009, aides said Nashiri might be tried in Charleston.^[a1]\n"
    )
    article = "President Bush had Nashiri moved to Guantánamo in September 2006."
    assert not date_pairs(rows_from_profile(profile), article, set())


def test_the_dates_must_sit_next_to_shared_words():
    # The hearing's date and the grenade's date: one sentence names both
    # events, but the 2010 sits by "hearing" and the 2002 by "grenade"
    hearing = (
        "At the May 2010 suppression hearing, prosecutors argued that Khadr "
        "confessed to throwing the grenade that killed Christopher Speer."
    )
    grenade = "Khadr admitted to throwing a grenade in July 2002 that killed a soldier."
    assert not dates_share_context(grenade, hearing)
    assert dates_share_context(
        "Khadr pleaded guilty to war crimes in October 2010.",
        "He had pleaded guilty to war crimes in 2011 in exchange for return to Canada.",
    )
