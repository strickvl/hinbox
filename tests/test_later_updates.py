"""Later updates versus real disputes (``src/engine/later_updates.py`` and
``later_update()`` in ``src/engine/edit_guards.py``)."""

import datetime as dt

from src.engine.edit_guards import guard_edits, later_update
from src.engine.later_updates import (
    article_dates,
    article_side,
    as_update,
    dated_apart,
    is_time_varying,
    own_citations,
    same_observation_day,
    to_date,
)
from src.engine.profile_edits import ProfileEdit

MARCH = "As of March 25, 2013, 28 captives were classified as hunger strikers.^[m25]"
CAPTURE = "Nashiri was captured in Dubai in November 2002.^[c1]"
PROFILE = f"## Hunger strike\n\n{MARCH}\n\n## Capture\n\n{CAPTURE}\n"
DATES = {"m25": "2013-03-25", "c1": "2013-03-25", "a17": "2013-04-17"}
APRIL = (
    "Sources disagree on the number of hunger strikers; while the profile "
    "states 28 captives were hunger strikers, the article reports that as of "
    "April 16, 2013, 59 captives were classified as hunger strikers.^[a17]"
)


def _add(text: str, anchor: str = "END") -> ProfileEdit:
    return ProfileEdit(op="add", anchor=anchor, text=text)


def test_counts_schedules_and_posts_change_over_time():
    assert is_time_varying("the number of hunger strikers rose to 59")
    assert is_time_varying("the judge pushed the trial date to Dec. 4")
    assert is_time_varying("the sources disagree about the judge's identity")
    # One-off past events don't change; a different year is a real dispute
    assert not is_time_varying("Nashiri was captured in November 2002")
    assert not is_time_varying("the sources disagree on the date of the conviction")
    # When a hunger strike began is a past event, how many are on strike isn't
    assert not is_time_varying("the hunger strike began on March 11, 2013")
    assert is_time_varying("since the strike began, 59 captives were hunger strikers")
    assert not is_time_varying("the attacks killed 2,974 people")
    # A post named next to a one-off event is about the event
    assert not is_time_varying("the current judge says Nashiri was captured in 2002")


def test_article_side_finds_the_article_s_own_words():
    # The Qwen wording: the article's side comes last
    assert article_side(APRIL.split("^")[0]) == (
        "says",
        "as of April 16, 2013, 59 captives were classified as hunger strikers",
        None,
    )
    # A dated source dates the count
    dated = (
        "Sources disagree on the trial date; while the profile cites Sept. 2, "
        "a March 4, 2014 report indicates the trial was pushed to Dec. 4"
    )
    assert article_side(dated) == (
        "says",
        "the trial was pushed to Dec. 4",
        "March 4, 2014",
    )
    # "with the article stating ..., while the profile ..."
    stating = (
        "The sources disagree on the trial date, with the article stating that "
        "Judge Pohl postponed the trial until February 2015, while the profile "
        "states September 2014"
    )
    assert article_side(stating) == (
        "says",
        "Judge Pohl postponed the trial until February 2015",
        None,
    )
    # The Luna wording: the article's side comes first
    luna = (
        "The article identifies Army Col. James L. Pohl as the judge and says "
        "the trial would not begin before 2020; the sources disagree about the "
        "judge's identity"
    )
    verb, side, _ = article_side(luna)
    assert verb == "identifies" and side.startswith("Army Col. James L. Pohl")
    # A side that still compares the two versions isn't the article's alone
    assert (
        article_side(
            "The article dates the order to 2009 rather than 2008; the sources disagree"
        )
        is None
    )


def test_as_update_dates_the_entry():
    april = dt.date(2013, 4, 17)
    assert (
        as_update("says", "the trial was pushed to Dec. 4", None, april, "^[a]")
        == "As of April 2013, the trial was pushed to Dec. 4.^[a]"
    )
    # Already dated: left as it is
    assert (
        as_update(
            "says", "on May 16, 2013, 102 men were on strike", None, april, "^[a]"
        )
        == "On May 16, 2013, 102 men were on strike.^[a]"
    )
    # The source's own date wins over the article's
    assert as_update(
        "says", "the trial was pushed", "March 4, 2014", april, "^[a]^[a]"
    ) == ("As of March 4, 2014, the trial was pushed.^[a]")
    assert as_update("identifies", "Pohl as the judge", None, april, "^[a]") == (
        "An April 2013 report identifies Pohl as the judge.^[a]"
    )


def test_a_changed_count_from_a_later_article_becomes_a_timeline_entry():
    edits, report = guard_edits(
        PROFILE, [_add(APRIL, anchor=MARCH)], DATES["a17"], DATES
    )
    assert edits[0].text == (
        "As of April 16, 2013, 59 captives were classified as hunger strikers.^[a17]"
    )
    assert report["later_updates"] and not report["disputes"]
    # The anchor isn't needed: the disputed sentence is found by its values
    edits, _ = guard_edits(PROFILE, [_add(APRIL)], DATES["a17"], DATES)
    assert edits[0].text.startswith("As of April 16, 2013, 59 captives")


def test_without_dates_or_from_the_same_day_the_dispute_stays():
    edits, report = guard_edits(PROFILE, [_add(APRIL, anchor=MARCH)])
    assert edits[0].text == APRIL and not report["later_updates"]
    same_day = {**DATES, "a17": "2013-03-25"}
    edits, _ = guard_edits(PROFILE, [_add(APRIL, anchor=MARCH)], "2013-03-25", same_day)
    assert edits[0].text == APRIL


def test_a_real_dispute_about_a_past_event_stays():
    conflict = (
        "Sources disagree on the capture; while the profile states November "
        "2002, the article reports that Nashiri was captured in Dubai in "
        "November 2003.^[a17]"
    )
    edits, report = guard_edits(
        PROFILE, [_add(conflict, anchor=CAPTURE)], DATES["a17"], DATES
    )
    assert edits[0].text == conflict and not report["later_updates"]


def test_only_dispute_wording_is_rewritten():
    # An ordinary new count anchored on the old one is news, left as it is
    news = "A April 17 report said 59 captives were hunger strikers.^[a17]"
    profile = [MARCH]
    assert later_update(news, profile, dt.date(2013, 4, 17), {}, MARCH) is None


def test_dates_by_article_id():
    assert to_date(1366171200) == dt.date(2013, 4, 17)  # Unix seconds
    assert to_date("2013-04-17T00:00:00") == dt.date(2013, 4, 17)
    assert to_date(None) is None
    dates = {k: to_date(v) for k, v in DATES.items()}
    assert dated_apart(dates["a17"], [MARCH], dates)
    assert not dated_apart(dates["m25"], [MARCH], dates)
    assert not dated_apart(dates["a17"], ["Uncited sentence."], dates)
    entity = {"articles": [{"article_id": "m25", "article_published_date": 1}]}
    assert article_dates(entity, "a17", 2) == {"m25": 1, "a17": 2}


def test_the_new_article_s_clause_is_picked_not_the_last_one():
    claim = (
        "Sources disagree on hunger-striker counts: the article reports that 59 "
        "men were on hunger strike, while a March 25, 2013 report states that "
        "28 men were on hunger strike"
    )
    assert article_side(claim) == ("says", "59 men were on hunger strike", None)
    # Two reports and neither is "the article": no way to tell, left alone
    ambiguous = (
        "Sources disagree on the count: a March 25, 2013 report states that 28 "
        "men were on strike, while an April 2, 2013 report states that 59 men "
        "were on strike"
    )
    assert article_side(ambiguous) is None


def test_two_counts_for_the_same_named_day_stay_a_dispute():
    same_day = (
        "Sources disagree on the number of hunger strikers; while the profile "
        "states 28, the article reports that on March 25, 2013, 30 captives "
        "were classified as hunger strikers.^[a17]"
    )
    edits, report = guard_edits(
        PROFILE, [_add(same_day, anchor=MARCH)], DATES["a17"], DATES
    )
    assert edits[0].text == same_day and not report["later_updates"]


def test_a_year_apart_is_not_the_same_day():
    old = "The trial was scheduled for Sept. 2, 2014.^[m25]"
    assert not same_observation_day(
        "it is now scheduled for Sept. 2, 2015", None, [old]
    )
    assert same_observation_day("it is now scheduled for Sept. 2", None, [old])


def test_the_rewrite_keeps_only_the_article_s_citations():
    both = (
        "Sources disagree on the count; while the profile states 28^[m25], the "
        "article reports that as of April 16, 2013, 59 captives were hunger "
        "strikers.^[a17]"
    )
    edits, _ = guard_edits(PROFILE, [_add(both, anchor=MARCH)], DATES["a17"], DATES)
    assert edits[0].text.endswith("hunger strikers.^[a17]")
    assert own_citations("x^[m25]", [MARCH]) == "^[m25]"  # nothing else to keep
