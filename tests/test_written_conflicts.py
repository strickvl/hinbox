"""The written page's last stages: disagreements told, sources not narrated,
two people's dates not taken for a disagreement. Made-up notes only (the
repository is public)."""

from src.engine.written_checks import code_check
from src.engine.written_conflicts import tell_conflicts, unnarrate
from src.engine.written_page import Note, Written
from src.engine.written_plan import (
    Conflict,
    LeftOut,
    Plan,
    PlanPart,
    clean_conflicts,
    restore_on_topic,
)
from src.engine.written_revise import Fill, Revisions


def _note(n: int, text: str, date: str = "2005-05-06", tag: str = "about") -> Note:
    return Note(
        n=n,
        key=f"k{n}",
        text=text,
        articles=[f"a{n}"],
        date=date,
        tag=tag,
        keys=[f"k{n}"],
    )


NOTES = [
    _note(1, "Port Elm is a harbour city in Elmland.", "2003-01-02"),
    _note(2, "Police captured Tom Vale in Port Elm on March 5, 2004.", "2004-03-05"),
    _note(3, "Police arrested Tom Vale in Port Elm in May 2005.", "2005-05-06"),
    _note(4, "Ann Roe was named a forever prisoner in 2009.", "2009-02-02"),
    _note(5, "Lee Moss was named a forever prisoner in 2010.", "2010-02-02"),
]
TEXTS = {n.n: n.text for n in NOTES}


def _conflict(one, other, a="2004", b="2005"):
    return Conflict(
        about="x", one_side=a, one_notes=one, other_side=b, other_notes=other
    )


def test_two_peoples_dates_are_no_disagreement():
    kept = clean_conflicts(
        [_conflict([2], [3]), _conflict([4], [5], "2009", "2010")],
        {1, 2, 3, 4, 5}, None, TEXTS, "Port Elm",
    )  # fmt: skip
    assert [c.one_notes for c in kept] == [[2]]


def test_an_untold_disagreement_gets_a_passage_after_its_first_sentence():
    written = [
        Written("Port Elm is a harbour city in Elmland.", [1]),
        Written("Police arrested Tom Vale in Port Elm in May 2005.", [3], "Arrests", 1),
        Written("The harbour is large.", [1], "Arrests", 1),
    ]
    conflict = _conflict([2], [3]).model_dump()
    seen = []

    def ask(msgs, model, max_tokens):
        seen.append(msgs[0]["content"])
        if model is Fill:
            return Fill.model_validate({"sentences": [{
                "text": "One account dates Tom Vale's capture to March 5, 2004; another "
                "gives May 2005.", "notes": [2, 3]}]})  # fmt: skip
        return model.model_validate({"checks": []})

    report = {}
    out = tell_conflicts(written, NOTES, "Port Elm", "place", [conflict], ask, report)
    assert "disagree" in seen[0] and "Tom Vale" not in seen[0][:40]
    assert out[2].text.startswith("One account dates") and out[2].section == "Arrests"
    assert not out[2].dropped and report["conflicts_untold"] == 0
    # Told already: no call
    seen.clear()
    tell_conflicts(out, NOTES, "Port Elm", "place", [conflict], ask, {})
    assert seen == []


def test_narrating_sentences_are_rewritten_or_kept_as_they_were():
    narr = Written(
        "In May 2005, a news account said police arrested Tom Vale in Port Elm.",
        [3], "Arrests", 1,
    )  # fmt: skip
    told = Written(
        "One account dates Tom Vale's capture to March 5, 2004; another gives May 2005.",
        [2, 3], "Arrests", 1,
    )  # fmt: skip
    conflict = _conflict([2], [3]).model_dump()

    def ask(msgs, model, max_tokens):
        if model is Revisions:
            return Revisions.model_validate({"revisions": [{
                "sentence": 1, "starts_with": "In May 2005,",
                "text": "In May 2005, police arrested Tom Vale in Port Elm."}]})  # fmt: skip
        return model.model_validate({"checks": []})

    report = {}
    out = unnarrate([narr, told], NOTES, "Port Elm", [conflict], ask, report)
    assert report["narrating"] == 1  # the disagreement may name its accounts
    assert (
        narr.dropped
        and out[1].text == "In May 2005, police arrested Tom Vale in Port Elm."
    )
    assert not told.dropped

    def fails(msgs, model, max_tokens):
        raise RuntimeError("down")

    again = Written(narr.revised or narr.text, [3], "Arrests", 1)
    out = unnarrate([again], NOTES, "Port Elm", [], fails, {})
    assert out == [again] and not again.dropped  # narrated beats lost


def test_an_article_date_on_an_older_report_is_flagged():
    notes = [_note(1, "The Senate report said the agency held 119 men.", "2018-04-02")]
    w = Written(
        "In April 2018, the Senate report said the agency held 119 men.", [1], "S"
    )
    ok = Written("In April 2018, the agency said it held 119 men.", [1], "S")
    code_check([w, ok], notes, "The agency")
    assert "older report" in w.dropped and not ok.dropped


def test_a_note_naming_the_persons_surname_goes_back_in():
    by_n = {
        1: _note(1, "Tom Vale was born in Port Elm."),
        2: _note(2, "His lawyer said Vale was a cook, not a fighter.", tag="context"),
        3: _note(3, "Ann Roe ran the harbour.", tag="other"),
    }
    plan = Plan(
        identity="", why_it_matters="", opening_notes=[1],
        parts=[PlanPart(heading="Life", covers="", notes=[1])],
        left_out=[LeftOut(note=2, why="someone else"), LeftOut(note=3, why="someone else")],
    )  # fmt: skip
    restore_on_topic(plan, by_n, "Tom Vale", "person")
    assert plan.parts[0].notes == [1, 2] and [o.note for o in plan.left_out] == [3]
