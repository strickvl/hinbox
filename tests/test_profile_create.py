import json
from typing import Dict, List

from src.engine import profiles
from src.engine.profile_create import (
    accept_headings,
    code_fixes,
    create_with_checks,
    model_problems,
)

AID = "a" * 36
GOOD = (
    "John Smith is a lawyer for the detainee and spoke to reporters on Monday.\n\n"
    f"### Role\nHe filed a motion asking the judge to delay the hearing^[{AID}]."
)


def _draft(text: str, tags=("lawyer", "defense")) -> str:
    return json.dumps({"text": text, "tags": list(tags), "confidence": 0.8})


def test_code_fixes_cites_uncited_paragraphs_and_snaps_ids():
    text = "First paragraph with a fact.\n\n### Role\nSecond fact^[a-garbled-id]."
    profile, fixed = code_fixes({"text": text, "tags": ["x"]}, AID, "people")
    assert profile["text"] == (
        f"First paragraph with a fact.^[{AID}]\n\n### Role\nSecond fact^[{AID}]."
    )
    assert profile["sources"] == [AID]
    assert profile["tags"] == ["x", "person"]
    assert "citations_fixed:2" in fixed


def test_code_fixes_leaves_good_profile_alone():
    profile, fixed = code_fixes(json.loads(_draft(GOOD)), AID, "people")
    assert fixed == ["citations_fixed:1"]  # the uncited first paragraph
    assert profile["text"].count(f"^[{AID}]") == 2


def test_model_problems():
    assert model_problems(None) == ["unparseable"]
    assert model_problems({"text": "short"}) == ["too_short"]
    assert model_problems({"text": "x " * 80}) == ["no_headings"]
    assert model_problems({"text": GOOD}) == []


def test_accept_headings_refuses_changed_sentences():
    draft = "Fact one^[x].\n\nFact two^[x]."
    assert accept_headings(
        draft, "### Background\nFact one^[x].\n\n### Role\nFact two^[x]."
    )
    assert (
        accept_headings(draft, "### Background\nFact one^[x].\n\nFact 2^[x].") is None
    )
    assert accept_headings(draft, draft) is None  # no heading added


def _run(drafts: List[str], text_reply: str = ""):
    calls: Dict[str, int] = {"draft": 0, "text": 0}

    def draft(_msgs):
        calls["draft"] += 1
        return drafts[min(calls["draft"], len(drafts)) - 1]

    def write_text(_msgs):
        calls["text"] += 1
        return text_reply

    profile, history = create_with_checks(
        draft=draft,
        write_text=write_text,
        generation_messages=[{"role": "user", "content": "article"}],
        article_id=AID,
        entity_type="people",
    )
    return profile, history, calls


def test_good_draft_needs_one_call():
    profile, history, calls = _run([_draft(GOOD)])
    assert calls == {"draft": 1, "text": 0}
    assert history[-1]["valid"] is True
    assert profile["text"].startswith("John Smith")


def test_missing_headings_gets_one_repair():
    body = "John Smith is a lawyer for the detainee and spoke to reporters on Monday."
    fixed = f"{body}^[{AID}]"
    profile, history, calls = _run(
        [_draft(body + " " + body)], text_reply=f"### Background\n{body} {fixed}"
    )
    assert calls == {"draft": 1, "text": 1}
    assert profile["text"].startswith("### Background")
    assert "headings_added" in history[-1]["fixed"]


def test_rewritten_repair_is_refused_and_draft_kept():
    body = "John Smith is a lawyer for the detainee and spoke to reporters on Monday."
    profile, history, calls = _run(
        [_draft(body + " " + body)],
        text_reply="### Background\nSomething entirely different.",
    )
    assert profile["text"] == f"{body} {body}^[{AID}]"
    assert "headings_repair_refused" in history[-1]["fixed"]
    assert history[-1]["valid"] is False


def test_empty_draft_gets_one_fresh_draft():
    profile, history, calls = _run(["{}", _draft(GOOD)])
    assert calls["draft"] == 2
    assert profile["text"].startswith("John Smith")


def test_generate_profile_uses_check_approach(monkeypatch):
    monkeypatch.setattr(
        profiles, "_generate_initial_profile_text", lambda *a, **k: _draft(GOOD)
    )

    def no_reflection(*a, **k):
        raise AssertionError("reflection must not run")

    monkeypatch.setattr(profiles, "iterative_improve", no_reflection)
    result, history = profiles.generate_profile(
        "people", "John Smith", "article text", AID, approach="check"
    )
    assert result["sources"] == [AID]
    assert f"^[{AID}]" in result["text"]
    assert history[-1]["check"] == "code"


def test_count_calls_reads_both_histories():
    from src.evals.profile_create import count_calls

    reflect = [
        {"valid": False, "reasoning": "no Background"},
        {"valid": False, "reasoning": "no Background"},
        {"valid": True, "reasoning": "ok"},
    ]
    assert count_calls("reflect", reflect) == 1 + 3 + 2
    assert (
        count_calls("reflect", [{"valid": False, "reasoning": "Reflection failed: x"}])
        == 2
    )
    check = [{"check": "code", "problems": ["no_headings"], "model_calls": 2}]
    assert count_calls("check", check) == 2
    redraft = [
        {"problems": ["too_short"]},
        {"problems": ["too_short"], "model_calls": 2},
    ]
    assert count_calls("check", redraft) == 2


def test_score_profile_flags_wrong_ids_and_novel_names():
    from src.evals.profile_create import score_profile

    article = "Judge James Pohl recessed the hearing on Monday until 2014."
    text = (
        "### Role\nPohl recessed the hearing on Monday^[a1].\n\n"
        "He was appointed by Obama in 2009^[zz]."
    )
    s = score_profile(text, "a1", article)
    assert s["wrong_ids"] == 1 and s["headings"] == 1 and s["uncited_paragraphs"] == 0
    assert s["names_not_in_article"] == ["2009", "obama"]


def test_heading_block_with_uncited_prose_gets_cited():
    text = "### Background\nUncited fact one.\n\n### Role\nUncited fact two."
    profile, _ = code_fixes({"text": text, "tags": ["a", "b"]}, AID, "people")
    assert profile["text"] == (
        f"### Background\nUncited fact one.^[{AID}]\n\n"
        f"### Role\nUncited fact two.^[{AID}]"
    )


def test_redraft_uses_the_repair_budget():
    body = "John Smith is a lawyer for the detainee and spoke to reporters on Monday."
    profile, history, calls = _run(["{}", _draft(body + " " + body)], "### X\n" + body)
    assert calls == {"draft": 2, "text": 0}
    assert history[-1]["model_calls"] == 2
