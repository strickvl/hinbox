"""Tests for the citation check that lists differences and lets code decide
(``src/utils/grounding_differences.py``). Every name and sentence here is
invented."""

import json
from typing import List
from unittest.mock import patch

from src.citecheck.__main__ import main
from src.citecheck.items import article_split, read_jsonl, write_jsonl
from src.citecheck.score import read_reply
from src.utils import grounding
from src.utils import grounding_differences as gd
from src.utils.grounding import SupportLevel, judge_claims

ARTICLE = "Ann Lee will be flown to the base in May, officials said."


def test_prompt_keeps_rules_and_replaces_reply_paragraph(monkeypatch):
    plain = grounding.grounding_system_prompt("strict")
    assert "first write the reasoning" in plain and "differences" not in plain
    monkeypatch.setattr(gd, "GROUNDING_DECIDE", "differences")
    for rules in ("strict", "standard"):
        prompt = grounding.grounding_system_prompt(rules)
        assert prompt.endswith(gd.DIFFERENCES_INSTRUCTION)
        assert "first write the reasoning" not in prompt
        assert ("dropped attribution:" in prompt) == (rules == "strict")
    assert grounding.grounding_judge("cloud").endswith("(decide=differences)")


def test_a_listed_difference_rules_out_supported():
    assert gd.decided_level(SupportLevel.SUPPORTED, ["'was' for 'will be'"]) == (
        SupportLevel.PARTIAL
    )
    assert gd.decided_level("supported", ["none"]) == "supported"
    assert gd.decided_level("supported", []) == "supported"
    assert gd.decided_level("not_supported", ["x"]) == "not_supported"
    assert gd.reasoning_with(["date"], "Says May.") == "Differences: date. Says May."
    assert gd.reasoning_with([], "Says May.") == "Says May."


def test_judge_call_asks_for_differences_and_applies_them(monkeypatch):
    monkeypatch.setattr(gd, "GROUNDING_DECIDE", "differences")
    reply = [
        gd.ClaimVerdictWithDifferences(
            claim_number=1,
            differences=["claim says flown; article says will be flown"],
            reasoning="Article: 'will be flown'.",
            support_level=SupportLevel.SUPPORTED,
        ),
        gd.ClaimVerdictWithDifferences(
            claim_number=2, differences=[], support_level=SupportLevel.SUPPORTED
        ),
    ]
    with patch("src.utils.llm.cloud_generation", return_value=reply) as gen:
        out = judge_claims(
            article_id="a1",
            article_text=ARTICLE,
            claims=[("Ann Lee", "Ann Lee was flown to the base."), ("Ann Lee", "x")],
            model_type="cloud",
            ask_passage=False,
            model="openai/test",
        )
    assert (
        gen.call_args.kwargs["response_model"] == List[gd.ClaimVerdictWithDifferences]
    )
    assert out[0][0] == SupportLevel.PARTIAL and out[0][1].startswith("Differences:")
    assert out[1][0] == SupportLevel.SUPPORTED


def test_raw_replies_json_and_text():
    js = json.dumps(
        {
            "items": [
                {
                    "claim_number": 1,
                    "differences": ["adds 'first'"],
                    "support_level": "supported",
                },
                {"claim_number": 2, "differences": [], "support_level": "supported"},
            ]
        }
    )
    assert read_reply(js, 2) == ["partial", "supported"]
    text = (
        "1.\nDifferences:\n- date not in article\nReasoning: close.\nSupport_level: supported\n"
        "2.\nDifferences: none\nReasoning: same.\nSupport_level: supported"
    )
    assert read_reply(text, 2) == ["partial", "supported"]
    # an older reply that only mentions "the difference" keeps its verdict
    old = "1. Reasoning: the only difference: wording.\nSupport_level: supported"
    assert read_reply(old, 1) == ["supported"]


def test_dev_set_uses_agreed_labels_off_the_test_articles(tmp_path, capsys):
    ids = [f"art{n}" for n in range(60)]
    off_test = [a for a in ids if article_split(a) != "test"][:4]
    on_test = [a for a in ids if article_split(a) == "test"][:1]
    rows = []
    for n, aid in enumerate(off_test + on_test):
        level = "supported" if n % 2 else "partial"
        for t in ("t1", "t2"):
            rows.append(
                {
                    "item_id": f"i{n}",
                    "teacher": t,
                    "article_id": aid,
                    "entity_name": "Ann Lee",
                    "claim": f"Claim {n}.",
                    "level": level,
                    "reasoning": "x",
                    "passages": ["q"],
                    "rules": grounding.GROUNDING_RULES,
                }
            )
    write_jsonl(str(tmp_path / "teacher_verdicts.jsonl"), rows)
    main(
        [
            "dev-set",
            "--teach-dir",
            str(tmp_path),
            "--out",
            str(tmp_path / "dev"),
            "--teachers",
            "t1,t2",
            "--n",
            "0",
        ]
    )
    items = read_jsonl(str(tmp_path / "dev" / "dev_items.jsonl"))
    labels = {
        r["item_id"]: r["label"]
        for r in read_jsonl(str(tmp_path / "dev" / "dev_labels.jsonl"))
    }
    assert {r["article_id"] for r in items} == set(off_test)
    assert labels == {"i0": "no", "i1": "yes", "i2": "no", "i3": "yes"}
