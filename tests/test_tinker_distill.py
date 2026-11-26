"""The distill step's checks in ``scripts/tinker_citecheck.py`` (no Tinker)."""

import json

from scripts.tinker_citecheck import (
    _parts,
    distilled_row,
    matches_gold,
    reply_levels,
    row_key,
)


def _reply(*levels: str) -> str:
    items = [
        {"claim_number": i + 1, "reasoning": "r", "support_level": lv}
        for i, lv in enumerate(levels)
    ]
    return json.dumps({"items": items})


def test_reply_levels_reads_json_with_fences_and_words_around() -> None:
    text = "Here it is:\n```json\n" + _reply("supported", "Not Supported") + "\n```"
    assert reply_levels(text) == {1: "supported", 2: "not_supported"}


def test_reply_levels_empty_for_plain_text_or_cut_off() -> None:
    assert reply_levels("1. Reasoning: ok. Support_level: supported") == {}
    assert reply_levels(_reply("supported")[:-5]) == {}


def test_matches_gold_needs_every_claim_and_no_extra() -> None:
    gold = reply_levels(_reply("supported", "partial"))
    assert matches_gold(_reply("supported", "partial"), gold)
    assert not matches_gold(_reply("supported"), gold)
    assert not matches_gold(_reply("supported", "partial", "supported"), gold)
    assert not matches_gold(_reply("supported", "not_supported"), gold)
    assert not matches_gold(_reply("supported"), {})


def test_loose_counts_partial_as_not_supported() -> None:
    gold = reply_levels(_reply("partial", "supported"))
    assert matches_gold(_reply("not_supported", "supported"), gold, loose=True)
    assert not matches_gold(_reply("supported", "supported"), gold, loose=True)


def test_distilled_row_keeps_reasoning_before_answer() -> None:
    prompt = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    row = distilled_row(prompt, "think", _reply("supported"), {"sample": 0})
    reply = row["messages"][-1]
    assert row["messages"][:2] == prompt and reply["role"] == "assistant"
    assert [p["type"] for p in reply["content"]] == ["thinking", "text"]
    assert _parts(reply) == ("think", _reply("supported"))
    assert [
        p["type"] for p in distilled_row(prompt, "", "x", {})["messages"][-1]["content"]
    ] == ["text"]


def test_row_key_ignores_the_reply() -> None:
    prompt = [{"role": "user", "content": "u"}]
    assert row_key(prompt) == row_key(list(prompt))
    assert row_key(prompt) != row_key([{"role": "user", "content": "v"}])
