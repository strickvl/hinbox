"""The distill step's checks in ``scripts/tinker_citecheck.py`` (no Tinker)."""

import json

from scripts.tinker_citecheck import (
    _parts,
    as_json_reply,
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


PLAIN = (
    "1. Reasoning: The article says he arrived in 2002. Support_level: supported\n\n"
    "2. **Reasoning:** No passage names the judge. **Support_level:** Not supported"
)


def test_reply_levels_reads_text_replies_in_order() -> None:
    assert reply_levels(PLAIN) == {1: "supported", 2: "not_supported"}
    assert reply_levels("I think it is fine.") == {}


def test_reply_levels_empty_when_cut_off() -> None:
    assert reply_levels(_reply("supported")[:-5]) == {}
    cut = _reply("supported", "partial")[:-3]  # every verdict, no closing brackets
    assert reply_levels(cut) == {}
    assert not matches_gold(cut, {1: "supported", 2: "partial"})


def test_text_reply_kept_as_json() -> None:
    gold = {1: "supported", 2: "not_supported"}
    assert matches_gold(PLAIN, gold)
    rebuilt = json.loads(as_json_reply(PLAIN))["items"]
    assert [i["support_level"] for i in rebuilt] == ["supported", "not_supported"]
    assert [i["claim_number"] for i in rebuilt] == [1, 2]
    assert rebuilt[0]["reasoning"] == "The article says he arrived in 2002."
    assert rebuilt[1]["reasoning"] == "No passage names the judge."
    assert reply_levels(as_json_reply(PLAIN)) == gold
    assert as_json_reply(_reply("partial")) == _reply("partial")


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


def test_missing_or_unknown_level_never_matches() -> None:
    gold = {1: "not_supported"}
    for text in ('{"items": [{"claim_number": 1}]}', _reply("wrong"), '{"items": []}'):
        assert reply_levels(text) == {}
        assert not matches_gold(text, gold, loose=True)


def test_hinted_prompt_adds_labels_to_last_user_message_only() -> None:
    from scripts.tinker_citecheck import HINT_HEADER, hinted_prompt

    prompt = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    asked = hinted_prompt(prompt, {2: "supported", 1: "partial"})
    assert asked[0] == prompt[0] and prompt[1]["content"] == "u"  # not changed
    text = asked[1]["content"]
    assert text.startswith("u\n\n" + HINT_HEADER)
    assert text.index("Claim 1: partial") < text.index("Claim 2: supported")
    parts = [{"type": "text", "text": "u"}]
    listed = hinted_prompt([{"role": "user", "content": parts}], {1: "partial"})
    assert listed[0]["content"][0] == parts[0] and len(listed[0]["content"]) == 2


def test_mentions_hint_catches_leaks_not_ordinary_words() -> None:
    from scripts.tinker_citecheck import mentions_hint

    for leak in (
        "The reference verdict says partial, so",
        "I was told the answer is partial.",
        "Matching the expected verdict.",
        "The hint points to partial.",
    ):
        assert mentions_hint("", leak), leak
    for fine in (
        "He was given a life sentence in 2008, the article says.",
        "The key witness testified; the article hints at a delay.",
        "The claim adds 'first', which the article does not state.",
    ):
        assert not mentions_hint(fine, ""), fine


def test_kept_keys_reads_only_kept_examples(tmp_path) -> None:
    from scripts.tinker_citecheck import kept_keys

    log = tmp_path / "log.jsonl"
    log.write_text(
        json.dumps({"key": "a", "kept": 0})
        + "\n"
        + json.dumps({"key": "b", "kept": None})
        + "\n"
    )
    assert kept_keys(str(log)) == {"a"} and kept_keys("") == set()


def test_hinted_distill_samples_with_hint_and_stores_plain_prompt(
    tmp_path, monkeypatch
) -> None:
    import argparse
    import sys
    import types

    import scripts.tinker_citecheck as tc

    asked = []
    answers = [  # thinking, text: a leak, then a clean match, then a miss
        ("The reference verdict is partial.", _reply("partial")),
        ("'Was' against 'will be' decides it.", _reply("partial")),
        ("Fine.", _reply("supported")),
    ]

    class Renderer:
        def build_generation_prompt(self, messages, effort):
            asked.append(messages)
            return messages

        def parse_response(self, tokens):
            thinking, text = answers[tokens[0]]
            content = [
                {"type": "thinking", "thinking": thinking},
                {"type": "text", "text": text},
            ]
            return {"role": "assistant", "content": content}, None

        def get_stop_sequences(self):
            return []

    class Sampler:
        def sample(self, prompt, num_samples, sampling_params):
            seqs = [types.SimpleNamespace(tokens=[i]) for i in range(num_samples)]
            return types.SimpleNamespace(sequences=seqs)

    fake = types.SimpleNamespace(
        ServiceClient=lambda: types.SimpleNamespace(
            create_sampling_client=lambda **k: Sampler()
        ),
        SamplingParams=lambda **k: k,
    )
    monkeypatch.setitem(sys.modules, "tinker", fake)
    monkeypatch.setattr(tc, "_renderer", lambda model: (Renderer(), None))
    prompt = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    data, out = tmp_path / "hard.jsonl", tmp_path / "hinted.jsonl"
    reply = {"role": "assistant", "content": _reply("partial")}
    data.write_text(json.dumps({"messages": prompt + [reply]}) + "\n")
    out.write_text("")  # skips the renderer weight check
    args = argparse.Namespace(
        model="m",
        checkpoint="base",
        data=str(data),
        out=str(out),
        max_tokens=10,
        temperature=1.0,
        workers=2,
        samples=3,
        effort=0.0,
        loose=True,
        hint=True,
        skip_kept="",
        limit=0,
        seed=6,
    )
    tc.cmd_distill(args)
    assert tc.HINT_HEADER in asked[0][-1]["content"]
    kept = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(kept) == 1 and kept[0]["messages"][:2] == prompt  # no hint stored
    assert tc._parts(kept[0]["messages"][-1])[0].startswith("'Was'")
    assert kept[0]["distill"]["hinted"] is True
    log = json.loads((tmp_path / "hinted.jsonl.log.jsonl").read_text())
    assert log["kept"] == 1 and log["leaked"] == 1


def test_limited_pick_is_the_same_on_every_call() -> None:
    from scripts.tinker_citecheck import limited

    rows = list(range(50))
    pick = limited(rows, 10, 6)
    assert len(pick) == 10 and pick == limited(rows, 10, 6) and rows == list(range(50))
    assert limited(rows, 0, 6) == rows and limited(rows, 99, 6) == rows
