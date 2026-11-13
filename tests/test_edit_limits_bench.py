"""Replaying a run's edit replies with the limits on (``scripts/edit_limits_bench.py``)."""

import json
from types import SimpleNamespace
from typing import Any, Dict, List

from scripts import edit_limits_bench as bench
from src.engine import edit_limits as el
from src.engine.profile_edits import ProfileEditList

PROFILE = "Born in Leeds.^[a1] Held since 2002.^[a2]"
LONG = "".join(f"Fact number {i} about the camp.^[a9] " for i in range(80))


def _span(n: int, reply: str, finish: str, name: str = "ProfileEditList") -> List[Dict]:
    user = f"Existing profile:\n{'-' * 16}\n{PROFILE}\n\nNew article (ID: a9):\n{'-' * 16}\nText {n}"
    messages = [
        {"role": "system", "content": "Edit."},
        {"role": "user", "content": user},
    ]
    params = {"temperature": 0, "max_tokens": 16384,
              "response_format": {"type": "json_schema", "json_schema": {"name": name}}}  # fmt: skip
    return [
        {"context": {"span_id": f"p{n}", "trace_id": "t"}, "parent_id": None,
         "attributes": {"hinbox.entity_name": f"Camp {n}", "hinbox.article_id": "a9"}},
        {"context": {"span_id": f"s{n}{finish}", "trace_id": "t"}, "parent_id": f"p{n}",
         "attributes": {
             "openinference.span.kind": "LLM",
             "hinbox.step": "merge.locations/profile.update",
             "llm.invocation_parameters": json.dumps(params),
             "input.value": json.dumps({"messages": messages}),
             "llm.output_messages.0.message.role": "assistant",
             "llm.output_messages.0.message.content": reply,
             "gen_ai.response.finish_reasons": [finish],
         }},
    ]  # fmt: skip


def _reply(*texts: str, anchor: str = "END") -> str:
    edits = [{"op": "add", "anchor": anchor, "text": t} for t in texts]
    return ProfileEditList.model_validate({"edits": edits}).model_dump_json()


def _spans() -> List[Dict]:
    return [
        *_span(1, '{"disagreements": [{"profile_sentence": "Held-a999-a999', "length"),
        *_span(1, '{"disagreements": [{"profile_sentence": "Held-a999-a999', "length"),
        *_span(2, _reply("Short.^[a9]"), "stop"),
        *_span(3, _reply(LONG), "stop"),
        *_span(4, "is_match: yes", "stop", name="MatchCheckResult"),
    ]


def _picked(tmp_path, runaway: int, long: int) -> List[Dict]:
    path = tmp_path / "run-1.jsonl"
    path.write_text("".join(json.dumps(s) + "\n" for s in _spans()))
    calls, notes = bench.scan([str(path)])
    assert [c["group"] for c in calls] == ["runaway", "runaway", "long"]
    assert calls[0]["key"] == calls[1]["key"] != calls[2]["key"]
    return bench.load(bench.pick(calls, runaway, long), notes)


def test_scan_pick_and_load_find_the_two_groups(tmp_path):
    picked = _picked(tmp_path, 5, 5)
    assert [(c["group"], c["entity"], c["article"], c["of"]) for c in picked] == [
        ("runaway", "Camp 1", "a9", 1),
        ("long", "Camp 3", "a9", 1),
    ]
    assert picked[0]["messages"][0] == {"role": "system", "content": "Edit."}
    assert picked[0]["params"]["max_tokens"] == 16384


def test_compare_counts_what_the_limits_left(tmp_path):
    call = _picked(tmp_path, 0, 5)[0]
    cut = LONG[: el.TEXT_CHARS]
    row = bench.compare(call, _reply(cut, "Later fact.^[a9]"))
    assert row["parsed"] and row["edits"] == 2 and row["edits_kept"] == 2
    assert row["cut"][0]["field"] == "edits[0].text"
    kept = row["new_edits"][0]["text"]
    assert kept.endswith("^[a9]") and len(kept) < el.TEXT_CHARS
    assert row["old_sentences"] == 80
    assert 50 < row["old_sentences_kept"] < 80
    assert row["placed"] == {"applied": 2, "appended": 0, "skipped": 0}
    assert row["edits_after_guards"] == 2 and row["notes"] == 0
    assert bench.compare(call, '{"edits": [{"op"') == {
        "parsed": False,
        "old_parsed": True,
        "reply_text": '{"edits": [{"op"',
    }


def test_replay_sends_the_recorded_request_with_the_limited_schema(
    tmp_path, monkeypatch
):
    sent: Dict[str, Any] = {}

    def create(**kwargs: Any) -> Any:
        sent.update(kwargs)
        message = SimpleNamespace(content=_reply("Moved in 2008.^[a9]"))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=SimpleNamespace(completion_tokens=12),
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    monkeypatch.setattr("src.utils.llm._create_raw_client", lambda target: client)
    call = _picked(tmp_path, 5, 0)[0]
    row = bench.replay(call, "ollama/test", limits=True)
    assert sent["messages"] == call["messages"] and sent["max_tokens"] == 16384
    schema = sent["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["edits"]["maxItems"] == el.MAX_EDITS
    assert row["finish"] == "stop" and row["tokens"] == 12 and row["edits_kept"] == 1
    assert row["old_finish"] == ["length"] and not row["old_parsed"]
    lines = bench.summary([row])
    assert "runaway: 1 prompts replayed" in lines[0]
    assert "ended normally and readable: 1" in lines[1]
    assert "Moved in 2008" in bench.review([row])
    bench.replay(call, "ollama/test", limits=False)
    assert "maxItems" not in json.dumps(sent["response_format"])
