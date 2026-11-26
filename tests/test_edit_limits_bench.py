"""Replaying a run's edit replies with the limits on (``scripts/edit_limits_bench.py``)."""

import json
from types import SimpleNamespace
from typing import Any, Dict, List

from scripts import edit_limits_acceptance as checks
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


def test_claim_review_uses_full_profile_and_only_applied_edits(tmp_path):
    spans = _span(1, "truncated", "length")
    parent = spans[0]
    parent["name"] = "profile.update"
    archived = {
        "existing_profile": {"text": PROFILE + "\nTransferred in 2004.^[a3]"},
        "entity_type": "location",
        "domain": "guantanamo",
        "new_article_id": "a9",
        "guarded_corrections": 0,
    }
    parent["attributes"]["input.value"] = json.dumps(archived)
    path = tmp_path / "trace.jsonl"
    path.write_text("".join(json.dumps(span) + "\n" for span in spans))
    calls, notes = bench.scan([str(path)])
    [call] = bench.load(bench.pick(calls, 1, 0), notes)
    reply = _reply("Returned to court in 2009.^[a9]")
    changes = bench.compare(call, reply)["claim_changes"]
    assert changes["old_kept"] == changes["old_cited"] == 3
    assert [a["claim"] for a in changes["additions"]] == ["Returned to court in 2009"]

    cut = bench.compare(call, _reply("Added fact.^[a9]", anchor="x" * el.ANCHOR_CHARS))
    assert cut["claim_changes"]["additions"] == []


def test_acceptance_requires_supported_novelty_and_flags_bad_attribution():
    after = PROFILE + "\nReceived a hearing in 2009.^[a9]\nMoved in 2010.^[a9]"
    changes = checks.claim_changes(PROFILE, after, "a9")
    first, second = changes["additions"]
    row = {"finish": "stop", "parsed": True, "claim_changes": changes}
    assert checks.acceptance(row, {})["status"] == "unverified"

    reviews = {
        first["id"]: {"verdict": "new_supported"},
        second["id"]: {"verdict": "unsupported", "reason": "The article says 2012"},
    }
    failed = checks.acceptance(row, reviews)
    assert failed["status"] == "fail"
    assert failed["unsupported"] == [{**second, "reason": "The article says 2012"}]
    assert failed["new_supported"] == [first]

    reviews[second["id"]] = {"verdict": "already_covered"}
    assert checks.acceptance(row, reviews)["status"] == "pass"
    reviews[first["id"]] = {"verdict": "off_topic"}
    assert checks.acceptance(row, reviews)["reason"] == (
        "no new supported fact from the incoming article"
    )

    before = PROFILE + "\nArrived in 2001.^[a3]"
    lost = {**row, "claim_changes": checks.claim_changes(before, after, "a9")}
    assert (
        checks.acceptance(lost, reviews)["reason"] == "old cited claims changed or lost"
    )
    uncited = checks.claim_changes(PROFILE, PROFILE + "\nEscaped in 2001.", "a9")
    no_source = checks.acceptance({**row, "claim_changes": uncited}, {})
    assert no_source["unsupported"][0]["reason"] == "missing citation"


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
