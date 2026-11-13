"""Trying the note guard on a run's edit replies (``scripts/profile_notes_bench.py``)."""

import json
from typing import Any, Dict, List

from scripts import profile_notes_bench as bench
from src.engine.dispute_check import DisputeVerdict, DisputeVerdicts
from src.engine.profile_edits import ProfileEditList

PROFILE = "The camp held 93 captives.^[a1] It opened in 2002.^[a2]"
NOTE = (
    "The article reports 41 captives at the camp, conflicting with the "
    "profile's count of 93.^[a9]"
)
PLAIN = "A new commander arrived in May.^[a9]"


def _span(n: int, *texts: str, finish: str = "stop") -> Dict[str, Any]:
    user = f"Existing profile:\n{'-' * 16}\n{PROFILE}\n\nNew article (ID: a9):\n{'-' * 16}\nText {n}"
    messages = [
        {"role": "system", "content": 'You maintain a profile of the location "Camp Nine". A new'},
        {"role": "user", "content": user},
    ]  # fmt: skip
    edits = [{"op": "add", "anchor": "END", "text": t} for t in texts]
    reply = ProfileEditList.model_validate({"edits": edits}).model_dump_json()
    params = {"response_format": {"json_schema": {"name": "ProfileEditList"}}}
    return {
        "context": {"span_id": f"s{n}", "trace_id": "t"}, "parent_id": None,
        "attributes": {
            "llm.invocation_parameters": json.dumps(params),
            "input.value": json.dumps({"messages": messages}),
            "llm.output_messages.0.message.role": "assistant",
            "llm.output_messages.0.message.content": reply,
            "gen_ai.response.finish_reasons": [finish],
        },
    }  # fmt: skip


def _places(tmp_path) -> List:
    spans = [_span(1, PLAIN), _span(2, f"{PLAIN} {NOTE}"), _span(3, NOTE, finish="length"),
             _span(2, f"{PLAIN} {NOTE}")]  # fmt: skip
    path = tmp_path / "run-1.jsonl"
    path.write_text("".join(json.dumps(s) + "\n" for s in spans))
    places, totals = bench.scan([str(path)])
    assert totals == {"replies": 2, "new sentences": 3, "notes": 1, "replies with a note": 1,
                      "replies to a prompt already counted": 1}  # fmt: skip
    return places


def test_only_replies_with_a_note_are_taken_and_the_note_is_rewritten(tmp_path):
    places = _places(tmp_path)
    assert places == [(str(tmp_path / "run-1.jsonl"), 1)]
    (call,) = bench.load(places)
    ((old, how, new),) = bench.guarded(call, {})
    assert old == NOTE and how == "note, plain"
    assert new == "A report notes 41 captives at the camp.^[a9]"
    assert "still naming the profile afterwards: 0" in bench.tally([[old, how, new]])
    was = bench.before(call, {})
    assert was == {NOTE: ["kept as written"]}
    assert bench.moves([[old, how, new]], was) == [("kept as written", "note, plain")]
    # The same note twice: before, the second was a repeat of the first
    twice = {NOTE: ["kept as written", "repeat, dropped"]}
    got = [[NOTE, "repeat, dropped", ""], [NOTE, "note, plain", new]]
    assert bench.moves(got, twice) == [
        ("kept as written", "note, plain"),
        ("repeat, dropped", "repeat, dropped"),
    ]


def test_published_dates_date_a_count_that_changes(tmp_path):
    (call,) = bench.load(_places(tmp_path))
    dates = {"a1": "2010-03-01", "a2": "2010-03-01", "a9": "2013-04-20"}
    ((_, how, new),) = bench.guarded(call, dates)
    assert how == "note, plain"
    assert new == "An April 2013 report notes 41 captives at the camp.^[a9]"


def test_with_a_model_its_verdict_is_what_happened(tmp_path):
    (call,) = bench.load(_places(tmp_path))

    def ask(instructions: str, items: str) -> DisputeVerdicts:
        return DisputeVerdicts(verdicts=[DisputeVerdict(
            item=1, reasoning="one count, two reports", verdict="conflict"
        )])  # fmt: skip

    ((_, how, new),) = bench.guarded(call, {}, ask)
    assert how == "note, conflict"
    assert "another report" in new.lower() and "profile" not in new
