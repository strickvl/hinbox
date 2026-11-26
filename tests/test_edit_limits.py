"""Length limits on a local model's edit reply (``src/engine/edit_limits.py``)."""

from typing import Any, Dict, List

from src.engine import edit_limits as el
from src.engine import profile_edits as pe
from src.engine.edit_limits import LimitedEditList, without_cut_off
from src.engine.profile_edits import Disagreement, ProfileEdit, ProfileEditList

TEXT = "Born in Leeds.^[a1] Held since 2002.^[a2]"


def _edit(anchor: str, text: str) -> ProfileEdit:
    return ProfileEdit(op="add", anchor=anchor, text=text)


def _cut(text: str, chars: int) -> str:
    """``text`` the way the server ends it at a limit: mid-word, exact length."""
    return (text * (chars // len(text) + 1))[:chars]


def _limits(schema: Dict[str, Any]) -> List[int]:
    return [int(n) for n in _values(schema, ("maxLength", "maxItems"))]


def _values(node: Any, keys: tuple) -> List[Any]:
    if isinstance(node, list):
        return [v for item in node for v in _values(item, keys)]
    if not isinstance(node, dict):
        return []
    here = [node[k] for k in keys if k in node]
    return here + [v for item in node.values() for v in _values(item, keys)]


def test_the_local_schema_has_the_limits_and_the_plain_one_does_not():
    schema = LimitedEditList.model_json_schema()
    defs, props = schema["$defs"], schema["properties"]
    assert defs["ProfileEdit"]["properties"]["text"]["maxLength"] == el.TEXT_CHARS
    assert defs["ProfileEdit"]["properties"]["anchor"]["maxLength"] == 1500
    assert defs["Disagreement"]["properties"]["article_says"]["maxLength"] == 1200
    assert props["new_facts"]["items"]["maxLength"] == el.NEW_FACT_CHARS
    assert props["edits"]["maxItems"] == el.MAX_EDITS
    assert props["disagreements"]["maxItems"] == el.MAX_DISAGREEMENTS
    # llama.cpp refuses a grammar that repeats anything over 2,000 times
    assert len(_limits(schema)) == 8 and max(_limits(schema)) <= 2000
    assert _limits(ProfileEditList.model_json_schema()) == []
    # only the limits differ, so the model is asked for the same reply
    plain = ProfileEditList.model_json_schema()
    assert _without_limits(schema) == {**plain, "title": schema["title"]}


def _without_limits(node: Any) -> Any:
    if isinstance(node, list):
        return [_without_limits(item) for item in node]
    if not isinstance(node, dict):
        return node
    skip = ("maxLength", "maxItems")
    return {k: _without_limits(v) for k, v in node.items() if k not in skip}


def test_the_limits_are_in_the_request_instructor_sends():
    import instructor
    from instructor.processing.response import handle_response_model

    _, kwargs = handle_response_model(
        LimitedEditList, mode=instructor.Mode.JSON_SCHEMA, messages=[]
    )
    sent = kwargs["response_format"]["json_schema"]["schema"]
    assert sent["$defs"]["ProfileEdit"]["properties"]["text"]["maxLength"] == 2000


def test_a_long_reply_from_a_server_without_limits_is_read_whole():
    long = "A long sentence. " * 200 + "^[a9]"
    reply = LimitedEditList.model_validate({"edits": [_edit("END", long).model_dump()]})
    edits, cut = without_cut_off(reply)
    assert edits[0].text == long and cut == []


def test_a_cut_anchor_drops_its_edit_and_the_others_stay():
    loop = _cut("Held since 2002.^[a2-a999-9999", el.ANCHOR_CHARS)
    reply = ProfileEditList(
        edits=[_edit(loop, "Lost.^[a9]"), _edit("END", "Kept.^[a9]")]
    )
    edits, cut = without_cut_off(reply)
    assert [e.text for e in edits] == ["Kept.^[a9]"]
    assert cut == [{"field": "edits[0].anchor", "dropped": True}]


def test_a_cut_text_keeps_its_whole_cited_sentences():
    start = "He was moved to Camp 5 in 2008.^[a9] A judge ruled in 2010.^[a9]. "
    text = (start + "Wait, the profile says he was moved earlier, " * 60)[
        : el.TEXT_CHARS
    ]
    edits, cut = without_cut_off(ProfileEditList(edits=[_edit("END", text)]))
    assert edits[0].text == start.strip()
    assert cut == [{"field": "edits[0].text", "kept_chars": len(start.strip())}]


def test_a_cut_text_that_loops_or_has_no_whole_sentence_is_dropped():
    loop = _cut("He was moved to Camp 5 in the year 2008.^[a9] ", el.TEXT_CHARS)
    prose = _cut("Wait, the profile says something else here. ", el.TEXT_CHARS)
    reply = ProfileEditList(edits=[_edit("END", loop), _edit("END", prose)])
    edits, cut = without_cut_off(reply)
    assert edits == []
    assert [c["field"] for c in cut] == ["edits[0].text", "edits[1].text"]
    assert all(c["dropped"] for c in cut)


def test_cut_notes_are_only_counted():
    reply = ProfileEditList(
        disagreements=[
            Disagreement(
                profile_sentence=_cut("x-a999", el.PROFILE_SENTENCE_CHARS),
                article_says=_cut("Wait, ", el.ARTICLE_SAYS_CHARS),
            )
        ],
        new_facts=["moved in 2008", _cut("and then ", el.NEW_FACT_CHARS)],
        edits=[_edit("END", "Moved in 2008.^[a9]")],
    )
    edits, cut = without_cut_off(reply)
    assert [e.text for e in edits] == ["Moved in 2008.^[a9]"]
    assert [c["field"] for c in cut] == [
        "disagreements[0].profile_sentence",
        "disagreements[0].article_says",
        "new_facts[1]",
    ]


def _run(monkeypatch, reply: ProfileEditList, model: str) -> Dict[str, Any]:
    seen: Dict[str, Any] = {}

    def fake_generate(messages, response_model, model, max_tokens):
        seen["schema"] = response_model
        return reply

    monkeypatch.setattr(pe, "local_generation", fake_generate)
    monkeypatch.setattr(pe, "cloud_generation", fake_generate)
    out = pe.edit_profile(
        "person", "N", {"text": TEXT, "sources": ["a1", "a2"]}, "x", "a9",
        model=model, guard=False,
    )  # fmt: skip
    return {**out, "schema": seen["schema"]}


def test_edit_profile_limits_a_local_reply_and_applies_what_is_whole(monkeypatch):
    loop = _cut("Held since 2002.^[a2-a999-9999", el.ANCHOR_CHARS)
    reply = ProfileEditList(
        edits=[_edit(loop, "Lost.^[a9]"), _edit("END", "Freed in 2016.^[a9]")]
    )
    out = _run(monkeypatch, reply, "ollama/q")
    assert out["schema"] is LimitedEditList
    assert "Freed in 2016.^[a9]" in out["text"] and "Lost" not in out["text"]
    assert out["edit_report"]["cut_off"] == [
        {"field": "edits[0].anchor", "dropped": True}
    ]
    assert [e["text"] for e in out["edits"]] == ["Freed in 2016.^[a9]"]
    assert out["sources"] == ["a1", "a2", "a9"]


def test_cloud_models_and_the_off_switch_get_the_plain_schema(monkeypatch):
    reply = ProfileEditList(edits=[_edit("END", "Freed in 2016.^[a9]")])
    out = _run(monkeypatch, reply, "openai/x")
    assert out["schema"] is ProfileEditList and "cut_off" not in out["edit_report"]
    monkeypatch.setenv(el.EDIT_LIMITS_ENV, "off")
    # with the limits off, a text of exactly the limit's length is not a cut one
    exact = _cut("Wait, the profile says something else here. ", el.TEXT_CHARS)
    out = _run(monkeypatch, ProfileEditList(edits=[_edit("END", exact)]), "ollama/q")
    assert out["schema"] is ProfileEditList and exact.strip() in out["text"]
