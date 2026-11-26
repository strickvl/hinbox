"""The citation check as a Kitaru step: recording, replay, Tinker judge,
evaluator and the investigation loop, without a model or a Kitaru server.
All names and articles are made up."""

from types import SimpleNamespace

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

from scripts import kitaru_step as ks
from src.citecheck import kitaru_loop as kl
from src.citecheck.kitaru_eval import citation_results, cost_usd
from src.citecheck.tinker_judge import is_tinker_model, parse_reply, reply_text
from src.utils.grounding import ClaimVerdict, SupportLevel

ARTICLE = "Mara Voss, a lawyer from Leeds, filed the appeal in March 2019."
CLAIM = "Mara Voss filed the appeal in 2019.^[a1]"


@pytest.fixture(autouse=True)
def sdk_provider(monkeypatch):
    if not isinstance(trace.get_tracer_provider(), TracerProvider):
        trace.set_tracer_provider(TracerProvider())
    monkeypatch.setenv("HINBOX_TRACING", "on")


def _item(**extra):
    return {
        "item_id": "i1",
        "entity_table": "people",
        "entity_name": "Mara Voss",
        "article_id": "a1",
        "claim": CLAIM,
        "split": "test",
        "stratum": "random",
        **extra,
    }


def _reply(*levels):
    return [
        ClaimVerdict(
            claim_number=i + 1, reasoning="The article says so.", support_level=lv
        )
        for i, lv in enumerate(levels)
    ]


# --- recording and replaying the step ------------------------------------------


def test_a_citation_check_is_recorded_as_a_replayable_session(monkeypatch):
    monkeypatch.setattr(
        "src.utils.llm.cloud_generation", lambda **kw: _reply(SupportLevel.SUPPORTED)
    )
    inputs = kl.step_inputs(_item(), ARTICLE, "anthropic/claude-haiku-5-5", gold="yes")
    session, error = ks.run_step(inputs, replay=False)
    assert error is None and session["status"] == "completed"
    assert ks.step_name(inputs) == "grounding"
    # what a replay is given again: the judge's arguments plus the labels
    assert session["inputs"]["claims"] == [["Mara Voss", CLAIM]]
    assert session["inputs"]["model"] == "anthropic/claude-haiku-5-5"
    assert session["inputs"]["labels"]["gold"] == ["yes"]
    assert session["metadata"]["item_ids"] == ["i1"]
    (verdict,) = session["outputs"]["verdicts"]
    assert verdict["support_level"] == "supported"
    assert session["name"] == "grounding: a1"
    rows = {
        r["name"]: r.get("score")
        for r in citation_results(session["inputs"], session["outputs"])
    }
    assert rows["agrees"] is True and rows["good_doubted"] is False


def test_a_replay_asks_the_override_model_through_the_same_function(monkeypatch):
    seen = {}

    def fake(**kw):
        seen.update(kw)
        return [(SupportLevel.PARTIAL, "Only the year.", None)]

    monkeypatch.setattr("src.utils.grounding.judge_claims", fake)
    inputs = kl.step_inputs(_item(), ARTICLE, "ollama/qwen3.6")
    fn, args = ks.step_call(
        ks.apply_override(
            inputs,
            SimpleNamespace(
                model="tinker://run-1/sampler_weights/final",
                system_prompt=None,
                prompt=None,
            ),
        )
    )
    fn(**args)
    assert seen["model"] == "tinker://run-1/sampler_weights/final"
    assert seen["model_type"] == "cloud" and seen["claims"] == [("Mara Voss", CLAIM)]
    assert "labels" not in seen


def test_tinker_names_survive_kitarus_provider_spelling():
    url = "tinker://run-1/sampler_weights/final"
    assert ks.hinbox_model(url) == url
    assert ks.hinbox_model("tinker:run-1/sampler_weights/final") == url
    assert ks.hinbox_model("tinker:thinkingmachines/Inkling-2-Mini") == (
        "tinker/thinkingmachines/Inkling-2-Mini"
    )
    assert ks.hinbox_model("ollama/qwen3.6:27b") == "ollama/qwen3.6:27b"
    assert is_tinker_model(url) and is_tinker_model("tinker/x/y")
    assert not is_tinker_model("ollama/tinkerbell")


def test_a_tinker_judge_gets_the_pipelines_messages(monkeypatch):
    from src.utils import grounding

    seen = {}

    def fake(messages, model, max_tokens):
        seen.update(messages=messages, model=model)
        return _reply(SupportLevel.NOT_SUPPORTED)

    monkeypatch.setattr("src.citecheck.tinker_judge.tinker_verdicts", fake)
    out = grounding.judge_claims(
        article_id="a1",
        article_text=ARTICLE,
        claims=[("Mara Voss", CLAIM)],
        model_type="cloud",
        model="tinker://run-1/sampler_weights/final",
    )
    assert out[0][0] == SupportLevel.NOT_SUPPORTED
    assert seen["messages"] == grounding.grounding_messages(
        "a1", ARTICLE, [("Mara Voss", CLAIM)], False
    )


def test_a_tinker_reply_is_read_like_a_structured_one():
    text = (
        '```json\n{"items": [{"claim_number": 1, "reasoning": "r", '
        '"support_level": "partial"}, {"claim_number": 2, "support_level": "wrong"}]}\n```'
    )
    (v,) = parse_reply(text)
    assert v.claim_number == 1 and v.support_level == SupportLevel.PARTIAL
    assert parse_reply("no json here") == []
    parts = {
        "content": [{"type": "thinking", "text": "hm"}, {"type": "text", "text": "{}"}]
    }
    assert reply_text(parts) == "{}"


def test_the_tinker_call_is_a_model_call_node(monkeypatch):
    """A Tinker sample leaves an LLM span that kitaru_traces turns into an
    llm_call node with its tokens."""
    from src.citecheck import tinker_judge as tj

    class Prompt:
        length = 120

    class Renderer:
        def build_generation_prompt(self, messages, effort):
            return Prompt()

        def get_stop_sequences(self):
            return [0]

        def parse_response(self, tokens):
            return {
                "content": '{"items": [{"claim_number": 1, "reasoning": "r", "support_level": "supported"}]}'
            }, True

    class Sampler:
        def sample(self, prompt, num_samples, sampling_params):
            return SimpleNamespace(sequences=[SimpleNamespace(tokens=[1] * 30)])

    fake_tinker = SimpleNamespace(SamplingParams=lambda **kw: kw)
    monkeypatch.setitem(__import__("sys").modules, "tinker", fake_tinker)
    monkeypatch.setattr(tj, "_sampler", lambda model: (Sampler(), Renderer()))
    inputs = kl.step_inputs(_item(), ARTICLE, "tinker://run-1/sampler_weights/final")
    session, error = ks.run_step(inputs, replay=True)
    assert error is None
    llm = [n for n in session["nodes"] if n["node_type"] == "llm_call"]
    assert len(llm) == 1
    assert llm[0]["tokens"] == {"input_tokens": 120, "output_tokens": 30}
    assert llm[0]["model"] == "tinker://run-1/sampler_weights/final"
    assert session["outputs"]["verdicts"][0]["support_level"] == "supported"


# --- the evaluator -----------------------------------------------------------


def _rows(gold, level, **kw):
    inputs = {
        "claims": [["Mara Voss", CLAIM]],
        "model": "tinker://x/sampler_weights/y",
        "labels": {"item_ids": ["i1"], "gold": [gold]},
    }
    outputs = {"verdicts": [{"support_level": level}] if level else [None]}
    return {r["name"]: r for r in citation_results(inputs, outputs, **kw)}


def test_a_wrong_claim_passed_is_the_error_counted():
    rows = _rows("no", "supported")
    assert rows["wrong_passed"]["score"] is True and rows["agrees"]["score"] is False
    assert "good_doubted" not in rows
    rows = _rows("no", "partial")
    assert rows["wrong_passed"]["score"] is False and rows["agrees"]["score"] is True


def test_a_failed_reply_never_counts_as_a_pass():
    rows = _rows("no", None)
    assert rows["answered"]["score"] is False
    assert rows["wrong_passed"]["score"] is False and rows["agrees"]["score"] is False
    rows = _rows("yes", None)
    assert rows["good_doubted"]["score"] is True


def test_unlabelled_sessions_get_only_the_answer_and_speed_rows():
    rows = _rows(None, "supported", seconds=2.5, input_tokens=1000, output_tokens=50)
    assert set(rows) == {
        "answered",
        "verdict",
        "seconds",
        "input_tokens",
        "output_tokens",
    }


def test_a_replay_without_labels_reads_the_baselines_metadata():
    inputs = {"claims": [["Mara Voss", CLAIM]], "labels": {"item_ids": ["i1"]}}
    rows = citation_results(
        inputs,
        {"verdicts": [{"support_level": "supported"}]},
        baseline_metadata={"item_ids": ["i1"], "gold": ["no"]},
    )
    assert {r["name"]: r.get("score") for r in rows}["wrong_passed"] is True


def test_cost_comes_from_prices_then_from_kitaru():
    prices = {"tinker://x/sampler_weights/y": [0.15, 0.5]}
    assert cost_usd(
        "tinker://x/sampler_weights/y", 2_000_000, 100_000, prices
    ) == pytest.approx(0.35)
    rows = _rows(
        "yes", "supported", input_tokens=2_000_000, output_tokens=100_000, prices=prices
    )
    assert rows["cost_usd"]["score"] == pytest.approx(0.35)
    rows = _rows("yes", "supported", input_tokens=1, output_tokens=1, kitaru_cost=0.02)
    assert rows["cost_usd"]["score"] == 0.02


def test_the_evaluator_script_wraps_every_row(monkeypatch):
    pytest.importorskip("kitaru.task.evaluator")
    from datetime import datetime, timedelta, timezone

    from scripts.kitaru_evaluators.citation_check_checks import evaluate

    start = datetime(2026, 11, 26, tzinfo=timezone.utc)
    s = SimpleNamespace(
        inputs={
            "claims": [["Mara Voss", CLAIM]],
            "model": "m",
            "labels": {"gold": ["no"]},
        },
        outputs={"verdicts": [{"support_level": "supported"}]},
        started_at=start,
        ended_at=start + timedelta(seconds=3),
        tokens=SimpleNamespace(input_tokens=900, output_tokens=40),
        cost=None,
    )
    results = evaluate(SimpleNamespace(session=s, baseline=None), prices={"m": [1, 2]})
    by = {r.name: r for r in results}
    assert by["wrong_passed"].score is True and by["seconds"].score == 3.0
    assert by["cost_usd"].score == pytest.approx((900 + 80) / 1e6)


# --- sessions, investigation, answers --------------------------------------------


def test_sessions_are_built_one_claim_each_and_resume(tmp_path):
    out = str(tmp_path / "sessions.jsonl")
    calls = []

    def run(inputs, replay):
        calls.append(inputs)
        return {
            "metadata": {"item_ids": inputs["labels"]["item_ids"]},
            "inputs": inputs,
        }, None

    items = [
        _item(),
        _item(item_id="i2", article_id="a2"),
        _item(item_id="i3", article_id="gone"),
    ]
    texts = {"a1": ARTICLE, "a2": ARTICLE}
    counts = kl.build_sessions(items, texts, {"i1": {"label": "no"}}, "m", out, run=run)
    assert counts == {"written": 2, "skipped": 0, "failed": 0, "no_text": 1}
    assert calls[0]["labels"]["gold"] == ["no"] and calls[1]["labels"]["gold"] == [None]
    again = kl.build_sessions(items, texts, {}, "m", out, run=run)
    assert again["skipped"] == 2 and len(calls) == 2


def test_the_investigation_asks_one_question_per_session_with_the_claim_marked():
    pytest.importorskip("kitaru.api_models.v1.investigation")
    import uuid

    sid = uuid.uuid4()
    inputs = kl.step_inputs(_item(), ARTICLE, "m")
    req = kl.investigation_request(uuid.uuid4(), "Test set", [(sid, inputs)])
    (row,) = req.sessions
    (q,) = row.questions
    assert row.session_id == sid and q.key == kl.QUESTION_KEY
    assert CLAIM in q.question and "Mara Voss" in q.question
    assert q.highlights[0].selector.path == "/inputs/claims/0/1"


@pytest.mark.parametrize(
    "value,label",
    [
        ("Yes", "yes"),
        ("n", "no"),
        (True, "yes"),
        (False, "no"),
        ({"value": "unsure"}, "unsure"),
        ("maybe", None),
        (3, None),
    ],
)
def test_answers_become_labels(value, label):
    assert kl.answer_label(value) == label


def test_gold_goes_into_the_session_labels():
    inputs = kl.step_inputs(_item(), ARTICLE, "m")
    new = kl.with_gold(inputs, ["no"])
    assert new["labels"]["gold"] == ["no"] and new["labels"]["item_ids"] == ["i1"]
    assert inputs["labels"]["gold"] == [None]  # not changed in place


def test_the_experiment_table_sets_baseline_beside_replay():
    stats = lambda mean, n: SimpleNamespace(mean=mean, count=n)  # noqa: E731
    agg = [
        SimpleNamespace(
            name="wrong_passed", baseline=stats(0.45, 40), result=stats(0.1, 40)
        ),
        SimpleNamespace(name="agrees", baseline=stats(0.7, 80), result=stats(0.9, 80)),
    ]
    rows = kl.aggregate_table(agg)
    assert [r["name"] for r in rows] == ["agrees", "wrong_passed"]
    text = kl.describe_table(rows, "tinker://x")
    assert "0.450" in text and "0.100" in text


def test_a_replays_model_params_reach_the_tinker_sample():
    """A Kitaru replay's model_params set the Tinker call's temperature,
    length, seed and effort; a value Tinker can't honour fails the call."""
    from src.citecheck.tinker_judge import sampling_settings
    from src.utils.llm import set_model_params

    try:
        assert sampling_settings(4096) == (
            0.0,
            {"max_tokens": 4096, "temperature": 0.0},
        )
        set_model_params(
            {"temperature": 0.7, "max_tokens": 512, "seed": 3, "reasoning_effort": 0.5}
        )
        assert sampling_settings(4096) == (
            0.5,
            {"max_tokens": 512, "temperature": 0.7, "seed": 3},
        )
        set_model_params({"reasoning_effort": "high"})
        with pytest.raises(ValueError):
            sampling_settings(4096)
        set_model_params({"reasoning_effort": 1.5})
        with pytest.raises(ValueError):
            sampling_settings(4096)
    finally:
        set_model_params(None)
