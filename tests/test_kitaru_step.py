"""Replaying one recorded step (scripts/kitaru_step.py) and its checks
(src/evals/kitaru_checks.py), without a model or a Kitaru server."""

import json
from types import SimpleNamespace

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

from scripts import kitaru_step as ks
from src.evals.kitaru_checks import (
    case_result,
    extraction_results,
    grounding_results,
    match_results,
    step_checks,
    step_results,
)

OLD = "## Background\nThe ACLU sued over the camp.^[a1] It filed again in 2009.^[a1]"


@pytest.fixture(autouse=True)
def sdk_provider(monkeypatch):
    # The global provider can be set only once per process: make sure it is
    # an SDK one, so the script's configure_tracing() never replaces it
    if not isinstance(trace.get_tracer_provider(), TracerProvider):
        trace.set_tracer_provider(TracerProvider())
    monkeypatch.setenv("HINBOX_TRACING", "on")


def _inputs(**extra):
    return {
        "call": "edit_profile",
        "entity_type": "organizations",
        "entity_name": "ACLU",
        "existing_profile": {"text": OLD, "sources": ["a1"]},
        "article_text": "The ACLU appealed in 2010.",
        "article_id": "a2",
        "model": "openai/gpt-6-luna",
        "domain": "guantanamo",
        "article_dates": {},
        "kind": "advocacy",
        **extra,
    }


def _override(**fields):
    base = {"model": None, "system_prompt": None, "prompt": None, "model_params": None}
    return SimpleNamespace(**{**base, **fields})


# --- the replay override ------------------------------------------------------


def test_kitaru_model_names_become_hinbox_ones():
    assert ks.hinbox_model("openai:gpt-6-luna") == "openai/gpt-6-luna"
    assert ks.hinbox_model("ollama:qwen2.5:32b") == "ollama/qwen2.5:32b"
    assert ks.hinbox_model("ollama/qwen2.5:32b") == "ollama/qwen2.5:32b"
    assert (
        ks.hinbox_model("openrouter:anthropic/claude-3.5-sonnet")
        == "openrouter/anthropic/claude-3.5-sonnet"
    )


def test_a_model_override_replaces_the_steps_model():
    inputs = _inputs()
    assert ks.apply_override(inputs, None) is inputs
    swap = _override(model="anthropic:claude-haiku-5-5")
    assert ks.apply_override(inputs, swap)["model"] == "anthropic/claude-haiku-5-5"
    for key in ("openai:gpt-6-luna", "openai/gpt-6-luna", "gpt-6-luna"):
        mapped = _override(model={key: "anthropic:claude-haiku-5-5"})
        assert (
            ks.apply_override(inputs, mapped)["model"] == "anthropic/claude-haiku-5-5"
        )
    with pytest.raises(ValueError, match="no entry"):
        ks.apply_override(inputs, _override(model={"openai:other": "x:y"}))


def test_prompt_overrides_are_refused():
    with pytest.raises(ValueError, match="system_prompt"):
        ks.apply_override(_inputs(), _override(system_prompt="Be brief."))


def test_model_params_go_on_every_call_and_unknown_ones_fail(monkeypatch):
    from src.utils import llm

    monkeypatch.setattr(llm, "_PARAM_OVERRIDE", {})
    override = _override(model_params={"temperature": 0.5, "frequency_penalty": 0.5})
    assert ks.apply_override(_inputs(), override) == _inputs()
    ks.apply_model_params(override)
    assert llm._PARAM_OVERRIDE == {"temperature": 0.5, "frequency_penalty": 0.5}
    with pytest.raises(ValueError, match="stop"):
        ks.apply_model_params(_override(model_params={"stop": ["\\n"]}))


# --- which function a step runs -----------------------------------------------


def test_an_eval_step_runs_edit_profile_with_its_own_arguments(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "src.evals.profile_update.edit_step", lambda args: seen.update(args)
    )
    fn, args = ks.step_call(_inputs())
    fn(**args)
    assert "call" not in seen and seen["article_id"] == "a2"


def test_a_pipeline_step_runs_update_profile_with_the_model_that_answered():
    recorded = {
        "entity_type": "people",
        "entity_name": "N",
        "existing_profile": {"text": OLD},
        "new_article_text": "x",
        "new_article_id": "a2",
        "model": None,
        "answering_model": "openai/gpt-6-luna",
        "approach": "edit",
    }
    fn, args = ks.step_call(recorded)
    assert fn.__name__ == "update_profile"
    assert args["model"] == "openai/gpt-6-luna"
    assert "answering_model" not in args and "versioned_profile" in args


def test_a_step_that_guarded_corrections_is_refused():
    with pytest.raises(ValueError, match="2 of the historian's sentences"):
        ks.step_call({**_inputs(call=None), "guarded_corrections": 2})
    fn, args = ks.step_call({**_inputs(), "guarded_corrections": 0})
    assert "guarded_corrections" not in args


# --- the re-run becomes a session ---------------------------------------------


def test_a_replayed_step_is_recorded_as_a_session_of_its_own(monkeypatch):
    new_text = OLD + " It appealed in 2010.^[a2]"

    def fake_edit_profile(**kwargs):
        return {
            **kwargs["existing_profile"],
            "text": new_text,
            "sources": ["a1", "a2"],
            "edits": [{"action": "add", "text": "It appealed in 2010.^[a2]"}],
            "disagreements": [],
            "edit_report": {"applied": 1},
        }

    monkeypatch.setattr("src.engine.profile_edits.edit_profile", fake_edit_profile)
    inputs = _inputs()
    session, error = ks.run_step(inputs, replay=True)
    assert error is None
    assert session["inputs"] == inputs  # what a later replay is given again
    assert session["outputs"]["profile"]["text"] == new_text
    assert session["status"] == "completed" and session["framework"] == "hinbox"
    assert session["name"] == "profile.update: ACLU"
    assert "article_session" not in session["metadata"]
    assert [n["name"] for n in session["nodes"]] == ["profile.update"]


def test_a_step_that_raises_is_recorded_failed(monkeypatch):
    def broken(**kwargs):
        raise RuntimeError("model down")

    monkeypatch.setattr("src.engine.profile_edits.edit_profile", broken)
    session, error = ks.run_step(_inputs(), replay=False)
    assert isinstance(error, RuntimeError)
    assert session["status"] == "failed" and "model down" in session["error"]


def test_the_session_becomes_kitarus_import_shape():
    importer = pytest.importorskip("kitaru.task.importer")
    session = {
        "status": "completed",
        "name": "profile.update: ACLU",
        "inputs": _inputs(),
        "outputs": {"profile": {"text": OLD}},
        "error": None,
        "started_at": "2026-11-14T22:00:00Z",
        "ended_at": "2026-11-14T22:00:05Z",
        "external_id": "hinbox:t:s",
        "metadata": {},
        "framework": "hinbox",
        "nodes": [
            {"index": 0, "parent_index": None, "external_id": "s", "name": "profile.update",
             "node_type": "span", "status": "completed", "inputs": None, "outputs": None,
             "attributes": {}},
            {"index": 1, "parent_index": 0, "external_id": "c", "name": "chat",
             "node_type": "llm_call", "status": "completed", "inputs": [], "outputs": [],
             "attributes": {}},
        ],
    }  # fmt: skip
    nodes = importer.flatten_nodes(ks.ingest_request(session, replay=True).nodes)
    assert [(n.external_id, n.parent_external_id) for n in nodes] == [
        ("s", None),
        ("c", "s"),
    ]


# --- the evaluator's checks ---------------------------------------------------


def test_checks_read_either_steps_argument_names():
    new = {"profile": {"text": OLD + " Appealed.^[a2]", "sources": ["a1", "a2"]}}
    pipeline = {
        "existing_profile": {"text": OLD, "sources": ["a1"]},
        "new_article_id": "a2",
        "new_article_text": "Appealed.",
    }
    for inputs in (_inputs(), pipeline):
        scores = step_checks(inputs, new)
        assert scores["passed"] and scores["cites_article"] and scores["updated"]


def test_a_collapsed_profile_with_an_invented_citation_fails():
    new = {"profile": {"text": "The ACLU sued.^[zz9]", "sources": ["a1", "a2"]}}
    rows = {r["name"]: r for r in step_results(_inputs(), new)}
    assert rows["update_ok"]["passed"] is False
    assert "collapsed" in rows["update_ok"]["explanation"]
    assert "zz9" in rows["update_ok"]["explanation"]
    assert rows["invented_citations"]["score"] == 1.0
    assert rows["length_ratio"]["passed"] is False


def test_a_step_that_recorded_no_profile_counts_as_not_updated():
    rows = {r["name"]: r for r in step_results(_inputs(), {"error": "failed"})}
    assert rows["updated"]["score"] is False
    assert rows["update_ok"]["passed"] is True  # the old profile is intact


def test_case_ok_applies_the_eval_rule_for_the_cases_kind():
    added = OLD + (
        " Lawyers filed motions in Miami.^[a2] Judges heard appeals in Virginia.^[a2]"
        " Reporters covered hearings at Guantanamo.^[a2]"
    )
    new = {"profile": {"text": added, "sources": ["a1", "a2"]}}
    rows = {
        r["name"]: r for r in step_results(_inputs(), new, {"eval_kind": "irrelevant"})
    }
    assert rows["case_ok"]["passed"] is False  # 3 sentences on an irrelevant article
    assert "3 sentences added" in rows["case_ok"]["explanation"]
    kept = {"profile": {"text": OLD, "sources": ["a1", "a2"]}}
    irrelevant = {"eval_kind": "irrelevant"}
    assert case_result(step_checks(_inputs(), kept, irrelevant))["passed"]
    missed = case_result(step_checks(_inputs(), kept, {"eval_kind": "new_fact"}))
    assert missed["passed"] is False and "not cited" in missed["explanation"]
    assert rows["update_ok"]["passed"] is True  # the case rule, not damage


DISPUTED = {
    "eval_kind": "disputed",
    "eval_expected": {
        "keep_both": True,
        "original_value": "2009",
        "disputed_value": "2010",
        "original_sources": ["a1"],
        "profile_sentence": "It filed again in 2009.^[a1]",
        "article_sentence": "The ACLU filed again in 2010.",
    },
}


def test_case_ok_on_a_disputed_case_needs_both_years():
    both = OLD + " A later report gives the year as 2010 instead.^[a2]"
    swapped = OLD.replace("2009.^[a1]", "2010.^[a2]")
    for text, ok in ((both, True), (swapped, False)):
        new = {"profile": {"text": text, "sources": ["a1", "a2"]}}
        assert case_result(step_checks(_inputs(), new, DISPUTED))["passed"] is ok


def test_case_ok_on_a_labelled_case_needs_its_facts():
    label = {
        "disputes_ok": False,
        "facts": [{"id": "appeal", "claim": "The ACLU appealed in 2010.", "core": True,
                   "match": [["appeal"], ["2010"]]}],
        "must_not": [],
    }  # fmt: skip
    labels = {"eval_kind": "new_fact", "eval_claim_label": label}
    stated = {
        "profile": {"text": OLD + " It appealed in 2010.^[a2]", "sources": ["a1", "a2"]}
    }
    vague = {
        "profile": {"text": OLD + " The ACLU is a group.^[a2]", "sources": ["a1", "a2"]}
    }
    assert case_result(step_checks(_inputs(), stated, labels))["passed"]
    missing = case_result(step_checks(_inputs(), vague, labels))
    assert (
        missing["passed"] is False
        and "misses 1 labelled fact" in missing["explanation"]
    )


def test_case_ok_is_left_out_without_the_cases_data():
    new = {"profile": {"text": OLD, "sources": ["a1"]}}
    for labels in (None, {"eval_kind": "disputed"},  # no expected: an older session
                   {"eval_kind": "new_fact", "eval_claim_label": True}):  # fmt: skip
        assert case_result(step_checks(_inputs(), new, labels)) is None


def test_a_replay_keeps_its_baselines_eval_labels(monkeypatch):
    def same(**kwargs):
        return {**kwargs["existing_profile"], "edits": [], "disagreements": [],
                "edit_report": {}}  # fmt: skip

    monkeypatch.setattr("src.engine.profile_edits.edit_profile", same)
    labels = {"eval_kind": "irrelevant", "eval_item_id": "irrelevant:ACLU:1"}
    session, error = ks.run_step(_inputs(labels=labels), replay=True)
    assert error is None
    assert session["inputs"]["labels"] == labels  # for the evaluator
    assert session["metadata"]["eval_kind"] == "irrelevant"
    fn, args = ks.step_call(_inputs(labels=labels))
    assert "labels" not in args  # the step itself never sees them


def test_every_check_is_a_valid_kitaru_result():
    evaluation = pytest.importorskip("kitaru.api_models.v1.evaluation")
    new = {"profile": {"text": OLD, "sources": ["a1", "a2"]}}
    names = [
        evaluation.EvaluationResult(**r).name for r in step_results(_inputs(), new)
    ]
    assert len(names) == len(set(names))
    json.dumps(step_results(_inputs(), new))  # plain JSON for the task result


# --- match_check steps ----------------------------------------------------------


def _match_inputs(**extra):
    return {
        "backend": "cloud",
        "new_name": "Camp 7",
        "existing_name": "Camp Seven",
        "new_profile_text": "Camp 7 held the former CIA prisoners.",
        "existing_profile_text": "Camp Seven is the secret prison.^[a1]",
        "entity_type": "locations",
        "model": "openai/gpt-6-luna",
        **extra,
    }


def test_a_match_check_runs_check_match_asked_where_the_model_runs(monkeypatch):
    from src.engine.match_checker import MatchCheckResult

    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return MatchCheckResult(is_match=True, confidence=0.9, reason="same camp")

    monkeypatch.setattr("src.engine.match_checker._check_match", fake)
    labels = {"eval_label": True, "eval_item_id": "merge:location:Camp 7|Camp Seven"}
    inputs = ks.apply_override(
        _match_inputs(labels=labels), _override(model="ollama:qwen3.6:35b")
    )
    fn, args = ks.step_call(inputs)
    fn(**args)
    assert seen["model"] == "ollama/qwen3.6:35b" and seen["backend"] == "local"
    assert "labels" not in seen


def test_a_replayed_match_check_is_a_session_the_evaluator_scores(monkeypatch):
    def down(*args, **kwargs):
        raise ConnectionError("model down")

    for name in ("text_generation", "cloud_generation"):
        monkeypatch.setattr(f"src.engine.match_checker.{name}", down)
    session, error = ks.run_step(
        _match_inputs(labels={"eval_label": False}), replay=True
    )
    # a failed call is no "different" verdict: the run and its session fail
    assert isinstance(error, RuntimeError)
    assert session["status"] == "failed" and session["error"].startswith("API error")
    assert session["name"] == "match_check: Camp 7 vs Camp Seven"
    assert session["inputs"]["labels"] == {"eval_label": False}
    (row,) = match_results(session["outputs"], session["inputs"]["labels"])
    assert row["name"] == "match_ok" and row["score"] is False


def test_match_results_say_which_way_the_answer_was_wrong():
    out = {"result": {"is_match": True, "confidence": 0.8, "reason": "both camps"}}
    rows = {r["name"]: r["score"] for r in match_results(out, {"eval_label": False})}
    assert rows == {
        "is_match": True,
        "confidence": 0.8,
        "match_ok": False,
        "false_merge": True,
        "missed_merge": False,
    }
    unlabelled = {r["name"] for r in match_results(out)}
    assert unlabelled == {"is_match", "confidence"}
    evaluation = pytest.importorskip("kitaru.api_models.v1.evaluation")
    for r in match_results(out, {"eval_label": True}):
        evaluation.EvaluationResult(**r)


# --- extract.<type> steps ------------------------------------------------------


def _extract_inputs(**extra):
    return {
        "entity_type": "locations",
        "article_content": "Hearings took place at Guantánamo Bay in Cuba.",
        "article_id": "a1",
        "domain": "guantanamo",
        "model_type": "gemini",
        "model": "openai/gpt-6-luna",
        **extra,
    }


def test_an_extraction_step_replays_with_overridden_model_and_scores(monkeypatch):
    seen = {}

    def fake_local(**kwargs):
        seen.update(kwargs)
        return [
            {"name": "Guantánamo Bay", "type": "military_base"},
            {"name": "Cuba", "type": "country"},
        ]

    monkeypatch.setattr("src.engine.extractors.extract_entities_local", fake_local)
    inputs = ks.apply_override(
        _extract_inputs(), _override(model="ollama:local-qwen36-32k")
    )
    assert ks.step_name(inputs) == "extract.locations"
    session, error = ks.run_step(inputs, replay=True)
    assert error is None
    assert seen["model"] == "ollama/local-qwen36-32k"
    assert session["status"] == "completed"
    assert session["name"] == "extract.locations: a1"
    assert session["metadata"]["step"] == "extract.locations"
    rows = {
        r["name"]: r for r in extraction_results(session["inputs"], session["outputs"])
    }
    assert rows["extract_ok"]["passed"] is True
    assert rows["entity_count"]["score"] == 2.0
    assert rows["mention_rate"]["score"] == 1.0
    assert rows["qc_flags"]["passed"] is True
    json.dumps(list(rows.values()))


def test_a_failed_extraction_replay_fails_and_scores_false(monkeypatch):
    monkeypatch.setattr(
        "src.engine.extractors.extract_entities_cloud",
        lambda **kw: (_ for _ in ()).throw(RuntimeError("output cut off")),
    )
    session, error = ks.run_step(_extract_inputs(), replay=True)
    assert isinstance(error, RuntimeError)
    assert session["status"] == "failed" and "output cut off" in session["error"]
    (row,) = extraction_results(session["inputs"], session["outputs"])
    assert row["name"] == "extract_ok" and row["passed"] is False


def test_extraction_results_flag_unmentioned_entities_and_valid_kitaru_schema():
    outputs = {
        "entities": [
            {"name": "Cuba", "type": "country"},
            {"name": "Atlantis", "type": "city"},
        ],
        "counts": {"raw_count": 3, "deduped": 1, "dropped_missing_required": 0},
        "qc_flags": ["many_duplicates"],
        "retry_attempted": True,
    }
    rows = {r["name"]: r for r in extraction_results(_extract_inputs(), outputs)}
    assert rows["mention_rate"]["score"] == 0.5
    assert rows["mention_rate"]["passed"] is False
    assert rows["qc_flags"]["passed"] is False
    assert rows["deduped"]["score"] == 1.0
    evaluation = pytest.importorskip("kitaru.api_models.v1.evaluation")
    for r in rows.values():
        evaluation.EvaluationResult(**r)


# --- grounding steps -----------------------------------------------------------


def _grounding_inputs(**extra):
    return {
        "article_id": "a1",
        "article_text": "Judge Vance Spath retired in 2018 after presiding over the Cole case.",
        "claims": [
            ["Vance Spath", "Vance Spath retired in 2018."],
            ["Vance Spath", "He presided over the 9/11 case."],
        ],
        "model_type": "gemini",
        "model": "anthropic/claude-haiku-5-5",
        "ask_passage": False,
        **extra,
    }


def test_a_grounding_step_replays_with_overridden_model_and_scores(monkeypatch):
    from src.utils.grounding import ClaimVerdict, SupportLevel

    seen = {}

    def fake_local(**kwargs):
        seen.update(kwargs)
        return [
            ClaimVerdict(
                claim_number=1,
                reasoning='Article says "Judge Vance Spath retired in 2018".',
                support_level=SupportLevel.SUPPORTED,
            ),
            ClaimVerdict(
                claim_number=2,
                reasoning='Article says "presiding over the Cole case", not 9/11.',
                support_level=SupportLevel.NOT_SUPPORTED,
            ),
        ]

    monkeypatch.setattr("src.utils.llm.local_generation", fake_local)
    labels = {"expected_levels": ["supported", "not_supported"]}
    inputs = ks.apply_override(
        _grounding_inputs(labels=labels), _override(model="ollama:local-qwen36-32k")
    )
    assert ks.step_name(inputs) == "grounding"
    session, error = ks.run_step(inputs, replay=True)
    assert error is None
    assert seen["model"] == "ollama/local-qwen36-32k"
    assert session["status"] == "completed"
    assert session["name"] == "grounding: a1"
    rows = {
        r["name"]: r
        for r in grounding_results(
            session["inputs"], session["outputs"], session["inputs"]["labels"]
        )
    }
    assert rows["grounding_ok"]["passed"] is True
    assert rows["supported_share"]["score"] == 0.5
    assert rows["accepted_share"]["score"] == 0.5
    assert rows["not_supported"]["score"] == 1.0
    assert rows["passage_rate"]["score"] == 1.0
    assert rows["label_agreement"]["score"] == 1.0
    assert rows["label_agreement"]["passed"] is True
    json.dumps(list(rows.values()))


def test_grounding_results_catch_missing_verdicts_and_errors(monkeypatch):
    partial_out = {
        "verdicts": [
            {
                "claim_number": 1,
                "entity_name": "Vance Spath",
                "claim": "Vance Spath retired in 2018.",
                "support_level": "supported",
                "reasoning": "Matches.",
            },
            {
                "claim_number": 2,
                "entity_name": "Vance Spath",
                "claim": "He presided over the 9/11 case.",
                "support_level": None,
                "reasoning": None,
            },
        ],
        "missing": 1,
    }
    rows = {r["name"]: r for r in grounding_results(_grounding_inputs(), partial_out)}
    assert rows["grounding_ok"]["passed"] is False
    assert rows["missing_verdicts"]["score"] == 1.0
    assert rows["missing_verdicts"]["passed"] is False

    monkeypatch.setattr(
        "src.utils.llm.cloud_generation",
        lambda **kw: (_ for _ in ()).throw(ConnectionError("judge offline")),
    )
    session, error = ks.run_step(_grounding_inputs(), replay=True)
    assert isinstance(error, ConnectionError)
    assert session["status"] == "failed"
    (row,) = grounding_results(session["inputs"], session["outputs"])
    assert row["name"] == "grounding_ok" and row["passed"] is False
    evaluation = pytest.importorskip("kitaru.api_models.v1.evaluation")
    for r in rows.values():
        evaluation.EvaluationResult(**r)
