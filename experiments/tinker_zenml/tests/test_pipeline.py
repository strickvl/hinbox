"""The ZenML pipeline end to end on a local ZenML store, with Tinker and
Kitaru replaced by fakes: data counts, training, sampling, scoring and the
Kitaru comparison are wired together, and only paths, counts and scores
reach ZenML. Run with ``just tinker-pipeline-test``. Names are made up."""

import json
import os
from types import SimpleNamespace

import pytest


@pytest.fixture()
def zen(tmp_path, monkeypatch):
    monkeypatch.setenv("ZENML_CONFIG_PATH", str(tmp_path / "zen"))
    monkeypatch.setenv("ZENML_ANALYTICS_OPT_IN", "false")
    monkeypatch.setenv("ZENML_LOGGING_VERBOSITY", "WARN")
    monkeypatch.chdir(tmp_path)
    from experiments.tinker_zenml import pipeline as p

    return p


def _write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _data(d):
    items = [
        {
            "item_id": f"i{k}",
            "entity_name": "Mara Voss",
            "article_id": "a1",
            "claim": f"Claim {k}.^[a1]",
            "stratum": "random",
        }
        for k in range(4)
    ]
    _write(d / "test_items.jsonl", items)
    _write(
        d / "test_labels.jsonl",
        [
            {"item_id": "i0", "label": "no"},
            {"item_id": "i1", "label": "no"},
            {"item_id": "i2", "label": "yes"},
            {"item_id": "i3", "label": "yes"},
        ],
    )
    _write(
        d / "test_prompts.jsonl",
        [{"item_ids": [f"i{k}"], "messages": []} for k in range(4)],
    )
    _write(d / "train.jsonl", [{"messages": []}] * 3)
    _write(d / "tune.jsonl", [{"messages": []}])


def _reply(level):
    return json.dumps(
        {"items": [{"claim_number": 1, "reasoning": "r", "support_level": level}]}
    )


def test_the_pipeline_trains_samples_scores_and_compares(zen, tmp_path, monkeypatch):
    p = zen
    d = tmp_path / "data"
    d.mkdir()
    _data(d)
    seen = {}

    def train(data_dir, cfg):
        seen["cfg"] = cfg
        return {
            "checkpoint": {
                "name": cfg["name"],
                "sampler": "tinker://run-1/sampler_weights/citecheck",
            },
            "log": [
                {"step": 0, "train_nll": 1.2, "tune_nll": 1.3},
                {"step": 1, "train_nll": 0.4, "tune_nll": 0.5},
            ],
        }

    def sample(checkpoint, prompts, out, cfg):
        levels = ["partial", "supported", "supported", "not_supported"]
        rows = [
            {
                "item_ids": [f"i{k}"],
                "reply": _reply(lv),
                "output_tokens": 40,
                "seconds": 0.5,
            }
            for k, lv in enumerate(levels)
        ]
        _write(out, rows)
        return rows

    def compare(name, model, prices, item_ids):
        seen["kitaru"] = (name, model, prices)
        seen["item_ids"] = item_ids
        return {
            "experiment_id": "e1",
            "run_id": "r1",
            "status": "completed",
            "table": [
                {
                    "name": "wrong_passed",
                    "baseline_mean": 0.45,
                    "result_mean": 0.1,
                    "baseline_count": 2,
                    "result_count": 2,
                },
                {
                    "name": "seconds",
                    "baseline_mean": 3.1,
                    "result_mean": 1.4,
                    "baseline_count": 4,
                    "result_count": 4,
                },
            ],
        }

    monkeypatch.setattr(p, "tinker_train", train)
    monkeypatch.setattr(p, "tinker_sample", sample)
    monkeypatch.setattr(p, "kitaru_compare", compare)
    baselines = tmp_path / "baselines.json"
    baselines.write_text(
        json.dumps({"Untuned": {"wrong_passed": 2, "good_doubted": [0, 2]}})
    )
    p.main(
        [
            "--data-dir",
            str(d),
            "--articles",
            "unused.parquet",
            "--steps",
            "2",
            "--kitaru",
            "--baselines",
            str(baselines),
        ]
    )

    assert seen["cfg"]["steps"] == 2 and seen["cfg"]["effort"] == 0.0
    assert seen["kitaru"][1] == "tinker://run-1/sampler_weights/citecheck"
    assert seen["item_ids"] == ["i0", "i1", "i2", "i3"]
    from zenml.client import Client

    run = Client().get_pipeline("citecheck_finetune").last_run
    assert str(run.status) in ("completed", "ExecutionStatus.COMPLETED")
    report = run.steps["score_test_set"].outputs["test_report"][0].load()
    # i1 labelled "no" was called supported: 1 of 2 wrong claims passed;
    # i3 labelled "yes" was doubted: 1 of 2 good claims doubted
    assert report["wrong_passed"]["count"] == 1 and report["wrong_passed"]["of"] == 2
    assert report["good_doubted"]["count"] == 1
    card = run.steps["report_card"].outputs["training_report"][0].load()
    assert "Wrong claims passed" in card and "50.0%" in card and "Untuned" in card
    # numbers only: no claim text reaches the ZenML artifact
    assert "Claim 1" not in card and "Mara Voss" not in card
    curve = run.steps["train_on_tinker"].outputs["loss_log"][0].load()
    assert curve[-1] == {"step": 1, "train_nll": 0.4, "tune_nll": 0.5}
    train_meta = run.steps["train_on_tinker"].run_metadata
    assert train_meta["best_tune_nll"] == 0.5
    kitaru_card = run.steps["compare_in_kitaru"].outputs["kitaru_report"][0].load()
    assert "45.0%" in kitaru_card and "10.0%" in kitaru_card and "1.4 s" in kitaru_card
    assert kitaru_card.count(">better<") == 2
    kitaru_meta = run.steps["compare_in_kitaru"].run_metadata
    assert kitaru_meta["wrong_passed_result"] == 0.1
    assert os.path.exists(
        p.replies_path(str(d), "tinker://run-1/sampler_weights/citecheck")
    )


def test_two_checkpoints_with_one_name_never_share_a_replies_file(zen):
    a = zen.replies_path("d", "tinker://run-1/sampler_weights/citecheck")
    b = zen.replies_path("d", "tinker://run-2/sampler_weights/citecheck")
    assert a != b and ".citecheck." in a


def test_sampling_starts_when_no_replies_file_exists(zen, tmp_path, monkeypatch):
    from scripts import tinker_citecheck as tc

    prompts = tmp_path / "prompts.jsonl"
    _write(prompts, [{"item_ids": ["i0"], "messages": []}])
    out = tmp_path / "new" / "replies.jsonl"
    out.parent.mkdir()

    class Service:
        def create_sampling_client(self, **kw):
            return "sampler"

    def sample_rows(sampler, renderer, todo, args, out_path):
        _write(out_path, [{"item_ids": p["item_ids"], "reply": "{}"} for p in todo])

    import tinker

    monkeypatch.setattr(tinker, "ServiceClient", Service)
    monkeypatch.setattr(tc, "_renderer", lambda model: ("renderer", None))
    monkeypatch.setattr(tc, "_sample_rows", sample_rows)
    rows = zen.tinker_sample("tinker://x", str(prompts), str(out), {"model": "m"})
    assert [r["item_ids"] for r in rows] == [["i0"]]


def test_a_rerun_reports_only_its_own_training_log(zen, tmp_path, monkeypatch):
    """A second training run in the same folder must not count the first
    run's steps or claim its lower tune loss."""
    from scripts import tinker_citecheck as tc

    d = tmp_path / "data"
    (d / "tinker").mkdir(parents=True)
    _write(d / "train.jsonl", [{"messages": []}])
    _write(d / "tinker" / "train_log.jsonl", [{"step": 0, "tune_nll": 0.01}])

    class Saved:
        path = "tinker://run-2/sampler_weights/citecheck"

    class Client:
        def save_weights_for_sampler(self, name):
            return SimpleNamespace(result=lambda: Saved())

        def save_state(self, name):
            return SimpleNamespace(result=lambda: Saved())

    def train_loop(args, train, tune, steps):
        tc._append(
            os.path.join(args.out, "train_log.jsonl"), {"step": 0, "tune_nll": 0.9}
        )
        return Client()

    monkeypatch.setattr(tc, "train_loop", train_loop)
    out = zen.tinker_train(str(d), {"name": "citecheck", "model": "m", "steps": 1})
    assert out["log"] == [{"step": 0, "tune_nll": 0.9}]
