"""Optimise Hinbox's edit-list profile updater with DSPy GEPA.

The updater (``src/engine/profile_edits.py``) asks a model for a short list of
``add``/``replace`` edits to an existing profile, and code applies them. Here
the same request is a DSPy program, and GEPA rewrites its instructions for a
given (small, local) model. Each candidate is scored by the profile-update
eval's own checks (``score_update``): no collapse, facts kept, at most a
sentence or two from a passing mention, the article cited for a new fact, and
both years kept for a disputed one. GEPA also gets the reasons in words, and
is told off for "the sources disagree" sentences on cases with no planted
dispute and for numbers the cited article never gives.

Data: ``profile_update_cases.jsonl`` from ``just eval-cases``. The optimizer
trains on the ``train`` entities and validates on ``dev``; scores are reported
on ``test`` by default. Unreviewed disputed cases are left out unless
``--include-unreviewed``: some pair two different events, and training on them
would teach the model to invent disagreements.

Instruction-only: the examples are whole profiles (up to ~30k characters), far
too long to use as few-shot demos.

Usage (from the repository root)::

    just dspy-profile --model ollama/local-qwen36-16k:latest \\
        --reflection-model openai/gpt-6-luna
    just dspy-profile --model ollama/local-qwen36-16k:latest --optimizer none
    just dspy-profile --model ollama/local-qwen36-16k:latest --load <saved.json>
"""

import argparse
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

import dspy

from experiments.dspy_opt.lm import make_lm
from src.config_loader import DomainConfig
from src.engine.profile_edits import Disagreement, ProfileEdit, apply_edits
from src.evals.profile_update import (
    load_cases,
    run_updates,
    score_update,
    summarize_updates,
)
from src.evals.runner import save_run
from src.logging_config import console

INPUT_FIELDS = (
    "entity_type",
    "entity_name",
    "article_id",
    "profile_text",
    "article_text",
)
# Unpassed cases still earn partial credit, so GEPA can tell "nearly" from
# "badly"; each invented dispute or unsupported number costs this much
PENALTY = 0.25


class UpdateProfile(dspy.Signature):
    """Replaced by the domain's profile_edit.md (see ``base_instructions``)."""

    entity_type: str = dspy.InputField()
    entity_name: str = dspy.InputField()
    article_id: str = dspy.InputField(desc="cite it as ^[<article_id>]")
    profile_text: str = dspy.InputField(desc="the existing profile")
    article_text: str = dspy.InputField(desc="the new article")
    disagreements: List[Disagreement] = dspy.OutputField()
    new_facts: List[str] = dspy.OutputField()
    edits: List[ProfileEdit] = dspy.OutputField()


def base_instructions(domain: str) -> str:
    """The pipeline's edit prompt, with its placeholders made generic.

    The entity and article come in as input fields, so GEPA's rewrite can be
    copied back into ``profile_edit.md`` by putting the placeholders back.
    """
    template = DomainConfig(domain).load_profile_prompt("edit")
    return (
        template.replace("{entity_name}", "the entity")
        .replace("{entity_type}", "entity")
        .replace("{article_id}", "<article_id>")
    )


class ProfileUpdater(dspy.Module):
    def __init__(self, instructions: str, chain_of_thought: bool = False):
        super().__init__()
        step = dspy.ChainOfThought if chain_of_thought else dspy.Predict
        self.update = step(UpdateProfile.with_instructions(instructions))

    def forward(self, **inputs: Any) -> dspy.Prediction:
        return self.update(**inputs)


def to_example(case: Dict[str, Any]) -> dspy.Example:
    return dspy.Example(
        entity_type=case["entity_type"],
        entity_name=case["entity_name"],
        article_id=case["article_id"],
        profile_text=case["profile"].get("text", ""),
        article_text=case["article_text"],
        case=case,
    ).with_inputs(*INPUT_FIELDS)


def apply_prediction(case: Dict[str, Any], pred: dspy.Prediction) -> str:
    text, _ = apply_edits(case["profile"].get("text", ""), list(pred.edits or []))
    return text


def score_and_feedback(case: Dict[str, Any], new_text: str) -> Dict[str, Any]:
    """The eval's checks as one score plus the reasons, in plain words."""
    s = score_update(case, new_text)
    expected = case.get("expected", {})
    notes: List[str] = []
    if s["collapsed"]:
        notes.append(f"The profile shrank to {s['length_ratio']:.0%} of its length.")
    if s["low_retention"]:
        notes.append(f"{s['facts_lost']} cited facts from the profile were lost.")
    if s["invented_citations"]:
        notes.append(f"Cited ids that don't exist: {s['invented_citations']}.")
    if case["kind"] == "irrelevant" and not s["passed"]:
        notes.append(
            f"The article only mentions {case['entity_name']} in passing, but "
            f"{s['added_sentences']} sentences were added (at most "
            f"{expected.get('max_new_sentences', 2)}). Add only facts about "
            "the entity itself, or nothing."
        )
    if case["kind"] == "new_fact" and not s["cites_article"]:
        notes.append(
            "The article has new facts about the entity, but nothing citing "
            f"^[{case['article_id']}] was added."
        )
    if case["kind"] == "disputed" and not s["keeps_both"]:
        notes.append(
            f"The article gives {expected['disputed_value']} where the profile "
            f"says {expected['original_value']} in: "
            f'"{expected["profile_sentence"]}". Keep that sentence and add one '
            "right after it giving the article's year, citing the article, "
            "and saying the sources disagree."
        )
    invented_disputes = s["dispute_claims"] if case["kind"] != "disputed" else 0
    if invented_disputes:
        notes.append(
            f"{invented_disputes} added sentence(s) say the sources disagree, "
            "but nothing in this article contradicts the profile. Different "
            "events, later steps and extra detail are not disagreements."
        )
    if s["unsupported_numbers"]:
        notes.append(
            f"{s['unsupported_numbers']} added sentence(s) cite the article "
            "but give numbers or years the article doesn't contain."
        )
    penalty = PENALTY * (invented_disputes + s["unsupported_numbers"])
    if s["passed"]:
        score = 1.0
    else:
        # Partial credit: whatever share of the old facts survived, halved
        score = 0.5 * s["fact_retention"] * (not s["collapsed"])
    return {
        "score": max(0.0, score - penalty),
        "feedback": " ".join(notes) or "Passed: minimal, cited and honest.",
        "scores": s,
    }


def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """GEPA metric: a score and written feedback for one case."""
    try:
        result = score_and_feedback(gold.case, apply_prediction(gold.case, pred))
    except Exception as e:  # noqa: BLE001 — a malformed answer scores zero
        result = {"score": 0.0, "feedback": f"The answer couldn't be used: {e}"}
    return dspy.Prediction(score=result["score"], feedback=result["feedback"])


def make_updater(program: dspy.Module):
    def update(case: Dict[str, Any]) -> str:
        example = to_example(case)
        return apply_prediction(case, program(**example.inputs()))

    return update


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True, help="e.g. ollama/local-qwen36-16k")
    parser.add_argument("--optimizer", choices=("none", "gepa"), default="gepa")
    parser.add_argument(
        "--reflection-model", default=None, help="GEPA only, e.g. openai/gpt-6-luna"
    )
    parser.add_argument("--auto", choices=("light", "medium", "heavy"), default="light")
    parser.add_argument("--cot", action="store_true", help="add a reasoning step")
    parser.add_argument("--load", default=None, help="score a saved program")
    parser.add_argument("--eval-split", default="test", choices=("dev", "test", "all"))
    parser.add_argument("--include-unreviewed", action="store_true")
    parser.add_argument("--domain", default="guantanamo")
    parser.add_argument("--eval-dir", default=None, help="default: {output_dir}/eval")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None, help="score first N only")
    args = parser.parse_args()

    eval_dir = args.eval_dir or os.path.join(
        DomainConfig(args.domain).get_output_dir(), "eval"
    )

    def cases(split: str) -> List[Dict[str, Any]]:
        return load_cases(eval_dir, split, args.include_unreviewed)

    dspy.configure(lm=make_lm(args.model, max_tokens=16384))
    program = ProfileUpdater(base_instructions(args.domain), chain_of_thought=args.cot)
    module = "cot" if args.cot else "predict"
    if args.load:
        program.load(args.load)
        optimizer = "loaded"
    else:
        optimizer = args.optimizer
        if optimizer == "gepa":
            if not args.reflection_model:
                parser.error("--optimizer gepa needs --reflection-model")
            train = [to_example(c) for c in cases("train")]
            val = [to_example(c) for c in cases("dev")]
            console.print(f"GEPA ({args.auto}): {len(train)} train / {len(val)} val")
            program = dspy.GEPA(
                metric=metric,
                auto=args.auto,
                reflection_lm=make_lm(args.reflection_model, max_tokens=16384),
                num_threads=args.workers,
            ).compile(program, trainset=train, valset=val)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        out = os.path.join(
            eval_dir,
            "dspy",
            f"profile-{module}-{optimizer}-{args.model.replace('/', '_')}-{stamp}.json",
        )
        os.makedirs(os.path.dirname(out), exist_ok=True)
        program.save(out)
        console.print(f"Saved program to {out}")

    held_out = cases(args.eval_split)[: args.limit]
    records = run_updates(held_out, make_updater(program), workers=args.workers)
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task": "profile_update",
        "dataset": "profile_update",
        "model": args.model,
        "approach": f"dspy-edit-{module}-{optimizer}",
        "split": args.eval_split,
        **summarize_updates(records),
    }
    path = save_run(eval_dir, summary, records)
    console.print(
        f"{summary['approach']} on {args.eval_split}: pass {summary['pass_rate']}, "
        f"errors {summary['errors']}/{summary['n']}, "
        f"dispute claims {summary['dispute_claims']}, "
        f"unsupported numbers {summary['unsupported_numbers']}. Runs: {path}"
    )


if __name__ == "__main__":
    main()
