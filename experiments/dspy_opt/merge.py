"""Optimise Hinbox's merge check ("are these two profiles the same entity?") with DSPy.

The pipeline asks this question with a hand-written prompt in
``src/engine/match_checker.py``. Here the same question is a DSPy program,
which an optimizer can improve for a given (small, local) model by choosing
few-shot examples and rewriting the instructions. Results go on the same
scoreboard as the pipeline baselines, under ``approach = dspy-<module>-<optimizer>``.

Data: the hand-labelled ``merge_gold_v2.jsonl`` set. The optimizer only sees the
``train`` and ``dev`` splits; scores are reported on ``test`` (by default).

Usage (from the repository root)::

    just dspy-merge --model ollama/qwen3.6:35b-a3b-q4_K_M --optimizer mipro
    just dspy-merge --model ollama/qwen3.6:35b-a3b-q4_K_M --optimizer none   # unoptimised DSPy program
    just dspy-merge --model ollama/qwen3.6:35b-a3b-q4_K_M --load <saved.json>  # re-score a saved program
"""

import argparse
import os
import random
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import dspy

from experiments.dspy_opt.lm import make_lm
from src.config_loader import DomainConfig
from src.engine.match_checker import _TYPE_SPECIFIC_RULES
from src.evals.datasets import EvalItem, filter_split, load_merge_items
from src.evals.runner import run_eval, save_run, summarize
from src.logging_config import console

INPUT_FIELDS = (
    "entity_type",
    "rules",
    "new_name",
    "new_profile_text",
    "existing_name",
    "existing_profile_text",
)
OPTIMIZERS = ("none", "bootstrap", "mipro", "gepa")


class MatchEntities(dspy.Signature):
    """Decide whether two profiles from news articles about Guantánamo Bay refer
    to the same real-world entity. Consider name variations (spellings,
    nicknames, titles, abbreviations, partial names), context (role,
    affiliations, actions, biographical details) and whether both profiles could
    describe one entity at different times. A vague generic phrase is almost
    never the same entity as a specific named one. Follow the rules given."""

    entity_type: str = dspy.InputField(
        desc="people, organizations, locations or events"
    )
    rules: str = dspy.InputField(desc="entity-type rules that must be followed")
    new_name: str = dspy.InputField()
    new_profile_text: str = dspy.InputField(desc="evidence from the new article")
    existing_name: str = dspy.InputField()
    existing_profile_text: str = dspy.InputField(desc="profile already in the database")
    is_match: bool = dspy.OutputField(desc="true if both describe the same entity")


class MatchChecker(dspy.Module):
    def __init__(self, chain_of_thought: bool = True):
        super().__init__()
        step = dspy.ChainOfThought if chain_of_thought else dspy.Predict
        self.decide = step(MatchEntities)

    def forward(self, **inputs: Any) -> dspy.Prediction:
        return self.decide(**inputs)


def to_example(item: EvalItem) -> dspy.Example:
    fields = dict(
        item.inputs,
        rules=_TYPE_SPECIFIC_RULES.get(item.inputs["entity_type"], "").strip(),
    )
    return dspy.Example(**fields, is_match=item.label).with_inputs(*INPUT_FIELDS)


def correct(example: dspy.Example, pred: dspy.Prediction, trace: Any = None) -> bool:
    return bool(pred.is_match) == bool(example.is_match)


def gepa_metric(
    gold, pred, trace=None, pred_name=None, pred_trace=None
) -> dspy.Prediction:
    """GEPA learns from written feedback, not just a score."""
    if correct(gold, pred):
        return dspy.Prediction(score=1.0, feedback="Correct.")
    truth = "the same entity" if gold.is_match else "different entities"
    return dspy.Prediction(
        score=0.0,
        feedback=(
            f"Wrong. A human labelled '{gold.new_name}' and '{gold.existing_name}' "
            f"({gold.entity_type}) as {truth}."
        ),
    )


def split_for_optimizer(
    examples: List[dspy.Example], seed: int
) -> Tuple[List[dspy.Example], List[dspy.Example]]:
    """Hold out a validation set with a higher share of "same" pairs.

    Only about 1 pair in 12 is a real match, so a plain random valset rewards
    "always no". Half the positives go to validation, with three negatives each.
    """
    rng = random.Random(seed)
    pos = [e for e in examples if e.is_match]
    neg = [e for e in examples if not e.is_match]
    rng.shuffle(pos)
    rng.shuffle(neg)
    n_val_pos = max(1, len(pos) // 2)
    val = pos[:n_val_pos] + neg[: 3 * n_val_pos]
    train = pos[n_val_pos:] + neg[3 * n_val_pos :]
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def optimize(
    program: dspy.Module,
    optimizer: str,
    train: List[dspy.Example],
    val: List[dspy.Example],
    *,
    threads: int,
    reflection_model: Optional[str],
) -> dspy.Module:
    if optimizer == "none":
        return program
    if optimizer == "bootstrap":
        opt = dspy.BootstrapFewShot(
            metric=correct, max_bootstrapped_demos=4, max_labeled_demos=8
        )
        return opt.compile(program, trainset=train)
    if optimizer == "mipro":
        opt = dspy.MIPROv2(metric=correct, auto="light", num_threads=threads)
        return opt.compile(program, trainset=train, valset=val)
    if optimizer == "gepa":
        if not reflection_model:
            raise SystemExit(
                "--optimizer gepa needs --reflection-model (a stronger model)"
            )
        opt = dspy.GEPA(
            metric=gepa_metric,
            auto="light",
            reflection_lm=make_lm(reflection_model),
            num_threads=threads,
        )
        return opt.compile(program, trainset=train, valset=val)
    raise ValueError(f"Unknown optimizer {optimizer!r}")


def make_predictor(program: dspy.Module):
    def predict(inputs: Dict[str, Any]) -> bool:
        example = to_example(EvalItem(item_id="", label=False, inputs=inputs))
        return bool(program(**example.inputs()).is_match)

    return predict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--model", required=True, help="e.g. ollama/qwen3.6:35b-a3b-q4_K_M"
    )
    parser.add_argument("--optimizer", choices=OPTIMIZERS, default="mipro")
    parser.add_argument(
        "--predict", action="store_true", help="no chain-of-thought step"
    )
    parser.add_argument(
        "--reflection-model", default=None, help="GEPA only, e.g. openai/gpt-6-luna"
    )
    parser.add_argument(
        "--load", default=None, help="score a saved program instead of optimising"
    )
    parser.add_argument("--eval-split", default="test", choices=("dev", "test", "all"))
    parser.add_argument("--domain", default="guantanamo")
    parser.add_argument("--eval-dir", default=None, help="default: {output_dir}/eval")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--limit", type=int, default=None, help="score only the first N items"
    )
    args = parser.parse_args()

    eval_dir = args.eval_dir or os.path.join(
        DomainConfig(args.domain).get_output_dir(), "eval"
    )
    items = load_merge_items(eval_dir, "v2")
    pool = [to_example(i) for s in ("train", "dev") for i in filter_split(items, s)]
    held_out = filter_split(items, args.eval_split)[: args.limit]

    dspy.configure(lm=make_lm(args.model))
    program = MatchChecker(chain_of_thought=not args.predict)
    module = "predict" if args.predict else "cot"
    if args.load:
        program.load(args.load)
        optimizer = "loaded"
    else:
        optimizer = args.optimizer
        train, val = split_for_optimizer(pool, args.seed)
        console.print(
            f"Optimising with {optimizer}: {len(train)} train / {len(val)} val examples"
        )
        program = optimize(
            program,
            optimizer,
            train,
            val,
            threads=args.workers,
            reflection_model=args.reflection_model,
        )
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        out = os.path.join(
            eval_dir,
            "dspy",
            f"merge-{module}-{optimizer}-{args.model.replace('/', '_')}-{stamp}.json",
        )
        os.makedirs(os.path.dirname(out), exist_ok=True)
        program.save(out)
        console.print(f"Saved optimised program to {out}")

    records = run_eval(held_out, make_predictor(program), workers=args.workers)
    summary = summarize(
        records,
        task="merge",
        model=args.model,
        split=args.eval_split,
        approach=f"dspy-{module}-{optimizer}",
        dataset="merge_v2",
    )
    save_run(eval_dir, summary, records)
    console.print(
        f"{summary['approach']} on {args.eval_split}: balanced accuracy {summary['balanced_accuracy']:.3f}, "
        f"F1 {summary['f1']:.3f}, errors {summary['errors']}/{summary['n']}"
    )


if __name__ == "__main__":
    main()
