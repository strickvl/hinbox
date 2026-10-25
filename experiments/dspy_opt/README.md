# DSPy experiments

This folder holds experiments that use [DSPy](https://dspy.ai) to improve Hinbox's prompts for a particular model, especially small local models. Nothing here is used by the pipeline. `src/` does not import from this folder.

## Why it lives on its own

DSPy brings many dependencies, so it has its own environment (`pyproject.toml` and `uv.lock` in this folder). Hinbox's main `uv sync` never installs it.

**LiteLLM is never installed.** DSPy 3.4 still lists LiteLLM as a dependency on Python < 3.15, but it doesn't need it. Its built-in "lm15" engine calls OpenAI, Anthropic, Gemini and Ollama directly. `pyproject.toml` overrides the LiteLLM requirement with a condition that can never be true (Python 2). As a result, LiteLLM is not in `uv.lock` and is never downloaded. `lm.py` also pins `engine="lm15"`, so an unsupported setting fails loudly instead of trying to fall back to LiteLLM. `tests/test_merge.py` checks that LiteLLM can't be imported.

## What's here

| File | What it does |
|---|---|
| `lm.py` | Turns a Hinbox model string (`openai/…`, `anthropic/…`, `gemini/…`, `ollama/…`) into a DSPy model. It uses the same environment variables as Hinbox (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OLLAMA_API_URL`, `HINBOX_OLLAMA_REASONING_EFFORT`). |
| `merge.py` | The merge check ("are these two profiles the same entity?") written as a DSPy program, plus the optimisation run. |
| `profile_update.py` | The edit-list profile updater (`src/engine/profile_edits.py`) as a DSPy program, optimised with GEPA and scored by the profile-update eval's checks. |
| `tests/` | Offline tests that make no model calls. |

## Running it

From the repository root:

```bash
# Unoptimised DSPy version of the merge check
just dspy-merge --model ollama/qwen3.6:35b-a3b-q4_K_M --optimizer none

# Optimised versions
just dspy-merge --model ollama/qwen3.6:35b-a3b-q4_K_M --optimizer bootstrap
just dspy-merge --model ollama/qwen3.6:35b-a3b-q4_K_M --optimizer mipro
just dspy-merge --model ollama/qwen3.6:35b-a3b-q4_K_M --optimizer gepa --reflection-model openai/gpt-6-luna

# Re-score a saved program, e.g. on another split
just dspy-merge --model ollama/qwen3.6:35b-a3b-q4_K_M --load data/guantanamo/entities/eval/dspy/<file>.json

just eval show      # results sit next to the pipeline baselines
just dspy-test      # offline tests
```

Each run does three things:
1. It saves the optimised program (its instructions and chosen examples) to `{output_dir}/eval/dspy/`.
2. It scores the program on the held-out `test` split.
3. It adds a row to the shared scoreboard with `approach = dspy-<cot|predict>-<optimizer>`. The pipeline's own prompt shows up there as `approach = pipeline`.

By default the program reasons step by step before answering (`ChainOfThought`). Pass `--predict` to have it answer directly.

## What each optimizer does

- **`none`** runs the DSPy version of the prompt as written, with no optimisation. It is the fair starting point for the other three, because DSPy formats prompts differently from the pipeline.
- **`bootstrap`** runs the model on training pairs and keeps a few cases it got right as worked examples. Those examples are then shown with every future question.
- **`mipro`** (MIPROv2, "light" setting) asks the model to write several alternative instructions. It tries combinations of instructions and examples, and keeps the combination that scores best on the validation pairs.
- **`gepa`** runs the model, shows a stronger "reflection" model the cases it got wrong (with the human label), and has it rewrite the instructions. It repeats this for several rounds. It needs `--reflection-model`, and that model's calls cost money if it's a cloud model.

## Data and its limits

The data is the hand-labelled `merge_gold_v2.jsonl` set. The optimizer only sees the `train` and `dev` splits (119 pairs). It holds back half of their 10 "same entity" pairs, each with three "different" pairs, as a validation set. It does this because a random validation set would be about 92% "different", and would reward a model that always answers "different".

Two cautions when reading results:
- **There are only 10 "same" examples to learn from.** Expect noisy results, and run more than one seed before trusting a gain.
- **13 of the 14 "same" pairs in `test` are places with identical names.** The `test` "same" cases are therefore easy. Watch the "different" side (false merges) as closely as F1.

## Profile updates (`profile_update.py`)

```bash
# The pipeline's edit prompt as a DSPy program, unoptimised (scored on test)
just dspy-profile --model ollama/local-qwen36-16k:latest --optimizer none

# GEPA rewrites the instructions; a stronger model does the rewriting
just dspy-profile --model ollama/local-qwen36-16k:latest --reflection-model openai/gpt-6-luna

# Re-score a saved program on dev
just dspy-profile --model ollama/local-qwen36-16k:latest --load data/guantanamo/entities/eval/dspy/<file>.json --eval-split dev
```

- **Starting point:** the instructions are the domain's `profile_edit.md`, or `profile_edit_local.md` for an `ollama/` model (the pipeline's own choice; override with `--base-prompt`). `profile_edit_local.md` is GEPA's result for Qwen3.6. The prompt is used with `{entity_name}` written as "the entity" and `{article_id}` as `<article_id>`. The entity and article come in as their own input fields. To use an optimised prompt in the pipeline, copy its instructions back into the matching prompt file and put the placeholders back.
- **Data:** GEPA trains on the `train` entities' cases from `profile_update_cases.jsonl` and validates on `dev`. Scores are reported on `test` (Camp 7, Vance Spath). Disputed cases still marked `needs_review` are left out unless you pass `--include-unreviewed`. Some of them pair two different events, and training on those would teach the model to invent disagreements.
- **Score:**
  - A case that passes the eval's checks scores 1.
  - A failed case scores half of its kept-fact share.
  - Each invented "the sources disagree" sentence and each sentence with a number the article doesn't contain costs 0.25.
  - GEPA also reads the failure reasons in words.
- **Instructions only:** no few-shot examples. One example is a whole profile plus an article, up to ~30k characters.
- The program answers directly (`Predict`) unless you pass `--cot`.

## Not done yet

- The pipeline can't use an optimised program yet. A later change could load a saved program in `src/engine/match_checker.py` behind a setting.
- The merge check and profile updates exist so far. Relevance and extraction would follow the same pattern: one file per task in this folder.
