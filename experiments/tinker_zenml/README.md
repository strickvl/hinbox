# Tuned citation checker: Kitaru + ZenML + Tinker

The goal: a small local model (Inkling 2 Mini, LoRA-tuned on Tinker) that checks Hinbox's citations as well as a cloud judge (Haiku, Sonnet, Opus) at a fraction of the cost and time, and runs with no internet once its weights are downloaded. This folder is the part that runs the training as a ZenML pipeline; the rest of the loop lives in Hinbox (`just citecheck`, `scripts/kitaru_step.py`, `scripts/kitaru_evaluators/`).

It is its own uv project (ZenML 0.99, Tinker SDK, tinker-cookbook, Kitaru 0.27.8), so none of that enters Hinbox's main environment. Everything it writes that holds article text stays in the `--data-dir` you give it (keep that under `data/`). ZenML only receives paths, counts, the checkpoint name and scores.

## What happens, start to finish

A worked example with the numbers made up:

1. **Test set.** `just citecheck sample` draws 400 claims from articles the training never sees (15% of articles, chosen by a hash of the article id). Half come from sentences the run's checker doubted, so there are enough wrong claims to tell checkers apart.
2. **Baseline sessions in Kitaru.** `just citecheck kitaru-sessions --model anthropic/claude-haiku-5-5 ...` asks Haiku about each claim, one claim per call, through the pipeline's own `judge_claims()`. Each call is recorded as one Kitaru session: inputs are exactly the judge's arguments (article, claim, model) plus `labels`; outputs are the verdict. Import the file with `kitaru session import`.
3. **Labelling in Kitaru.** `just citecheck kitaru-investigate` opens one investigation over those sessions, one question each ("Does the article state everything in this sentence about Mara Voss? yes, no or unsure"), with the sentence and the article highlighted. You answer in Kitaru's investigation view. (The local labelling page, `just citecheck label`, writes the same label file if you prefer it.)
4. **Answers back.** `just citecheck kitaru-pull --investigation <id>` writes each answer to `test_labels.jsonl` and into the session's `labels.gold` and metadata, so every later replay and its evaluator know the right answer.
5. **Training data.** `plant`, `teach`, `audit` (see the main README): planted changed-detail errors plus claims where Opus and Sonnet agree.
6. **The pipeline** (this folder): export → train on Tinker → sample the test prompts → score → Kitaru experiment with the new checkpoint as the model override.
7. **Comparison.** The experiment replays every labelled session with the tuned model and scores both sides with the `hinbox-citation-check-checks` evaluator. Say Haiku passed 9 of the 200 wrong claims (`wrong_passed` 0.045) in 3.1 s and $0.0021 per claim; the tuned model might pass 7 (0.035) in 1.4 s at $0. Kitaru shows those means side by side, per evaluation.

## One-time setup in Kitaru

```bash
# the step agent: its run command replays a recorded citation check
kitaru agent register hinbox-citation-check --spec spec.yaml
```

with `spec.yaml`:

```yaml
version:
  run_spec:
    command: uv run --project experiments/tinker_zenml python -m scripts.kitaru_step
    working_dir: /path/to/hinbox
    runtime_capabilities: {overrides: true, tool_policies: false}
```

(The Tinker project's environment, so a replay can sample a `tinker://` checkpoint; for cloud and local judges only, `uv run --extra kitaru` works too.)

```bash
uvx --from 'kitaru[cli]' kitaru evaluator register hinbox-citation-check-checks \
    --script scripts/kitaru_evaluators/citation_check_checks.py \
    --entrypoint evaluate --agent-id <hinbox-citation-check agent id>
just kitaru-worker-tinker          # keep it running: it does the replays and evaluations
```

## Commands, in order

```bash
D=data/citecheck; A=data/guantanamo/raw_sources/miami_herald_articles.parquet
just citecheck sample --entities <run>/entities --articles $A --out $D
just citecheck kitaru-sessions --items $D/test_items.jsonl --articles $A \
    --model anthropic/claude-haiku-5-5 --file $D/sessions.haiku.jsonl
kitaru session import $D/sessions.haiku.jsonl --importer kitaru/kitaru-jsonl@latest \
    --agent hinbox-citation-check@latest --media-type application/x-ndjson --wait
just citecheck kitaru-investigate --name "Citation test set"
#   ... answer in Kitaru ...
just citecheck kitaru-pull --investigation <id> --labels $D/test_labels.jsonl
just citecheck plant ... && just citecheck teach ... && just citecheck audit ...
just tinker-pipeline --data-dir $D --articles $A --kitaru --prices prices.json
```

`prices.json` prices models Kitaru has no price for, per million tokens, e.g. `{"tinker://.../sampler_weights/citecheck": [0, 0], "ollama/inkling-citecheck": [0, 0]}`. Any other judge can be compared the same way without training: `just citecheck kitaru-experiment --name "Opus" --model anthropic/claude-opus-5-5`, or a local model served after download (`ollama/...`).

## The pipeline's steps

| step | does | ZenML records |
| --- | --- | --- |
| `prepare_data` | `citecheck export` and `prompts` if their files are missing | row counts |
| `train_on_tinker` | LoRA training (code from `scripts/tinker_citecheck.py`, effort 0) and a sampler checkpoint | checkpoint path, loss curve ends, minutes, config |
| `sample_test_set` | one reply per test prompt at temperature 0 | replies, mean output tokens and seconds |
| `score_test_set` | `citecheck score --replies` against the hand labels | wrong passed (with 95% range), good doubted, failed replies |
| `report_card` | one HTML page (`report_card.py`): both rates with 95% ranges, any `--baselines` beside them, the loss curve, rates per kind of claim, settings | the `training_report` artifact (`HTMLString`, shown in the dashboard; numbers only) |
| `compare_in_kitaru` | Kitaru experiment with the checkpoint as override (`--kitaru`) | experiment and run ids, baseline and tuned mean per evaluation; the `kitaru_report` artifact (`HTMLString`: each measure, baseline against tuned) |

`--baselines baselines.json` puts other checkers' scores on the same test set beside the tuned one in the report card: `{"Untuned Inkling 2 Mini": {"wrong_passed": 36, "good_doubted": 26}}`, each a count of the test set's wrong or good claims, or `[count, of]`.

`train_on_tinker` also returns the loss curve (`loss_log`: step, train and tune loss) as an artifact.

## A ZenML server with an S3 artifact store

The project carries ZenML's S3 integration packages (`s3fs`, `boto3`, `aws-profile-manager`), so after `zenml login <server>` a stack can keep artifacts in a bucket:

```bash
Z="uv run --project experiments/tinker_zenml zenml"
$Z artifact-store register s3_store --flavor=s3 --path=s3://<bucket>/hinbox
$Z stack register hinbox -o default -a s3_store --set
```

Credentials come from the machine's AWS profile unless the store is given a service connector. Only counts, paths, scores, the loss curve and the report card are written there.

`just tinker-pipeline-test` runs the whole pipeline on a local ZenML store with Tinker and Kitaru replaced by fakes.

## Known risks

- Getting the weights out: Tinker's documented path is `save_weights_external()` and then `tinker checkpoint download`, and on 25 Nov a Thinking Machines engineer confirmed on X that Inkling 2 Mini LoRAs download this way. Nobody has said so for the `:free` model ID specifically, so `scripts/tinker_citecheck.py smoke` checks it first: 15 training steps on the free ID, then the download, then a list of the tensors. The cookbook's PEFT adapter builder lists no Inkling support; smoke tries it and reports what happens. If the raw LoRA can't be turned into something llama.cpp serves, the fallback is training with Unsloth on the same data.
- Free tier: it needs a positive prepaid balance and keeps at most 100 checkpoints or 100 GB, so delete smoke and old checkpoints (`tinker checkpoint delete`).
- The tuned model must be served with the same "Thinking effort level: 0.0" system message it was trained with.
- Kitaru's means are per session; the sessions hold one claim each so the means equal per-claim shares.
