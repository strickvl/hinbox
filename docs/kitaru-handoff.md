# Handoff: using Kitaru to improve Hinbox

Written 14 Nov 2026 at the end of a cloud session, and updated the same day in a local session that rebased the branch onto `main`, imported a real run into the cloud Kitaru workspace and changed the import route (see "Part 1"). Branch: `claude/determined-rubin-swvdij`.

## The goal

Use Kitaru on Hinbox itself (dogfooding). That needs two things:

1. **A way to get Hinbox's traces into Kitaru** (an importer), so Kitaru's insights, investigations and evaluators can look at every run.
2. **A replay adapter**, so Kitaru can re-run a piece of Hinbox with one thing changed (a model, a prompt) and compare the results against the original.

Part 1 is done (PR #138). Part 2 is built and was tried on the cloud workspace on 14 Nov ("Part 2" below): step sessions, the replay script, and a first evaluator. What is left is under "Still open".

## Short version

- **Hinbox writes Kitaru's own session format.** `just kitaru-traces` turns Hinbox's trace files into `kitaru-jsonl` sessions, and they import into the cloud workspace with the built-in `kitaru/kitaru-jsonl` importer. The first try used Kitaru's `kitaru/phoenix` importer on the raw spans: everything imported, but the viewer showed model calls as raw JSON and steps as empty nodes (details below; Kitaru issue [zenml-io/kitaru#1517](https://github.com/zenml-io/kitaru/issues/1517)).
- **Replay is possible, but only if one Kitaru session is one pipeline step** (one `profile.update`, one `match_check`), not one article. Hinbox would need to start recording each step's arguments, plus a small replay script. Details under "Part 2".
- A correction to an earlier assumption: Hinbox does not keep traces in SQLite. It writes `{output_dir}/traces/run-<timestamp>.jsonl`. The SQLite file is Phoenix's own database under `~/.phoenix/` when you run `just phoenix`.

## Part 1: importing traces (done)

### Where things stand

The cloud workspace this project uses (`kitaru status` shows the URL; server 0.27.6) has one agent, `hinbox`, holding one run: the 9 Nov full-run trace (`run-20261109-212405.jsonl`, 43 articles), imported as 66 sessions with the tag `run-20261109-212405`. Of those, 42 are completed articles, 1 is article #31, which Hinbox marked incomplete (`extract_locations` hit the token limit), and 23 are "unfinished" traces: articles extracted ahead of the merge thread when the run stopped, whose article span was never written.

Only a small sample goes into the cloud workspace, not every run: the traces hold full article text.

### Why not the Phoenix importer

The first import read the raw spans with `kitaru/phoenix`, after copying `span_kind`, `status_code` and `project` to the top level of each line (which that importer needs). It imported every session, model call and failure, but in the viewer:

- **A model call** showed its messages as a folded JSON tree of `message.role`/`message.content` fields, not as a conversation. The importer keeps OpenInference's field names and sets none of the text selectors (the JSON Pointers that tell the viewer where the system prompt, prompt and reply are).
- **A step** (`extract.locations`, `profile.update`) showed no inputs or outputs. The importer only reads `input.value`/`output.value`, which Hinbox writes on model calls, not on steps; a step only carries `hinbox.*` labels.
- **A session's status** came from its top span alone, so an article Hinbox marked incomplete still showed "completed".

That route was dropped, and the top-level-fields change to `src/utils/tracing.py` was reverted. The importer fix is tracked in [zenml-io/kitaru#1517](https://github.com/zenml-io/kitaru/issues/1517).

### What `scripts/kitaru_traces.py` writes

One `kitaru-jsonl` session line per trace:

| Part of the trace | Becomes |
|---|---|
| Model-call span | `llm_call` node: messages as `{"role", "content"}` (tool calls nested as `{"id", "function": {"name", "arguments"}}`), `hinbox.llm.reasoning` as a `reasoning` message, selectors for the system prompt, the last user message and the reply (the tool-call arguments for a structured reply), model, provider, tokens, request parameters without the response schema |
| Step span | `span` node: `hinbox.*` labels about what went in (`INPUT_LABELS`: entity name, article ID, names compared…) as inputs, the rest as outputs; a step that recorded its arguments and result (`profile.update`, `match_check`) shows those, with the labels beside them under `labels` |
| Merge decision events | `decisions` in their `merge.<type>` step's outputs |
| Article span | the session's name, inputs (title, URL, date…) and outputs (outcome); the session fails when the outcome is `relevance_error` or starts with `incomplete` |
| Trace with no top span | a failed session named `unfinished (<first step>)`, metadata `complete: false` |
| Resource | session metadata: run ID, git commit, domain, local flag (not the `HINBOX_*` settings) |

Model-call attributes keep everything except the messages and `input.value`/`output.value`, which are already the node's inputs and outputs, so a part is about half the size of the raw trace file (8.1 MiB for the 15 MiB trial run). Parts are at most 45 MiB, under the cloud workspace's 50 MiB upload limit (`max_blob_size_bytes` in `kitaru status`; a self-run server's default is 100 MiB). Each session is one line, so a trace is never split across parts.

### Import a run

No local worker is needed: the cloud server runs the importer itself.

```bash
uvx --from 'kitaru[cli]' kitaru status            # confirms the workspace login
just kitaru-traces <output_dir>/traces/run-<timestamp>.jsonl --out <scratch>/kitaru
uvx --from 'kitaru[cli]' kitaru session import <scratch>/kitaru/part-001.jsonl \
  --importer kitaru/kitaru-jsonl@latest --agent hinbox@latest \
  --media-type application/x-ndjson --tag run-<timestamp> --wait
```

Notes:
- **Eval traces** (`traces/eval-*.jsonl`, `replay-*.jsonl`) go under a separate agent, `hinbox-eval` (not registered yet), so questions about production runs never count the eval cases (some change a year on purpose).
- **Duplicates:** Kitaru skips a session whose `external_id` (`hinbox:<trace id>`) it already has from the same importer, so importing a run twice is safe. A session can't be deleted on its own, only its whole agent (soft delete), which is how the Phoenix-imported sessions were removed.
- **Cost:** Kitaru shows token counts but no cost; Hinbox doesn't record cost.
- **The Kitaru MCP server** in local Claude sessions may point at another server; the CLI uses the stored login.

## Part 2: the replay adapter (built)

### What a Kitaru replay does

A replay names a baseline session and an agent version. A Kitaru worker starts the agent version's run command in your environment, with these variables set:

| Variable | What it holds |
|---|---|
| `KITARU_API_URL`, `KITARU_API_TOKEN` | Where the server is, and a token for this one task |
| `KITARU_TASK_ID`, `KITARU_TASK_INPUTS` | Which task this is, and the baseline's input |
| `KITARU_REPLAY_ID` | Which replay this is; set only during a replay |

The adapter reads the replay's override: `model`, `system_prompt`, `prompt` or `model_params`. It applies the override where model calls go out, answers tool calls from the recording, and records the re-run as a new session (`origin: replay`). Evaluators then score both the baseline and the replay.

For reference, see these paths in the Kitaru repo:
- `docs/book/concepts/replay.md`
- `docs/book/concepts/workers.md`
- `docs/book/guides/replay-and-overrides.md`
- the PydanticAI adapter at `plugins/packages/pydantic-ai/src/kitaru_pydantic_ai/capability.py`, especially `for_run()` and `wrap_run()`.

### Why one article per session can't be replayed fairly

Say article #43 merged into Khalid Sheikh Mohammed's profile. A fair replay needs the entity tables exactly as they were just before article #43: which entities existed, and what his profile said then. Today's tables already contain articles 44–150, so a replay against them sees different merge candidates and a longer profile. The difference between the two runs would be mostly that, not your change.

Kitaru's usual answer is to answer the outside world from the recording (tool policies). Hinbox reads its tables through in-memory dictionaries and embedding search, not tool calls, so nothing would be recorded or answered. Turning those reads into recorded calls would mean a large refactor of `src/engine/mergers.py` and `src/engine/merge_candidates.py`.

### What works: one step per session

A `profile.update` is self-contained. In: entity type, name and kind, the old profile, the article text and ID, and the article dates. Out: the new profile and its edit report. It needs nothing else from the tables. The same holds for `match_check`, `relevance`, `extract.<type>` and grounding. These are what the profile-update eval cases already are; Kitaru would provide the same kind of harness over real production steps.

A step span used to record only the prompts sent to the model, not the arguments the function was called with. Rebuilding the old profile from the prompt text would break easily (the prompt may show a shortened `profile_view()`, for one), so the arguments are now recorded directly (step 1 below).

### Planned work, in order

1. **Record step inputs and outputs (done for `profile.update` and `match_check`).** `record_step_input()` and `record_step_output()` in `src/utils/tracing.py` put a step's arguments as one JSON object in its span's `input.value` and its result in `output.value` (`input.mime_type`/`output.mime_type` say `application/json`).
   - **`profile.update`** (`update_profile()` in `src/engine/profiles.py`) records `entity_type`, `entity_name`, `kind`, `existing_profile` (the whole dict, not only its text: the `reorganised`, `split_refused_json` and `fact_rows_json` keys change what an update does), `new_article_text`, `new_article_id`, `article_dates`, `model_type`, `domain`, `model` (the argument, often empty), `answering_model` (the model that answers, after defaults) and `approach`. Out: `profile` (the whole updated dict), plus `edits`, `disagreements` and `edit_report` from the edit updater, `fact_report` from the fact-list updater, or `error` when the updater failed and the old profile was kept. These are `update_profile()`'s own parameter names, so a replay can mostly pass them back; it must drop `answering_model` and `labels`, and give a fresh `VersionedProfile()`.
   - **`match_check`** (`_check_match()`) records `backend`, both names, both profile texts (the existing one in full, before `profile_view()` shortens it), `entity_type` and `model`; out: `result` (`is_match`, `confidence`, `reason`), and `memo_hit` when the answer came from the memo with no model call.
   - **Size:** OpenTelemetry sets no length cap on a value by default, but an `OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT` or `OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT` variable would have cut every long one without a word. `span_limits()` now sets both to `UNSET`, which makes the SDK ignore the variables. A test trace with the cap set to 1,000 kept a 547,945-character profile (Carol Rosenberg's, from the full run) and a 76,134-character article whole; its Kitaru part was 1.2 MiB. An update's span is now about the size of the old profile, the new one and the article together, so trace files of long runs grow by that much per update. Measured on 14 Nov: four real `profile.update` calls (Luna, small profiles) each added 1.2-1.5 bytes per character of old profile, article and new profile (JSON wrapping, the `sources` list and the edit report on top). Applying that to the 7 Nov Luna and 9 Nov Qwen 150-article runs (1,301 and 582 updates, 2,645 and 985 match checks) estimates +27% and +28% on their 122 MB and 95 MB trace files, about 0.2 MB per article: roughly a third bigger once small profiles' overhead is counted.
   - **Eval steps record too:** `just eval profile-update` with the default edit approach calls `edit_profile()`, not `update_profile()`, so its steps record `edit_profile()`'s own arguments plus `call: edit_profile` (`edit_step()` in `src/evals/profile_update.py`), and a replay runs that function. Calling `update_profile()` from the eval instead would have changed what it scores: for long people profiles `update_profile()` also reorganises and splits sections. Eval step sessions carry the case's `eval_item_id`, `eval_kind` and `eval_split` in their metadata.
   - **Privacy:** this is article text, so the same rule applies as now: keep it local.
   - **Not yet:** `relevance`, `extract.<type>`, `profile.create` and grounding record nothing new. Add them the same way when a replay needs them.
2. **One session per step (done).** `just kitaru-traces <traces...> --out <dir> --step profile.update` writes each `profile.update` span that recorded its arguments as its own session (`step_sessions()`): its inputs are exactly those arguments (what a replay gets in `KITARU_TASK_INPUTS`), its outputs the recorded result, its nodes the step's spans and model calls; it fails when the step span failed or recorded an `error`. Its `external_id` is `hinbox:<trace id>:<span id>`, and its metadata names the step, the article ID and title, and the article's own session (`article_session`). A step span without recorded arguments (traces from before #139) is left out. Import these under their own agent, `hinbox-profile-update`, which the replay script will be registered as. Checked on 14 Nov with four real updates imported into a scratch agent: the session's inputs held the old profile, article text, dates and model.
3. **A replay script (done)**, `scripts/kitaru_step.py`, the agent's run command (`uv run --extra kitaru python -m scripts.kitaru_step`). It:
   - reads the step's inputs with `kitaru.task.get_task_inputs()` (from `KITARU_TASK_INPUTS`, or the task spec when they were too long for the environment);
   - reads the override with `client.replays.get(KITARU_REPLAY_ID)` and applies `model` (`provider:model` → `provider/model`; a map matches the step's model in either spelling or without the provider). `system_prompt`, `prompt` and `model_params` fail the run with a message saying why;
   - calls the function the step ran, `update_profile()` (with the model that answered, so a replay doesn't pick up another environment's defaults) or `edit_step()`, with tracing on and the finished spans also kept in memory;
   - turns those spans into a session with `step_sessions()`, the same code that converts traces for import, so a replay is laid out like its baseline;
   - sends session and nodes in one request with Kitaru's own `ingest_session()`, and exits 1 if the step raised.

   **How it differs from the plan:** no session is opened before the step runs and no `create` hook was written. Kitaru's core `ImporterBackedAdapter` does the same thing (run the function, then parse its trace into one session and ingest it; `asyncio.run()` only before and after), so the sync/async question goes away: Hinbox's code never runs inside an event loop. The one thing taken from that class's behaviour that matters: the session is sent with `imported_from: hinbox`. Without it the server refuses the nodes (`SessionNotIngestable`: a finished session takes nodes only when it is imported), and `ingest_session()` reports that 409 as "already exists", which is misleading. Unlike that class, the script applies the model override, so the agent version declares `overrides: true`, `tool_policies: false`.
4. **Evaluators (first one done).** `hinbox-profile-update-checks` (`scripts/kitaru_evaluators/profile_update_checks.py`, rows from `step_results()` in `src/evals/kitaru_checks.py`) runs `score_update()`'s checks that need no case label on one session: `update_ok`, `length_ratio` (collapse), `fact_retention` and `invented_citations` pass or fail; `unchanged_share`, `added_sentences`, `cites_article`, `updated` (the article joined `sources`), `dispute_claims` and `unsupported_numbers` are reported. It declares no dependencies, so the worker runs it on its own Python and it imports Hinbox's scorer instead of copying it: the worker must run in Hinbox's environment (`just kitaru-worker`). On the 5 eval cases below its numbers matched the eval run's own scores field for field. Not yet: the kind-specific checks (a planted dispute kept, a labelled fact stated), which need the case's labels in the session, and the `kitaru-validate-evaluator` method (hand labels, then an agreement check).

### What the trial on 14 Nov showed

The workspace now has an agent `hinbox-profile-update` (version 1, its `working_dir` this session's worktree) holding 5 eval step sessions (one entity, the American Civil Liberties Union; 3 `irrelevant` and 2 `new_fact` dev cases; tag `eval-dev-20261114`), the evaluator, and the replays of one `new_fact` case:

| Replay | Result |
|---|---|
| Unchanged (Luna) | completed; every pass/fail check the same as the baseline; length ratio 1.446 against 1.41, 2 added sentences against 1 |
| Model fork onto `anthropic:claude-haiku-5-5` | completed; the model call went to Haiku; every check passed, length ratio 1.552 |
| Two earlier tries | failed: before the `imported_from` fix, and a fork whose worker had no `ANTHROPIC_API_KEY` (the shell had only the OpenAI key; `just kitaru-worker` now passes `--env-file .env`) |

A pipeline step (`update_profile()`, from the 4-update trace of PR #140) was replayed locally with `--inputs`/`--out` and passed too; it was not sent to the workspace.

### Still open

- **The agent version points at this worktree.** Register version 2 with `working_dir` the main checkout once this is merged, or replays fail when the worktree is removed.
- **More baselines.** Replays need step sessions: a few more eval cases, or a short pipeline run's `--step profile.update` sessions. Keep it to a small sample: they hold article text.
- **An experiment over a cohort** (the regression suite): freeze the step sessions into a cohort and replay all of them against a new agent version, e.g. one whose checkout has an edited `profile_edit.cloud.md`.
- **`model_params`** (temperature etc.): would need a hook in `_sampling_kwargs()`; refused for now.
- **`match_check`** records its arguments too; a second step type would need its own `--step match_check` sessions, `call` route and evaluator.

### How replay changes map onto Hinbox

| Kitaru override | Fit with Hinbox |
|---|---|
| `model` | Easy. `update_profile(model=…)` exists, as do `HINBOX_<TASK>_MODEL` variables. Kitaru writes models as `provider:name`; Hinbox uses `openai/gpt-6-luna` and `ollama/…`, so translate between them. |
| `model_params` (temperature etc.) | Easy: apply them in the `create` wrapper, after `_sampling_kwargs()`. Claude 5-family models take no `temperature`. |
| `system_prompt` | Awkward. One `profile.update` makes several model calls: the edit call, the dispute check, and the reorganise, placement and split calls. Each has its own prompt, so "replace the system prompt" doesn't say which one. Better: test a prompt change by registering a new agent version whose working directory is a git worktree with the edited prompt file. Kitaru replays a baseline against any agent version (`agent_version_id`; see `replay-and-overrides.md`). |
| `prompt` (user prompt) | No sensible mapping; declare it unsupported. |
| Tool policies | Not applicable (no tools). Every replay makes live model calls, which costs money or local GPU time. |

### Sync versus async

*Resolved: the script runs the step first and records it afterwards, as Kitaru's `ImporterBackedAdapter` does (step 3 above), so none of the bridge below was needed. Kept for the reasoning.*

- **The mismatch:** Kitaru's recording client (`KitaruAPIClient`, with `sessions.create`, `sessions.ingest_nodes` and `sessions.update`) is async. `KitaruSyncClient` in `src/kitaru/client/sync_client.py` (Kitaru repo) has no session or node write methods; I checked. Hinbox is synchronous and uses thread pools.
- **The plan:** because the replay script is our own process, it can start with `asyncio.run(main())`, run the Hinbox step in a thread with `asyncio.to_thread`, collect model-call records in a thread-safe list, and write the nodes afterwards.
- **Why that's acceptable:** the `kitaru-adapter-builder` skill warns against bridges between sync and async code. Its worry is breaking cancellation or context inside someone else's framework. Here we own the whole process, so that risk mostly doesn't apply, but a reviewer should know the rule exists.

### Limits to expect

- **Replays won't match exactly.** A replay with nothing changed won't reproduce the original, because model output varies and nothing is answered from the recording. Run each case more than once, or compare score distributions, not single runs.
- **Some steps have extra calls.** A `profile.update` replay also re-runs the dispute check and any reorganise or split the profile size triggers, so one replayed step can mean 1 to about 6 model calls.

## Decisions made

1. **How Kitaru gets installed:** the optional `kitaru` extra (`kitaru[cli,worker]>=0.27.7,<0.28`; `uv sync --extra kitaru`), since the worker runs the replay script and the evaluator inside Hinbox's environment. CI doesn't install it; the tests that need it are skipped there.
2. **First step to replay:** `profile.update`.
3. **Eval traces:** a separate `hinbox-eval` agent.

## Where to read further

| What | Where |
|---|---|
| Adapter skill (choosing a route, lifecycle, tests, approval steps) | `kitaru-skills` repo: `skills/kitaru-adapter-builder/SKILL.md` and `references/python-adapters.md`, `references/adapter-method.md` |
| Importer skill (script importers, source map, validation) | `kitaru-skills` repo: `skills/kitaru-importer-builder/SKILL.md` and `references/importer-contract.md` |
| Phoenix importer: accepted formats, how fields map, limits | Kitaru repo: `docs/book/guides/import-phoenix-traces.md` |
| Importer-backed adapters (Phoenix's `PhoenixAdapter` records after the run but can't apply overrides) | Kitaru repo: `docs/book/adapters/importer-backed.md`, `plugins/packages/phoenix-importer/src/kitaru_phoenix_importer/adapter.py` |
| No adapter for your framework | Kitaru repo: `docs/book/adapters/custom.md` |
| Run spec and `runtime_capabilities` | Kitaru repo: `docs/book/concepts/agents-and-sessions.md` |

The versions checked were Kitaru `develop` at `0.27.7+dev` and `kitaru` 0.27.7 from PyPI for the install check. Recheck the installed versions before building Part 2: the adapter skill treats the installed SDK as the authority, not this document.
