# Handoff: using Kitaru to improve Hinbox

Written 14 Nov 2026 at the end of a cloud session, and updated the same day in a local session that rebased the branch onto `main`, imported a real run into the cloud Kitaru workspace and changed the import route (see "Part 1"). Branch: `claude/determined-rubin-swvdij`.

## The goal

Use Kitaru on Hinbox itself (dogfooding). That needs two things:

1. **A way to get Hinbox's traces into Kitaru** (an importer), so Kitaru's insights, investigations and evaluators can look at every run.
2. **A replay adapter**, so Kitaru can re-run a piece of Hinbox with one thing changed (a model, a prompt) and compare the results against the original.

Part 1 is done (PR #138). Part 2 is designed below; its step 1 (recording step arguments) is done, the rest is not built.

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

## Part 2: the replay adapter (designed, not built)

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
   - **Eval traces don't record arguments:** `just eval profile-update` with the default edit approach calls `edit_profile()` inside its own `profile.update` span (`_update_step()` in `src/evals/profile_update.py`), not `update_profile()`, so its spans carry no `input.value` and make no step sessions.
   - **Privacy:** this is article text, so the same rule applies as now: keep it local.
   - **Not yet:** `relevance`, `extract.<type>`, `profile.create` and grounding record nothing new. Add them the same way when a replay needs them.
2. **One session per step (done).** `just kitaru-traces <traces...> --out <dir> --step profile.update` writes each `profile.update` span that recorded its arguments as its own session (`step_sessions()`): its inputs are exactly those arguments (what a replay gets in `KITARU_TASK_INPUTS`), its outputs the recorded result, its nodes the step's spans and model calls; it fails when the step span failed or recorded an `error`. Its `external_id` is `hinbox:<trace id>:<span id>`, and its metadata names the step, the article ID and title, and the article's own session (`article_session`). A step span without recorded arguments (traces from before #139) is left out. Import these under their own agent, `hinbox-profile-update`, which the replay script will be registered as. Checked on 14 Nov with four real updates imported into a scratch agent: the session's inputs held the old profile, article text, dates and model.
3. **A replay script**, `scripts/kitaru_step.py`, registered as the agent's run command, for example `kitaru agent register hinbox-profile-update --command "uv run python -m scripts.kitaru_step"`. It would:
   - read the step input from `KITARU_TASK_INPUTS`, or from the task's spec through `client.tasks.get_spec()`;
   - read the override through `client.replays.get(KITARU_REPLAY_ID)`;
   - create a session plus a root node at index 0;
   - call `update_profile(**inputs, model=…)`;
   - record each model call as an `llm_call` node, by hooking the same `create` wrapper tracing uses (`_traced_create()` in `src/utils/tracing.py`);
   - finish the root node and the session, marking them failed on error.

   Declare `runtime_capabilities.tool_policies: false`. Hinbox has no tools: Instructor's "tool call" is a way to get structured output, not a tool.
4. **Evaluators.** Port the deterministic checks from `src/evals/profile_update.py` (collapse, kept facts, unchanged text, invented citations) as Kitaru evaluators, using the `kitaru-validate-evaluator` and `kitaru-investigation` skills. A replay without evaluators only produces two profiles to compare by eye.

My rough guesses: 1–2 days for steps 1 and 2, 2–4 days for step 3 with tests, 1 day for the first two evaluators.

### How replay changes map onto Hinbox

| Kitaru override | Fit with Hinbox |
|---|---|
| `model` | Easy. `update_profile(model=…)` exists, as do `HINBOX_<TASK>_MODEL` variables. Kitaru writes models as `provider:name`; Hinbox uses `openai/gpt-6-luna` and `ollama/…`, so translate between them. |
| `model_params` (temperature etc.) | Easy: apply them in the `create` wrapper, after `_sampling_kwargs()`. Claude 5-family models take no `temperature`. |
| `system_prompt` | Awkward. One `profile.update` makes several model calls: the edit call, the dispute check, and the reorganise, placement and split calls. Each has its own prompt, so "replace the system prompt" doesn't say which one. Better: test a prompt change by registering a new agent version whose working directory is a git worktree with the edited prompt file. Kitaru replays a baseline against any agent version (`agent_version_id`; see `replay-and-overrides.md`). |
| `prompt` (user prompt) | No sensible mapping; declare it unsupported. |
| Tool policies | Not applicable (no tools). Every replay makes live model calls, which costs money or local GPU time. |

### Sync versus async

- **The mismatch:** Kitaru's recording client (`KitaruAPIClient`, with `sessions.create`, `sessions.ingest_nodes` and `sessions.update`) is async. `KitaruSyncClient` in `src/kitaru/client/sync_client.py` (Kitaru repo) has no session or node write methods; I checked. Hinbox is synchronous and uses thread pools.
- **The plan:** because the replay script is our own process, it can start with `asyncio.run(main())`, run the Hinbox step in a thread with `asyncio.to_thread`, collect model-call records in a thread-safe list, and write the nodes afterwards.
- **Why that's acceptable:** the `kitaru-adapter-builder` skill warns against bridges between sync and async code. Its worry is breaking cancellation or context inside someone else's framework. Here we own the whole process, so that risk mostly doesn't apply, but a reviewer should know the rule exists.

### Limits to expect

- **Replays won't match exactly.** A replay with nothing changed won't reproduce the original, because model output varies and nothing is answered from the recording. Run each case more than once, or compare score distributions, not single runs.
- **Some steps have extra calls.** A `profile.update` replay also re-runs the dispute check and any reorganise or split the profile size triggers, so one replayed step can mean 1 to about 6 model calls.

## Decisions made

1. **How Kitaru gets installed:** not in Hinbox's dependencies for now; run the CLI with `uvx --from 'kitaru[cli]' kitaru` or `uv tool install kitaru`. Once `scripts/kitaru_step.py` exists, add Kitaru as an optional extra (`uv sync --extra kitaru`), like the `anthropic` extra, because the worker runs the replay script inside Hinbox's environment and it imports the Kitaru SDK.
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
