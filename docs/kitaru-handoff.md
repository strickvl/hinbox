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
   - reads the override with `client.replays.get(KITARU_REPLAY_ID)` and applies `model` (`provider:model` → `provider/model`; a map matches the step's model in either spelling or without the provider). `model_params` goes on every model call through `set_model_params()` (15 Nov); `system_prompt` and `prompt` fail the run with a message saying why;
   - calls the function the step ran, `update_profile()` (with the model that answered, so a replay doesn't pick up another environment's defaults) or `edit_step()`, with tracing on and the finished spans also kept in memory;
   - turns those spans into a session with `step_sessions()`, the same code that converts traces for import, so a replay is laid out like its baseline;
   - sends session and nodes in one request with Kitaru's own `ingest_session()`, and exits 1 if the step raised.

   **How it differs from the plan:** no session is opened before the step runs and no `create` hook was written. Kitaru's core `ImporterBackedAdapter` does the same thing (run the function, then parse its trace into one session and ingest it; `asyncio.run()` only before and after), so the sync/async question goes away: Hinbox's code never runs inside an event loop. The one thing taken from that class's behaviour that matters: the session is sent with `imported_from: hinbox`. Without it the server refuses the nodes (`SessionNotIngestable`: a finished session takes nodes only when it is imported), and `ingest_session()` reports that 409 as "already exists", which is misleading. Unlike that class, the script applies the model override, so the agent version declares `overrides: true`, `tool_policies: false`.
4. **Evaluators (first one done).** `hinbox-profile-update-checks` (`scripts/kitaru_evaluators/profile_update_checks.py`, rows from `step_results()` in `src/evals/kitaru_checks.py`) runs `score_update()`'s checks that need no case label on one session: `update_ok`, `length_ratio` (collapse), `fact_retention` and `invented_citations` pass or fail; `unchanged_share`, `added_sentences`, `cites_article`, `updated` (the article joined `sources`), `dispute_claims` and `unsupported_numbers` are reported. It declares no dependencies, so the worker runs it on its own Python and it imports Hinbox's scorer instead of copying it: the worker must run in Hinbox's environment (`just kitaru-worker`). On the 5 eval cases below its numbers matched the eval run's own scores field for field. Not yet: the kind-specific checks (a planted dispute kept, a labelled fact stated), which need the case's labels in the session, and the `kitaru-validate-evaluator` method (hand labels, then an agreement check).

### What the trial on 14 Nov showed

The workspace now has an agent `hinbox-profile-update` (see "Where replays run" for its versions) holding 5 eval step sessions (one entity, the American Civil Liberties Union; 3 `irrelevant` and 2 `new_fact` dev cases; tag `eval-dev-20261114`), the evaluator, and the replays of one `new_fact` case:

| Replay | Result |
|---|---|
| Unchanged (Luna) | completed; every pass/fail check the same as the baseline; length ratio 1.446 against 1.41, 2 added sentences against 1 |
| Model fork onto `anthropic:claude-haiku-5-5` | completed; the model call went to Haiku; every check passed, length ratio 1.552 |
| Two earlier tries | failed: before the `imported_from` fix, and a fork whose worker had no `ANTHROPIC_API_KEY` (the shell had only the OpenAI key; `just kitaru-worker` now passes `--env-file .env`) |

A pipeline step (`update_profile()`, from the 4-update trace of PR #140) was replayed locally with `--inputs`/`--out` and passed too; it was not sent to the workspace.

### The first cohort experiment (14 Nov)

Cohort `profile-update-dev` freezes 15 eval dev step sessions: 6 entities, 8 `irrelevant` and 7 `new_fact` cases, all Luna baselines that pass the eval. Version 1 holds the first imports. Version 2 holds the same 15 cases run again after their step sessions began carrying their labels (`labels` in the inputs, below). Two experiments ran on it with evaluator version 3, on agent version 3:

| | Baseline | `luna-unchanged-labelled` | `haiku-fork-labelled` |
|---|---|---|---|
| `case_ok` (the eval's pass rule for the case's kind) | 15/15 | 15/15 | 15/15 |
| added sentences per update | 1.2 | 1.33 | 1.27 |
| length ratio | 1.23 | 1.25 | 1.30 |

Every replay completed, and both runs pass every case. This cohort is too easy to tell models apart: no baseline fails, so there is nothing to fix or break.

**Why the labels are in the inputs:** an agent task gets only the baseline's inputs, and its task token can't read the baseline session (403). An experiment run's settings are frozen once it has runs, so a changed evaluator needs a new experiment. The earlier experiments (`luna-unchanged`, `haiku-fork`, `luna-unchanged-case`, `haiku-fork-case`) stay in the workspace. The `-case` runs failed on that 403.

### The hard cohort (15 Nov)

`case_ok` now applies the eval's own `case_passed()` for every kind: a step session's `labels` also carry the case's `eval_expected` (a disputed case's two years, sources and sentences) and its claim label. Cohort `profile-update-hard` freezes 19 cases Luna failed 2 or more times across 51 past eval runs: dev and train only, at most 2 per entity, 11 entities. On the new Luna baselines `case_ok` gave the same verdict as the eval on all 19. Both experiments ran with evaluator version 4:

| Kind | Cases | Luna baseline | `luna-unchanged-hard` | `haiku-fork-hard` |
|---|---|---|---|---|
| `new_fact` | 9 | 7 | 6 | 9 |
| `disputed` | 6 | 2 | 0 | 2 |
| `irrelevant` | 4 | 2 | 3 | 2 |
| all | 19 | 11 | 9 | 13 |

How to read it:

- **Noise is large.** Luna run again flipped 4 of 19 verdicts against its own baseline: both of al Nashiri's disputed cases from pass to fail, and two cases near a limit. One irrelevant case moved from 3 added sentences to 2, the other way round for a new_fact case. A difference of 2 cases out of 19 is within that noise.
- **New facts lean towards Haiku.** On Camp Justice and the Red Cross, Luna left the article uncited in the baseline and the re-run; Haiku cited it and passed all 9. That is one run each, so it is a hint, not a result: repeat runs would settle it.
- **Disputes are hard for both.** 4 of the 6 disputed cases failed in every run, on both models: neither kept the old year beside the new one. That is a prompt or guard problem, not a model choice, and the place a prompt experiment would start.
- **Irrelevant cases sit on the line.** Their verdicts turn on 2 against 3 added sentences.

### A prompt experiment: disagreements in passing clauses (15 Nov)

**Why the disputed cases fail:** in the 4 that every run fails, the model lists no disagreement at all; no check after the model call removes one. In each case the article gives the other year in a passing clause ("the Pentagon prison system that opened in January 2003", "the October 2001 bombing of the destroyer"). `profile_edit.cloud.md` asks the model to compare only the dates "that concern" the entity, and such a clause doesn't. In one of the 4 (Khadr, case 30) the reason is different: the profile is 22,140 characters, `profile_view()` shows the model 14,585, and the cut leaves out the very sentence with the old year, so no prompt can fix that case. `profile_view()` now keeps such a sentence (see "Still open"), but in two Luna runs with it shown the model still listed no disagreement.

**The change:** the `disagreements` instruction also covers anything the profile itself states, including passing and background clauses, and asks for a check the other way round as well: for each date in the profile, look for the same event anywhere in the article. Local tries on the 6 disputed cases scored 2/6 and 3/6, against 2/6 before. The change ran as experiment `dispute-prompt` on agent version 4, a worktree with the edit (`.worktrees/kitaru-prompt-disputes`, local branch `experiment/dispute-background-clauses`, not pushed):

| Hard cohort | Cases | Luna baseline | Luna again | Prompt change |
|---|---|---|---|---|
| `disputed` | 6 | 2 | 0 | 3 |
| `new_fact` | 9 | 7 | 6 | 5 |
| `irrelevant` | 4 | 2 | 3 | 2 |
| all | 19 | 11 | 9 | 10 |

- **Disputed cases:** Khadr's case 20 passed for the first time; both al Nashiri cases held.
- **New facts:** both Abu Zubaydah cases left the new article uncited, where both Luna runs had cited it. That looks like the extra comparing crowding out the new facts, but it is one run.
- **The easy cohort** stayed at 15/15, with no "sources disagree" sentence invented.
- **Verdict:** no net gain, so the change was not landed.

Two things to take from it. A prompt change needs repeat runs before a 1 or 2 case difference means anything. And Khadr's case 30 is a `profile_view()` problem: the view could always keep the sentences that share a year or an event's words with the article.

### The date check after the edits (15 Nov)

`profile_view()` now keeps every passage with a sentence the article dates differently (`date_pairs()`), so the model sees both years. That changed the view in 7 of 46 long-profile eval cases, and the 6 that weren't Khadr's case 30 passed as often as before. But with both years shown, Luna still listed no disagreement for Khadr's case 30, in two runs out of two.

So the check now runs in code too. `HINBOX_EDIT_DATE_CHECK` (on for cloud models by default since 15 Nov, see below; `on`, `off`) runs the fact-list prototype's date check after the edit-list updater's edits (`src/engine/edit_date_check.py`):

1. code pairs each dated article sentence with the profile's cited sentences that give it another year;
2. one model call judges each pair;
3. each conflict becomes an `add` after the old sentence.

The trial ran as experiment `edit-date-check` on agent version 5. That version is the replay worktree with `HINBOX_EDIT_DATE_CHECK: "on"` in its run spec, so once this is merged it is main with the check on.

| | Luna baseline | Luna again | Date check on |
|---|---|---|---|
| Hard cohort: `disputed` (6) | 2 | 0 | 5 |
| Hard cohort: `new_fact` (9) | 7 | 6 | 6 |
| Hard cohort: `irrelevant` (4) | 2 | 3 | 2 |
| Hard cohort: all (19) | 11 | 9 | 13 |
| Easy cohort (15) | 15 | | 15 |

**Results:**
- **Hard cohort:** al Nashiri's two cases held, and Pohl's and both of Khadr's passed for the first time.
- **Red Cross case:** still fails. The code forms no pair for it: its two sentences share too few words.
- **The rest of the cohort:** on the 28 cases that aren't disputed, the code formed 2 pairs and the model rejected both.
- **Dev split (81 cases):** with the check on, 77 passed against 73 off. Disputed went from 0/5 to 3/5, irrelevant from 36 to 37 of 38, and new_fact stayed at 37/38.
- **Wrong disputes:** none. Of the 7 disputes written across the cohorts and the dev split, every one was real. One was on a real `new_fact` case: the profile says Khadr "pleaded guilty to war crimes in 2011", the article says October 2010, and the article is right.

**The wider check (train and test splits, 161 cases, Luna):**

| Split | Off | On | Disputed off | Disputed on |
|---|---|---|---|---|
| train (82) | 78 | 76 | 2/4 | 4/4 |
| test (79) | 61 | 65 | 0/4 | 1/4 |

- **The pass/fail flips:** every case that went from pass to fail with the check on got no edit from it. That is the model's own run-to-run noise, mostly 3 or 4 sentences added on an irrelevant article.
- **The disputes it wrote:** 5. Three were on disputed cases, all right. One found a real error in Camp X-Ray's profile: Abbasi's release, January 2008 against the correct January 2005. One is wrong: in Spath's profile, "Nashiri was moved in 2004" (out of the CIA site) was paired with "began his U.S. military custody at Guantánamo in September 2006", two different moves.
- **Across all the runs:** 1 wrong dispute in 276 cases.
- **The verdict:** since 15 Nov the check is on by default for cloud models. It stays off for local models until Qwen is measured.

**A pipeline run with the check on (15 Nov):** 40 articles on Luna, main `2837f9b`, 17 minutes, exit 0. The relevance check left 27 to process.

- **Pairs:** in 181 profile updates, code found dated pairs in 7 updates, 10 pairs in all. So the check made 7 extra calls in the whole run.
- **Verdicts:** the model called all 10 `different`, and all 10 were, checked by hand.
  - **Two events, 8 pairs:** the 9/11 accused's arrival in 2006 against the 2007 FBI interrogations; a hearing in March 2019 against a judge who presided from 2012.
  - **A publication date, 1 pair:** "originally published October 19, 2018" against calls made in 2001.
  - **Dates that agree, 1 pair:** bin Attash "captured in 2003 and taken to Guantánamo more than three years later", against the profile's 2006.
- **Disputes written:** none, and no real conflict was missed.
- **What it means:** in ordinary updates the check is cheap and quiet. It matters in the rare update where an article does contradict a profile.

### Repeat runs on the hard cohort (15 Nov)

Six runs on `profile-update-hard`, all on agent version 3 (main, with the date check on for cloud models): three of `luna-unchanged-hard`, three of `haiku-fork-hard`. Every replay completed.

| Kind | Cases | Luna (3 runs) | Haiku (3 runs) |
|---|---|---|---|
| `new_fact` | 9 | 4, 7, 5 | 9, 9, 9 |
| `disputed` | 6 | 5, 5, 5 | 4, 4, 4 |
| `irrelevant` | 4 | 3, 2, 2 | 2, 2, 1 |
| all | 19 | 12, 14, 12 | 15, 15, 14 |

- **Noise:** two Luna runs disagree on 18% of cases, about 3 of 19. Two Haiku runs disagree on 4%, under 1. One Luna run can be off by about 3 cases; a Haiku run is close to its average.
- **New facts:** Haiku passed 27 of 27, Luna 16 of 27. Nearly every Luna failure added the fact but left the article uncited. Camp Justice and Abu Zubaydah's case 20 failed in all three Luna runs.
- **Disputes:** with the date check, Luna passed Pohl and both Khadr cases in every run. Haiku failed Khadr's case 20 in all three, for a reason not yet looked into. The Red Cross case fails for both, because code can't pair it.
- **Irrelevant articles:** Luna held back slightly better, 7 of 12 against Haiku's 5. Haiku added too much on Camp Justice and the hunger strike, but passed U.S. Southern Command every time, which Luna passed 2 times in 3.
- **What it suggests:** on hard cases, Haiku as the profile model is about 2 cases in 19 better, and much steadier. Switching (`HINBOX_PROFILE_MODEL`) is a separate decision: these are only hard cases, cost isn't compared yet, and Haiku is weaker on irrelevant articles. A full dev-split run with Haiku would settle it.

### Haiku on the full dev split (15 Nov)

To check whether Haiku's lead on the hard cohort holds for ordinary cases, `just eval profile-update --split dev` ran once on each model, both on main with the date check on:

| Kind | Cases | Luna | Haiku |
|---|---|---|---|
| `new_fact` | 38 | 36 | 36 |
| `irrelevant` | 38 | 38 | 36 |
| `disputed` | 5 | 4 | 3 |
| all | 81 | 78 | 75 |

| | Calls | Input tokens | Output tokens | Cost at $0.10 / $0.50 per M |
|---|---|---|---|---|
| Luna | 92 | 391k | 25k | $0.051 |
| Haiku | 94 | 637k | 41k | $0.084 |

- **No damage on either side:** neither model collapsed a profile, lost cited sentences or invented a citation.
- **Irrelevant articles:** Haiku failed two USS Cole cases by adding 3 sentences where 2 is the limit. That's the same tendency as on the hard cohort.
- **New facts:** tied. Each model failed two cases the other passed, all by leaving the new article uncited.
- **Cost:** both list at the same price (Luna's from third-party trackers), but Haiku counts the same prompts as about 1.6x the tokens and writes about 1.65x the output. So an update costs about 65% more on Haiku.
- **The verdict:** Luna stays the profile model. One run each is not decisive (Luna's dev score ranged from 73 to 78 across 14-15 Nov), but Haiku would need a clear lead to be worth 65% more, and it trades small advantages instead: better on hard new facts, worse at holding back on irrelevant articles.
- **To run a Haiku eval**, use `uv run --env-file .env ...`. The shell has no `ANTHROPIC_API_KEY`, and without it every case errors.

### Match-check replays (15 Nov)

The second step type. A `match_check` step answers "is this new mention the same entity as this existing profile?". Its sessions (`--step match_check`) carry the merge eval pair's hand label (`eval_label`). Agent `hinbox-match-check` v1 runs `_check_match()` again from `.worktrees/kitaru-replay`, and evaluator `hinbox-match-check-checks@1` scores `match_ok` (agrees with the label), `false_merge` and `missed_merge`.

**Cohort `match-check-hard`** (tag `merge-hard-20261115`): the 28 merge v2 pairs that at least one of 38 earlier eval runs got wrong, 6 labelled the same and 22 different. Luna's baselines are from 15 Nov. Two runs each:

| | Run 1 | Run 2 | False merges | Missed merges |
|---|---|---|---|---|
| Luna baseline | 20/28 | | 6 | 2 |
| `match-luna-unchanged` | 19/28 | 21/28 | 7, 5 | 2, 2 |
| `match-haiku-fork` | 22/28 | 22/28 | 4, 4 | 2, 2 |

- **Noise:** Luna's two runs disagree on 4 of 28 pairs, Haiku's on none. About 2 points of Haiku's lead could be Luna's own variation.
- **Luna's misses:** it merges related but separate things. It joined Heather Rosenberg into her husband Evan's profile in one run, because both profiles mention her ombudsman job. It also merges sibling events of the Rosenberg custody story ("DCF taking legal custody" with "Conflict between…", "Rosenbergs refuse to reclaim…"). It doesn't count "New York" as New York City.
- **Haiku's misses:**
  - It merged "Ron DeSantis" with "Cliff Li" and with "Zhonggang Li" in both runs. But both pairs are broken: the evidence mined for "Ron DeSantis" describes Li, and DeSantis appears only as the governor Li campaigned for. Haiku judged the text, Luna the name. Those two pairs test the mining more than the model.
  - It kept "Miami Beach" apart from "Miami Beach", because one profile is about concerts on the beach and the other about a school soccer team.
- **Both fail:**
  - "Guantanamo Bay Navy Base" against "U.S. Navy base", whose label "different" is arguable.
  - "Evan Rosenberg" against "Evan D. Rosenberg", which both miss.
  - "Rosenbergs given time to retrieve son" against "Boy's stay in psychiatric treatment center".
- **The DeSantis pairs, fixed after the runs:**
  - **Why the evidence was wrong:** `mine_merge_pairs.py` copies each entity's saved profile. The February tables it mined (`data/guantanamo/entities`, from before stub profiles) had "Ron DeSantis" saved with a profile about Li, the article's main subject.
  - **Other bad profiles:** a check of every mined profile's first sentence found the same fault in "Houston" (about Cindy Yang at FIU), "Miami" (about the lawyer Alan Mishael), "Brown" (several Browns) and a few unlabelled pairs. Only the DeSantis pairs were unfair; in the others the wrong text makes "different" easier, not harder.
  - **Today's pipeline:** the same check found no profile about someone else in the 13 Nov full run, the 8 Nov Luna run or the 15 Nov pilot. A new entity's stub holds only the sentences that name it.
  - **The fix:** both pairs' DeSantis text is now his stub from the same article, the one sentence that names him (eval data, not in git; backup `merge_candidates_v2.jsonl.bak-before-desantis-stub`).
  - **Scored again:** Luna and Haiku both answer "different" on both pairs, twice each. So Haiku's corrected score on the cohort is 24/28. The Kitaru cohort still holds the old sessions; a `match-check-hard` v2 would need new baselines for those two pairs.
- **Verdict:** no change. Haiku leads, more so with the DeSantis pairs fixed (24 against 19-21), but 28 pairs and two runs are too few to switch the match model, and cost and speed weren't measured. The replay itself works: 112 replays, none failed.

### Where replays run

From a worktree of their own, `.worktrees/kitaru-replay` (branch `kitaru-replay`, tracking `origin/main`, with its own `.env` and `uv sync --extra kitaru --extra anthropic`), so a replay never runs whatever branch the main checkout is on. To use it:

1. Run `git pull` in that worktree to pick up new code.
2. Start the worker there: `just kitaru-worker --job-id <job> --claim agent --claim evaluator`.
3. Name its agent version in every replay: `kitaru replay create <session> --agent hinbox-profile-update@3 …`.

The agent versions:

| Version | `working_dir` | Use |
|---|---|---|
| 1 | the PR worktree the trial ran from | the 5 baselines are bound to it, so a replay without `--agent` runs it and fails once that worktree is gone |
| 2 | the main checkout | don't: it is usually on another branch |
| 3 | `.worktrees/kitaru-replay` | the one to use; an unchanged replay on it passed every check on 14 Nov |
| 4 | `.worktrees/kitaru-prompt-disputes` | the prompt experiment above; only for comparing against it |
| 5 | `.worktrees/kitaru-replay`, `HINBOX_EDIT_DATE_CHECK=on` | the date check after the edits |

Register a new version when a change should be replayed against older baselines, such as a prompt edit in a worktree of its own. Match checks have their own agent, `hinbox-match-check`: v1 is `.worktrees/kitaru-replay`. Start its worker with `--claim agent=<v1's version id>`.

### Still open

- **Haiku and Khadr's case 20:** find why Haiku fails it in every run while Luna passes.
- **More baselines.** Replays need step sessions: a few more eval cases, or a short pipeline run's `--step profile.update` sessions. Keep it to a small sample: they hold article text.
- **The edit date check on local models.** Run the same check with local Qwen on beast (back on from Monday 16 Nov): `HINBOX_EDIT_DATE_CHECK=on just eval profile-update --model ollama/<qwen> --split <dev|train|test>` against `=off`, then count wrong "Another report differs" sentences. Earlier notes have Qwen calling pairs about two different things conflicts. If it holds up, make `on` the default for local models too.
- **Pages with historian corrections.** A pipeline update runs inside `correction_scope()`, where `guard_corrections()` keeps struck sentences out and the historian's own sentences unchanged. A replay has no such scope, so `update_profile()` now records `guarded_corrections` (how many struck and written sentences were guarded) and the replay refuses a step where it is above 0. Restoring the rules (recording the active correction lines and putting them back in scope) would let those pages replay too. Sessions recorded before this field existed can't be checked.
- **Match checks on local Qwen:** fork `match-check-hard` onto an `ollama/` model once beast is back. The replay asks it locally, since `backend` follows the model.

### How replay changes map onto Hinbox

| Kitaru override | Fit with Hinbox |
|---|---|
| `model` | Easy. `update_profile(model=…)` exists, as do `HINBOX_<TASK>_MODEL` variables. Kitaru writes models as `provider:name`; Hinbox uses `openai/gpt-6-luna` and `ollama/…`, so translate between them. |
| `model_params` (temperature etc.) | Done: `set_model_params()` puts them on top of `_sampling_kwargs()` for every call. A key the model refuses (`temperature` on a Claude 5-family model) fails the call. |
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
