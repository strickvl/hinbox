#!/usr/bin/env bash
# Free v6 of the tuned citation checker (hinted reasoning), as one chain to
# run detached:
#
#   A=<articles parquet> nohup scripts/citecheck_v6.sh >/dev/null 2>&1 &
#
# 1. `distill --hint` on v4's hard claims (the claims untuned Inkling got
#    wrong in every try) that v4's 16 tries never kept: the label is shown
#    in the sampling prompt as a reference verdict, an answer that lands on
#    it without mentioning it is kept, stored under the prompt without the
#    hint. A seeded pick of at most LIMIT_TRAIN / LIMIT_TUNE claims, so the
#    hard rows don't swamp v3's ordinary ones.
# 2. `citecheck mix`: v3's kept rows (less those holding only planted year
#    and number claims), v4's kept hard rows and the hinted rows, once each.
# 3. The ZenML pipeline (train, temperature-0 answers on the 400, score).
# 4. 7 answers per test prompt at temperature 0.7 from the new checkpoint,
#    then `citecheck votes` and `citecheck paired` against untuned and v4
#    (their 7-answer files), on all 400 and on the random half alone.
#
# A re-run reuses the checkpoint already saved under $NAME (no second
# training) and its own answer file, named after the checkpoint.
#
# Everything goes to $OUT (default data/citecheck/distill_v6): run.log,
# RESULT.md, and FAILED.txt naming the step if one fails. Re-running
# resumes. Files hold article text: data/ only. Free: Tinker's free model
# only, no cloud calls.
set -euo pipefail
cd "$(dirname "$0")/.."

D=${D:-data/citecheck}
V3=${V3:-$D/distill_v3}         # v3's kept rows and test files
V4=${V4:-$D/distill_v4}         # v4's hard claims and their distill logs
VOTES=${VOTES:-$D/votes}        # untuned_x7.jsonl, v4_x7.jsonl
PLANTED=${PLANTED:-$D/teach_strict/planted.jsonl}
OUT=${OUT:-$D/distill_v6}
NAME=${NAME:-citecheck-strict-v6-hint}
LIMIT_TRAIN=${LIMIT_TRAIN:-2000}
LIMIT_TUNE=${LIMIT_TUNE:-200}
A=${A:?set A to the articles parquet}

mkdir -p "$OUT"
exec >>"$OUT/run.log" 2>&1
step=start
trap 'echo "step $step failed (line $LINENO): $BASH_COMMAND" | tee "$OUT/FAILED.txt"' ERR
rm -f "$OUT/FAILED.txt"
say() { echo "[$(date -u +%H:%M:%SZ)] $*"; }

set -a; source .env; set +a
export HINBOX_GROUNDING_RULES=strict

step=check
for f in "$V3/train.jsonl" "$V3/tune.jsonl" "$V3/test_items.jsonl" "$V3/test_labels.jsonl" \
  "$V3/test_prompts.jsonl" "$V4/hard_train.jsonl" "$V4/hard_tune.jsonl" \
  "$V4/hard_train.distilled.jsonl" "$V4/hard_tune.distilled.jsonl" \
  "$V4/hard_train.distilled.jsonl.log.jsonl" "$V4/hard_tune.distilled.jsonl.log.jsonl" \
  "$VOTES/untuned_x7.jsonl" "$VOTES/v4_x7.jsonl" "$PLANTED" "$A"; do
  [[ -e "$f" ]] || { echo "missing $f"; false; }
done
say "inputs found"

for split in train tune; do
  step="hinted distill $split"
  limit=$LIMIT_TRAIN; [[ $split == tune ]] && limit=$LIMIT_TUNE
  say "hinted distill of hard_$split (at most $limit claims, 4 tries each)"
  uv run --script scripts/tinker_citecheck.py distill --data "$V4/hard_$split.jsonl" \
    --out "$OUT/hard_$split.hinted.jsonl" --hint \
    --skip-kept "$V4/hard_$split.distilled.jsonl.log.jsonl" --limit "$limit" \
    --samples 4 --temperature 0.7 --loose --max-tokens 8192 --workers 8
  touch "$OUT/hard_$split.hinted.jsonl"
done

step=mix
uv run python -m src.citecheck mix --base "$V3" \
  --extra "$V4/hard_train.distilled.jsonl" "$V4/hard_tune.distilled.jsonl" \
  "$OUT/hard_train.hinted.jsonl" "$OUT/hard_tune.hinted.jsonl" --repeat 1 \
  --planted "$PLANTED" --drop-kinds planted:year,planted:number --out "$OUT"

checkpoint() {  # the last checkpoint saved under $NAME, or nothing
  find "$OUT" -name checkpoints.jsonl -exec cat {} + 2>/dev/null | uv run python -c "
import json, sys
rows = [json.loads(l) for l in sys.stdin if l.strip()]
named = [r for r in rows if r['name'] == '$NAME']
print(named[-1]['sampler'] if named else '')"
}

step=pipeline
ck=$(checkpoint)
if [[ -z "$ck" ]]; then  # a resumed chain never trains a second checkpoint
  say "training $NAME in ZenML"
  just tinker-pipeline --data-dir "$OUT" --articles "$A" --name "$NAME" --epochs 1 \
    --max-tokens 8192
  ck=$(checkpoint)
fi
[[ -n "$ck" ]] || { echo "no checkpoint saved for $NAME"; false; }
say "checkpoint $ck"

step=votes
tag=$(printf %s "$ck" | shasum -a 256 | cut -c1-10)  # answer files are per checkpoint
votes="$OUT/v6_x7.$tag.jsonl"
uv run --script scripts/tinker_citecheck.py sample --checkpoint "$ck" \
  --prompts "$OUT/test_prompts.jsonl" --out "$votes" --votes 7 \
  --temperature 0.7 --max-tokens 8192 --workers 8

step="temperature 0"
# the chain's own temperature-0 answers, resumable even when an earlier run
# stopped inside the pipeline's sampling after the checkpoint was saved
t0="$OUT/v6_t0.$tag.jsonl"
uv run --script scripts/tinker_citecheck.py sample --checkpoint "$ck" \
  --prompts "$OUT/test_prompts.jsonl" --out "$t0" --max-tokens 8192

step=report
uv run python -c "
import json
rows = [json.loads(l) for l in open('$OUT/test_items.jsonl')]
with open('$OUT/test_items.random.jsonl', 'w') as f:
    f.writelines(json.dumps(r) + '\n' for r in rows if r.get('stratum') == 'random')"
{
  echo "# $NAME"
  echo
  echo '```'
  cat "$OUT/mix_summary.json"
  for split in train tune; do
    echo "hinted $split: $(wc -l <"$OUT/hard_$split.hinted.jsonl") kept"
  done
  echo
  echo "## temperature 0"
  uv run python -m src.citecheck score --items "$OUT/test_items.jsonl" \
    --labels "$OUT/test_labels.jsonl" --replies "$t0"
  for set in all random; do
    items="$OUT/test_items.jsonl"; [[ $set == random ]] && items="$OUT/test_items.random.jsonl"
    echo
    echo "## 7 answers, $set"
    for who in untuned:"$VOTES/untuned_x7.jsonl" v4:"$VOTES/v4_x7.jsonl" v6:"$votes"; do
      uv run python -m src.citecheck votes --items "$items" \
        --labels "$OUT/test_labels.jsonl" --replies "${who#*:}" | sed "s/^/${who%%:*} /"
    done
    uv run python -m src.citecheck paired --items "$items" --labels "$OUT/test_labels.jsonl" \
      --sources untuned="$VOTES/untuned_x7.jsonl" v4="$VOTES/v4_x7.jsonl" \
      v6="$votes" v6_t0="$t0"
  done
  echo '```'
} >"$OUT/RESULT.md"
say "done: $OUT/RESULT.md"
