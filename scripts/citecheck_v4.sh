#!/usr/bin/env bash
# Free v4 of the tuned citation checker, as one chain to run detached:
#
#   A=<articles parquet> nohup scripts/citecheck_v4.sh >/dev/null 2>&1 &
#
# 1. `citecheck hard`: the claims of v3's training examples that untuned
#    Inkling put on the wrong side of yes/no in every one of its 4 tries,
#    each as an example of its own.
# 2. `distill` those with 16 tries at temperature 1.0 (Inkling's own
#    reasoning; an answer that lands on the label is kept).
# 3. Optional, off unless LUNA=on: score GPT-6 Luna on the 400 test claims,
#    then let it answer the hard claims still never kept (`citecheck
#    reteach`), stopping first if the estimate is over LUNA_CAP dollars
#    (needs LUNA_PRICE_IN and LUNA_PRICE_OUT, $ per million tokens).
# 4. `citecheck mix`: v3's kept rows (less those holding only planted year
#    and number claims, which it already gets right) plus the new kept rows
#    4 times, and the test files.
# 5. The ZenML pipeline, then the score into RESULT.md.
#
# Everything goes to $OUT (default data/citecheck/distill_v4): run.log, and
# FAILED.txt naming the step if one fails. Re-running resumes: distill and the
# pipeline's data skip what is already there. Files hold article text: data/ only.
set -euo pipefail
cd "$(dirname "$0")/.."

D=${D:-data/citecheck}
EXPORT=${EXPORT:-$D/export_v3}       # v3's export: train.jsonl, tune.jsonl
V3=${V3:-$D/distill_v3}              # v3's kept rows, distill logs, test files
PLANTED=${PLANTED:-$D/teach_strict/planted.jsonl}
OUT=${OUT:-$D/distill_v4}
NAME=${NAME:-citecheck-strict-v4-free}
A=${A:?set A to the articles parquet}
LUNA=${LUNA:-off}
LUNA_MODEL=${LUNA_MODEL:-openai/gpt-6-luna}

mkdir -p "$OUT"
exec >>"$OUT/run.log" 2>&1
step=start
trap 'echo "step $step failed (line $LINENO): $BASH_COMMAND" | tee "$OUT/FAILED.txt"' ERR
rm -f "$OUT/FAILED.txt"
say() { echo "[$(date -u +%H:%M:%SZ)] $*"; }

set -a; source .env; set +a
export HINBOX_GROUNDING_RULES=strict

step=check
for f in "$EXPORT/train.jsonl" "$EXPORT/tune.jsonl" "$V3/train.jsonl" "$V3/tune.jsonl" \
  "$V3/train.jsonl.log.jsonl" "$V3/tune.jsonl.log.jsonl" "$V3/test_items.jsonl" \
  "$V3/test_labels.jsonl" "$V3/test_prompts.jsonl" "$PLANTED" "$A"; do
  [[ -e "$f" ]] || { echo "missing $f"; false; }
done
say "inputs found; LUNA=$LUNA"

step=hard
uv run python -m src.citecheck hard --export "$EXPORT" --distilled "$V3" \
  --planted "$PLANTED" --out "$OUT"

for split in train tune; do
  step="distill $split"
  if [[ -s "$OUT/hard_$split.jsonl" ]]; then
    say "distilling hard_$split (16 tries each)"
    uv run --script scripts/tinker_citecheck.py distill --data "$OUT/hard_$split.jsonl" \
      --out "$OUT/hard_$split.distilled.jsonl" --samples 16 --temperature 1.0 --loose \
      --max-tokens 8192 --workers 8
  else
    say "no hard claims in $split: nothing to distill"
  fi
  # distill writes nothing when no answer is kept (or no claim is hard)
  touch "$OUT/hard_$split.distilled.jsonl" "$OUT/hard_$split.distilled.jsonl.log.jsonl"
done

extra=("$OUT/hard_train.distilled.jsonl" "$OUT/hard_tune.distilled.jsonl")
if [[ "$LUNA" == on ]]; then
  : "${LUNA_PRICE_IN:?set LUNA_PRICE_IN}" "${LUNA_PRICE_OUT:?set LUNA_PRICE_OUT}"
  step="luna score"
  [[ -s "$OUT/luna_score.json" ]] || uv run python -m src.citecheck score \
    --items "$V3/test_items.jsonl" --labels "$V3/test_labels.jsonl" --articles "$A" \
    --model "$LUNA_MODEL" --workers 4 --save "$OUT/luna_score.json"
  for split in train tune; do
    step="luna $split"
    [[ -e "$OUT/hard_$split.luna.jsonl" ]] || uv run python -m src.citecheck reteach \
      --prompts "$OUT/hard_$split.jsonl" --log "$OUT/hard_$split.distilled.jsonl.log.jsonl" \
      --teacher "$LUNA_MODEL" --out "$OUT/hard_$split.luna.jsonl" \
      --max-dollars "${LUNA_CAP:-5}" --price-in "$LUNA_PRICE_IN" --price-out "$LUNA_PRICE_OUT"
    extra+=("$OUT/hard_$split.luna.jsonl")
  done
fi

step=mix
uv run python -m src.citecheck mix --base "$V3" --extra "${extra[@]}" --repeat 4 \
  --planted "$PLANTED" --drop-kinds planted:year,planted:number --out "$OUT"

step=pipeline
say "training $NAME in ZenML"
just tinker-pipeline --data-dir "$OUT" --articles "$A" --name "$NAME" --epochs 1 \
  --max-tokens 8192 ${BASELINES:+--baselines "$BASELINES"}

step=score
replies=$(ls -t "$OUT"/test_replies."$NAME".*.jsonl | head -1)
{
  echo "# $NAME"
  echo
  echo '```'
  cat "$OUT/hard_summary.json" "$OUT/mix_summary.json"
  uv run python -m src.citecheck score --items "$OUT/test_items.jsonl" \
    --labels "$OUT/test_labels.jsonl" --replies "$replies"
  echo '```'
} >"$OUT/RESULT.md"
say "done: $OUT/RESULT.md"
