#!/usr/bin/env bash
set -euo pipefail

# Aggregate first-wave auto-research results after Run:AI jobs complete.
# Missing roots are logged instead of being treated as successful evidence.

STAMP="${STAMP:-260709next}"
OUT_DIR="${OUT_DIR:-autoresearch/improve-260709-0836/results}"
mkdir -p "$OUT_DIR"
MISSING="$OUT_DIR/missing_outputs.txt"
: > "$MISSING"

collect_existing() {
  local arr_name="$1"
  shift
  local existing=()
  for path in "$@"; do
    if [ -e "$path" ]; then
      existing+=("$path")
    else
      echo "$arr_name missing: $path" >> "$MISSING"
    fi
  done
  printf '%s\n' "${existing[@]}"
}

mapfile -t E3_ROOTS < <(collect_existing e3 \
  outputs/sparse-moe-clean-final-selected-img15-260709a \
  "outputs/sparse-moe-mm-rank49-fullbank-cap7-${STAMP}" \
  "outputs/sparse-moe-mm-prefix75-balanced-cap8-${STAMP}" \
  "outputs/sparse-moe-mm-whisper-space-cap6-${STAMP}" \
  "outputs/sparse-moe-route-z3e5-cap7-img15-${STAMP}" \
  "outputs/sparse-moe-route-drop02-cap7-img15-${STAMP}" \
  "outputs/sparse-moe-route-dynbias-cap7-img15-${STAMP}" \
  "outputs/sparse-moe-mm-hardtext-rank19-cap7-${STAMP}" \
  "outputs/sparse-moe-mm-routerlite-cap7-img15-${STAMP}" \
  "outputs/sparse-moe-mm-frozenrouter-cap7-img15-${STAMP}")

if [ "${#E3_ROOTS[@]}" -gt 0 ]; then
  BASE_CANDIDATE_CSV="${BASE_CANDIDATE_CSV:-outputs/sparse-moe-mm-rank49-fullbank-cap7-260709next/experiment_loop/candidate_comparison.csv}"
  BASE_CANDIDATE_ARGS=()
  if [ -f "$BASE_CANDIDATE_CSV" ]; then
    BASE_CANDIDATE_ARGS=(--root-csv "$BASE_CANDIDATE_CSV")
  fi
  python scripts/compare_e3_candidates.py "${BASE_CANDIDATE_ARGS[@]}" "${E3_ROOTS[@]}" --output-dir "$OUT_DIR/e3_candidates"
fi

mapfile -t DISTILL_ROOTS < <(collect_existing top2_distill \
  outputs/sparse-moe-top2-distill-kl005-260709e \
  outputs/sparse-moe-top2-distill-kl005-router-260709f \
  "outputs/sparse-moe-top2-distill-hidden-cos-${STAMP}" \
  "outputs/sparse-moe-top2-distill-longce-${STAMP}" \
  "outputs/sparse-moe-top2-distill-gammatrain-${STAMP}")

if [ "${#DISTILL_ROOTS[@]}" -gt 0 ]; then
  python scripts/compare_top2_distillation.py "${DISTILL_ROOTS[@]}" --output-dir "$OUT_DIR/top2_distillation"
fi

EVAL_ROOT="${EVAL_ROOT:-outputs/eval-controls-rank49-260709}"
if [ -d "$EVAL_ROOT" ]; then
  python scripts/collect_eval_controls.py \
    --root "$EVAL_ROOT" \
    --output-csv "$OUT_DIR/heldout_eval_controls.csv" \
    --output-md "$OUT_DIR/heldout_eval_controls.md"
  python scripts/summarize_prefix_controls.py \
    --input-csv "$OUT_DIR/heldout_eval_controls.csv" \
    --output-json "$OUT_DIR/heldout_prefix_sensitivity.json" \
    --output-md "$OUT_DIR/heldout_prefix_sensitivity.md"
else
  echo "eval_controls missing: $EVAL_ROOT" >> "$MISSING"
fi

echo "Wrote analysis under $OUT_DIR"
echo "Missing-output log: $MISSING"
