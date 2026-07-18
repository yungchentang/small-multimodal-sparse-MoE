#!/usr/bin/env bash
set -euo pipefail

PROTOCOL_PATH="${PROTOCOL_PATH:?PROTOCOL_PATH is required}"
MATRIX_ROOT="${MATRIX_ROOT:-outputs/review_repair/sealed_controls}"
OUT="${OUT:-outputs/review_repair/sealed_matrix_analysis}"
JOB_NAME="${JOB_NAME:-sme-sealed-analysis-260709}"
PROJECT="${PROJECT:?PROJECT is required; configure it in .env.runai or the environment}"
CONTROLS=(real shuffled zero random no-prefix)
CELLS=(r5 r10 h10 f250)

if [ ! -s "$PROTOCOL_PATH" ]; then
  echo "missing frozen protocol: $PROTOCOL_PATH" >&2
  exit 2
fi
if [ -e "$OUT" ]; then
  echo "refusing to overwrite sealed analysis: $OUT" >&2
  exit 2
fi
for cell in "${CELLS[@]}"; do
  for control in "${CONTROLS[@]}"; do
    for artifact in metrics.json per_query.jsonl; do
      path="${MATRIX_ROOT}/${cell}-${control}/${artifact}"
      if [ ! -s "$path" ]; then
        echo "sealed matrix is incomplete: $path" >&2
        exit 2
      fi
    done
  done
done
python scripts/freeze_evaluation_protocol.py --verify "$PROTOCOL_PATH"

env \
  PROJECT="$PROJECT" JOB_NAME="$JOB_NAME" \
  MODE=sealed-matrix-analysis GPU=0 CPU="${CPU:-8}" MEMORY="${MEMORY:-64G}" \
  PROTOCOL_PATH="$PROTOCOL_PATH" MATRIX_ROOT="$MATRIX_ROOT" OUT="$OUT" \
  BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-10000}" \
  PERMUTATION_SAMPLES="${PERMUTATION_SAMPLES:-10000}" SEED="${SEED:-20260709}" \
  bash scripts/submit_runai.sh
