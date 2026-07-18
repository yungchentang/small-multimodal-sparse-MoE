#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
PROJECT="${PROJECT:?PROJECT is required}"
EXPECTED_CHECKPOINT_SHA256="${EXPECTED_CHECKPOINT_SHA256:?EXPECTED_CHECKPOINT_SHA256 is required}"
OUT="${OUT:?OUT is required}"
CHECKPOINT="${CHECKPOINT:-$RUN_ROOT/E3_final_multimodal_top2/checkpoint_final.pt}"
JOB_NAME="${JOB_NAME:-sme-final-e6-feasibility}"

for required in "$RUN_ROOT/manifest.json" "$CHECKPOINT"; do
  if [ ! -s "$required" ]; then
    echo "missing E6 input: $required" >&2
    exit 2
  fi
done
if [ -e "$OUT" ]; then
  echo "refusing to overwrite E6 output: $OUT" >&2
  exit 2
fi

actual_sha256="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
if [ "$actual_sha256" != "$EXPECTED_CHECKPOINT_SHA256" ]; then
  echo "selected checkpoint SHA-256 mismatch" >&2
  exit 2
fi

env \
  PROJECT="$PROJECT" JOB_NAME="$JOB_NAME" MODE=e6-feasibility \
  GPU="${GPU:-1}" CPU="${CPU:-10}" MEMORY="${MEMORY:-120G}" \
  RUN_OUTPUT_DIR="$RUN_ROOT" CHECKPOINT="$CHECKPOINT" OUT="$OUT" \
  EXPECTED_CHECKPOINT_SHA256="$EXPECTED_CHECKPOINT_SHA256" \
  DATA_DIR="${DATA_DIR:-data/real_subset_clean_260708b}" \
  FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR:-$RUN_ROOT/feature_cache}" \
  bash scripts/submit_runai.sh
