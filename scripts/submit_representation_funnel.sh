#!/usr/bin/env bash
set -euo pipefail

# Run the frozen, development-fitted representation-retention diagnostic once.
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
PROTOCOL_PATH="${PROTOCOL_PATH:?PROTOCOL_PATH is required}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$RUN_ROOT/E3_final_multimodal_top2/checkpoint_final.pt}"
GAMMA_JSON="${GAMMA_JSON:-$RUN_ROOT/calibration/gamma.json}"
DEV_IMAGE_MANIFEST="${DEV_IMAGE_MANIFEST:-outputs/review_repair/development_eval_v1/image_val.jsonl}"
DEV_SPEECH_MANIFEST="${DEV_SPEECH_MANIFEST:-outputs/review_repair/development_eval_v1/speech_val.jsonl}"
IMAGE_MANIFEST="${IMAGE_MANIFEST:-data/sealed_eval_v1/image_test.jsonl}"
SPEECH_MANIFEST="${SPEECH_MANIFEST:-data/sealed_eval_v1/speech_test.jsonl}"
OUT="${OUT:-outputs/review_repair/representation_funnel}"
JOB_NAME="${JOB_NAME:-sme-repfunnel-260709}"
PROJECT="${PROJECT:?PROJECT is required; configure it in .env.runai or the environment}"

for required in \
  "$CHECKPOINT_PATH" "$GAMMA_JSON" "$PROTOCOL_PATH" \
  "$DEV_IMAGE_MANIFEST" "$DEV_SPEECH_MANIFEST" \
  "$IMAGE_MANIFEST" "$SPEECH_MANIFEST"; do
  if [ ! -s "$required" ]; then
    echo "missing required representation-funnel input: $required" >&2
    exit 2
  fi
done
if [ -e "$OUT" ]; then
  echo "refusing to overwrite representation-funnel output: $OUT" >&2
  exit 2
fi
if [ "${#JOB_NAME}" -gt 55 ]; then
  echo "Run:AI job name exceeds 55 characters: $JOB_NAME" >&2
  exit 2
fi

python scripts/freeze_evaluation_protocol.py --verify "$PROTOCOL_PATH" --verify-git-state

env \
  PROJECT="$PROJECT" JOB_NAME="$JOB_NAME" \
  MODE=representation-funnel GPU="${GPU:-1}" CPU="${CPU:-8}" MEMORY="${MEMORY:-120G}" \
  CHECKPOINT="$CHECKPOINT_PATH" GAMMA_JSON="$GAMMA_JSON" \
  PROTOCOL_PATH="$PROTOCOL_PATH" OUT="$OUT" \
  DEV_IMAGE_MANIFEST="$DEV_IMAGE_MANIFEST" DEV_SPEECH_MANIFEST="$DEV_SPEECH_MANIFEST" \
  IMAGE_MANIFEST="$IMAGE_MANIFEST" SPEECH_MANIFEST="$SPEECH_MANIFEST" \
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}" \
  bash scripts/submit_runai.sh
