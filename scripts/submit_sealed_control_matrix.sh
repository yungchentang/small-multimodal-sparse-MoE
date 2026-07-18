#!/usr/bin/env bash
set -euo pipefail

source scripts/sealed_evaluation_defaults.sh

# Submit the frozen sealed conditional-matching matrix. Training/model selection
# must be complete before this script is used.

RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$RUN_ROOT/E3_final_multimodal_top2/checkpoint_final.pt}"
PROTOCOL_PATH="${PROTOCOL_PATH:?PROTOCOL_PATH is required}"
BASE_OUT="${BASE_OUT:-outputs/review_repair/sealed_controls}"
JOB_STAMP="${JOB_STAMP:-260709}"
PROJECT="${PROJECT:?PROJECT is required; configure it in .env.runai or the environment}"
STAGE_B_CHECKPOINT="${STAGE_B_CHECKPOINT:?verified Stage-B checkpoint is required}"
STAGE_B_CHECKPOINT_SHA256="${STAGE_B_CHECKPOINT_SHA256:?verified Stage-B checkpoint SHA256 is required}"
if [[ ! "$STAGE_B_CHECKPOINT_SHA256" =~ ^[0-9a-fA-F]{64}$ ]]; then
  echo "Stage-B checkpoint SHA256 must be an exact 64-character digest" >&2
  exit 2
fi
GPU="${GPU:-1}"
CPU="${CPU:-8}"
MEMORY="${MEMORY:-100G}"
CAPACITY_FACTOR="${CAPACITY_FACTOR:?CAPACITY_FACTOR is required}"
AUX_COEF="${AUX_COEF:?AUX_COEF is required}"
CANDIDATE_SEED="${CANDIDATE_SEED:-314159}"
CONTROL_SEED="${CONTROL_SEED:-42}"
ONLY_CELLS=",${ONLY_CELLS:-r5,r10,h10,f250},"
ONLY_CONTROLS=",${ONLY_CONTROLS:-real,shuffled,zero,random,no-prefix},"

IMAGE_MANIFEST="${IMAGE_MANIFEST:-data/sealed_eval_v1/image_test.jsonl}"
SPEECH_MANIFEST="${SPEECH_MANIFEST:-data/sealed_eval_v1/speech_test.jsonl}"
DATA_DIR="${DATA_DIR:-$(dirname -- "$IMAGE_MANIFEST")}"

if [ "$(dirname -- "$IMAGE_MANIFEST")" != "$DATA_DIR" ] || [ "$(dirname -- "$SPEECH_MANIFEST")" != "$DATA_DIR" ]; then
  echo "image and speech manifests must belong to DATA_DIR: $DATA_DIR" >&2
  exit 2
fi

for required in "$CHECKPOINT_PATH" "$PROTOCOL_PATH" "$IMAGE_MANIFEST" "$SPEECH_MANIFEST"; do
  if [ ! -s "$required" ]; then
    echo "missing required sealed-evaluation input: $required" >&2
    exit 2
  fi
done
python scripts/freeze_evaluation_protocol.py --verify "$PROTOCOL_PATH" --verify-git-state

submit_condition() {
  local cell="$1" candidates="$2" negatives="$3" negative_mode="$4"
  local control="$5" eval_path="$6" prefix_control="$7"
  local label="${cell}-${control}"
  local job_name="sme-sv1-${label}-${JOB_STAMP}"
  local output_dir="${BASE_OUT}/${label}"

  if [[ "$ONLY_CELLS" != *",${cell},"* ]] || [[ "$ONLY_CONTROLS" != *",${control},"* ]]; then
    return 0
  fi
  if [ -e "$output_dir" ]; then
    echo "refusing to overwrite sealed output: $output_dir" >&2
    return 1
  fi
  if [ "${#job_name}" -gt 55 ]; then
    echo "Run:AI job name exceeds 55 characters: $job_name" >&2
    return 1
  fi

  env \
    PROJECT="$PROJECT" JOB_NAME="$job_name" MODE=conditional-eval \
    GPU="$GPU" CPU="$CPU" MEMORY="$MEMORY" \
    OUT="${output_dir}/metrics.json" PER_QUERY_OUTPUT="${output_dir}/per_query.jsonl" \
    FEATURE_CACHE_DIR="${output_dir}/feature_cache" \
    RUN_OUTPUT_DIR="$RUN_ROOT" CHECKPOINT="$CHECKPOINT_PATH" \
    STAGE_B_CHECKPOINT="$STAGE_B_CHECKPOINT" STAGE_B_CHECKPOINT_SHA256="$STAGE_B_CHECKPOINT_SHA256" \
    DATA_DIR="$DATA_DIR" IMAGE_MANIFEST="$IMAGE_MANIFEST" SPEECH_MANIFEST="$SPEECH_MANIFEST" \
    PROTOCOL_PATH="$PROTOCOL_PATH" PROTOCOL_NAME=sealed_evaluation_v1 \
    EVAL_SPLIT_NAME=sealed_test RANDOMIZE_POSITIVE_POSITION=1 \
    EVALUATION_SCOPE=final \
    CANDIDATE_SEED="$CANDIDATE_SEED" CONTROL_SEED="$CONTROL_SEED" \
    CAPACITY_FACTOR="$CAPACITY_FACTOR" AUX_COEF="$AUX_COEF" \
    IMAGE_PREFIX_TOKENS=50 AUDIO_PREFIX_TOKENS=64 ENCODER_FEATURE_TOKENS=100 \
    IMAGE_EVAL_SAMPLES=250 SPEECH_EVAL_SAMPLES=250 CONDITIONAL_QUERIES=250 \
    CONDITIONAL_CANDIDATES="$candidates" CONDITIONAL_NEGATIVES="$negatives" \
    CONDITIONAL_BATCH_SIZE="$CONDITIONAL_BATCH_SIZE" NEGATIVE_MODE="$negative_mode" \
    EVAL_PATH="$eval_path" PREFIX_CONTROL="$prefix_control" \
    QUERY_OFFSET=0 CANDIDATE_OFFSET=0 BOOTSTRAP_SAMPLES=2000 BOOTSTRAP_SEED=12345 \
    bash scripts/submit_runai.sh
}

submit_cell() {
  local cell="$1" candidates="$2" negatives="$3" negative_mode="$4"
  submit_condition "$cell" "$candidates" "$negatives" "$negative_mode" real shared_prefix real
  submit_condition "$cell" "$candidates" "$negatives" "$negative_mode" shuffled shared_prefix shuffled
  submit_condition "$cell" "$candidates" "$negatives" "$negative_mode" zero shared_prefix zero
  submit_condition "$cell" "$candidates" "$negatives" "$negative_mode" random shared_prefix random
  submit_condition "$cell" "$candidates" "$negatives" "$negative_mode" no-prefix no_prefix_lm real
}

submit_cell r5 5 4 random
submit_cell r10 10 9 random
submit_cell h10 10 9 hard_text
submit_cell f250 250 -1 random
