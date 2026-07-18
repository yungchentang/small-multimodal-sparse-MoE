#!/usr/bin/env bash
set -euo pipefail

# Corrected Top-2-aux development sweep. This never reads the sealed test split.
# Override ONLY with a comma-separated subset of: cap7a02,cap7a01,cap7a04,cap8a02,cap6a02.

STAMP="${STAMP:-260709fix}"
ONLY=",${ONLY:-cap7a02,cap7a01,cap7a04,cap8a02,cap6a02},"
DATA_DIR="${DATA_DIR:-data/real_subset_clean_260708b}"
GPU="${GPU:-1}"
CPU="${CPU:-10}"
MEMORY="${MEMORY:-110G}"
SEED="${SEED:-42}"
OUTPUT_TAG="${OUTPUT_TAG:-}"

if [[ ! "$OUTPUT_TAG" =~ ^[A-Za-z0-9_-]*$ ]]; then
  echo "OUTPUT_TAG must contain only letters, digits, underscores, or hyphens" >&2
  exit 2
fi

case "$DATA_DIR" in
  *sealed_eval*)
    echo "refusing to train on sealed evaluation data: $DATA_DIR" >&2
    exit 2
    ;;
esac

submit_candidate() {
  local label="$1"
  local capacity="$2"
  local aux_coef="$3"
  local job_name="sme-corr-${label}-${STAMP}"
  local output_dir="outputs/review_repair/corrected_${label}_seed${SEED}${OUTPUT_TAG}"

  if [[ "$ONLY" != *",${label},"* ]]; then
    echo "Skipping ${label} due to ONLY=${ONLY}"
    return 0
  fi
  if [ -e "$output_dir" ]; then
    echo "refusing to overwrite existing output: $output_dir" >&2
    return 1
  fi

  echo "Submitting ${job_name}: capacity=${capacity} aux=${aux_coef}"
  env \
    JOB_NAME="$job_name" OUT="$output_dir" DATA_DIR="$DATA_DIR" \
    GPU="$GPU" CPU="$CPU" MEMORY="$MEMORY" SEED="$SEED" \
    FINAL_STEPS=6000 ABLATION_STEPS=0 CAPACITY_ABLATION_STEPS=0 EXPERT_ABLATION_STEPS=0 \
    POSTPROCESS_REQUIRED_RUNS=0 \
    CAPACITY_FACTOR="$capacity" AUX_COEF="$aux_coef" \
    TRAIN_BATCH_SIZE=4 EVAL_BATCH_SIZE=8 \
    MODALITY_CYCLE=text,image,image,speech,speech \
    TEXT_EVAL_BLOCKS=160 RETRIEVAL_EVAL_SAMPLES=250 CONDITIONAL_EVAL_SAMPLES=250 \
    IMAGE_EVAL_SAMPLES=250 SPEECH_EVAL_SAMPLES=250 \
    IMAGE_PREFIX_TOKENS=50 AUDIO_PREFIX_TOKENS=64 ENCODER_FEATURE_TOKENS=100 \
    TRAIN_ROUTER_GATES=0 TRAIN_EXPERTS=0 TRAIN_LM_HEAD=1 \
    LEARNING_RATE=0.0005 RETRIEVAL_HEAD_LEARNING_RATE=0.0015 \
    ROUTER_LEARNING_RATE=0.0001 LM_HEAD_LEARNING_RATE=0.00001 \
    WEIGHT_DECAY=0.0 GRAD_CLIP=5.0 \
    CONTRASTIVE_COEF=0.15 IMAGE_CONTRASTIVE_COEF=0.4 SPEECH_CONTRASTIVE_COEF=0.6 \
    IMAGE_CONTRASTIVE_NEGATIVES=-1 SPEECH_CONTRASTIVE_NEGATIVES=-1 \
    CONDITIONAL_NEGATIVES=9 CONDITIONAL_RANKING_NEGATIVES=49 \
    CONDITIONAL_RANKING_NEGATIVE_MODE=random CONDITIONAL_RANKING_TEMPERATURE=0.7 \
    IMAGE_CONDITIONAL_RANKING_COEF=1.0 SPEECH_CONDITIONAL_RANKING_COEF=2.5 \
    LOG_EVERY_STEPS=150 SAVE_EVERY_STEPS=750 \
    REAL_MAX_SOURCE_AUDIO_SECONDS=6 REAL_MAX_TRANSCRIPT_WORDS=18 \
    CAPTION_MIN_ASCII_RATIO=0.85 CAPTION_MIN_LETTERS=8 \
    bash scripts/submit_e3_candidate_runai.sh
}

submit_candidate cap7a02 7.0 0.02
submit_candidate cap7a01 7.0 0.01
submit_candidate cap7a04 7.0 0.04
submit_candidate cap8a02 8.0 0.02
submit_candidate cap6a02 6.0 0.02
