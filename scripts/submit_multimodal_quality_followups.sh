#!/usr/bin/env bash
set -euo pipefail

# Pre-sealed development follow-ups for the weakest corrected E3 metric.
STAMP="${STAMP:-260710a}"
ONLY=",${ONLY:-speech3rank49,speech3rank19,hard19,balanced19,speechlight49},"
DATA_DIR="${DATA_DIR:-data/real_subset_clean_260708b}"
GPU="${GPU:-1}"
CPU="${CPU:-10}"
MEMORY="${MEMORY:-110G}"
SEED="${SEED:-42}"

case "$DATA_DIR" in
  *sealed_eval*)
    echo "refusing to train on sealed evaluation data: $DATA_DIR" >&2
    exit 2
    ;;
esac

submit_candidate() {
  local label="$1"
  local modality_cycle="$2"
  local rank_negatives="$3"
  local negative_mode="$4"
  local image_rank_coef="$5"
  local speech_rank_coef="$6"
  local speech_contrastive_coef="$7"
  local job_name="sme-follow-${label}-${STAMP}"
  local output_dir="outputs/review_repair/followup_${label}_seed${SEED}"

  if [[ "$ONLY" != *",${label},"* ]]; then
    return 0
  fi
  if [ -e "$output_dir" ]; then
    echo "refusing to overwrite existing output: $output_dir" >&2
    return 1
  fi

  echo "Submitting ${job_name}"
  env \
    JOB_NAME="$job_name" OUT="$output_dir" DATA_DIR="$DATA_DIR" \
    GPU="$GPU" CPU="$CPU" MEMORY="$MEMORY" SEED="$SEED" \
    FINAL_STEPS=6000 ABLATION_STEPS=0 CAPACITY_ABLATION_STEPS=0 EXPERT_ABLATION_STEPS=0 \
    POSTPROCESS_REQUIRED_RUNS=0 \
    CAPACITY_FACTOR=8.0 AUX_COEF=0.02 \
    TRAIN_BATCH_SIZE=4 EVAL_BATCH_SIZE=8 \
    MODALITY_CYCLE="$modality_cycle" \
    TEXT_EVAL_BLOCKS=160 RETRIEVAL_EVAL_SAMPLES=250 CONDITIONAL_EVAL_SAMPLES=250 \
    IMAGE_EVAL_SAMPLES=250 SPEECH_EVAL_SAMPLES=250 \
    IMAGE_PREFIX_TOKENS=50 AUDIO_PREFIX_TOKENS=64 ENCODER_FEATURE_TOKENS=100 \
    TRAIN_ROUTER_GATES=0 TRAIN_EXPERTS=0 TRAIN_LM_HEAD=1 \
    LEARNING_RATE=0.0005 RETRIEVAL_HEAD_LEARNING_RATE=0.0015 \
    ROUTER_LEARNING_RATE=0.0001 LM_HEAD_LEARNING_RATE=0.00001 \
    WEIGHT_DECAY=0.0 GRAD_CLIP=5.0 \
    CONTRASTIVE_COEF=0.15 IMAGE_CONTRASTIVE_COEF=0.4 \
    SPEECH_CONTRASTIVE_COEF="$speech_contrastive_coef" \
    IMAGE_CONTRASTIVE_NEGATIVES=-1 SPEECH_CONTRASTIVE_NEGATIVES=-1 \
    CONDITIONAL_NEGATIVES=9 CONDITIONAL_RANKING_NEGATIVES="$rank_negatives" \
    CONDITIONAL_RANKING_NEGATIVE_MODE="$negative_mode" \
    CONDITIONAL_RANKING_HARD_POOL_SIZE=512 CONDITIONAL_RANKING_TEMPERATURE=0.7 \
    IMAGE_CONDITIONAL_RANKING_COEF="$image_rank_coef" \
    SPEECH_CONDITIONAL_RANKING_COEF="$speech_rank_coef" \
    LOG_EVERY_STEPS=150 SAVE_EVERY_STEPS=750 \
    REAL_MAX_SOURCE_AUDIO_SECONDS=6 REAL_MAX_TRANSCRIPT_WORDS=18 \
    CAPTION_MIN_ASCII_RATIO=0.85 CAPTION_MIN_LETTERS=8 \
    bash scripts/submit_e3_candidate_runai.sh
}

submit_candidate speech3rank49 text,image,image,speech,speech,speech 49 random 1.0 4.0 0.8
submit_candidate speech3rank19 text,image,image,speech,speech,speech 19 random 1.0 3.5 0.8
submit_candidate hard19 text,image,image,speech,speech 19 hard_text 1.5 3.0 0.6
submit_candidate balanced19 text,image,image,speech,speech 19 random 1.0 3.5 0.8
submit_candidate speechlight49 text,image,image,speech,speech 49 random 1.0 3.0 0.7
