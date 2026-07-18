#!/usr/bin/env bash
set -euo pipefail

# Submit one-GPU conditional-eval controls for an existing E3 checkpoint.
# Example:
#   RUN_ROOT=outputs/sparse-moe-mm-rank49-fullbank-cap7-260709next \
#   bash scripts/submit_eval_control_sweep.sh

STAMP="${STAMP:-$(date +%y%m%d%H%M)}"
RUN_ROOT="${RUN_ROOT:-outputs/sparse-moe-mm-rank49-fullbank-cap7-260709next}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${RUN_ROOT}/E3_final_multimodal_top2/checkpoint_final.pt}"
DATA_DIR="${DATA_DIR:-data/real_subset_clean_260708b}"
BASE_OUT="${BASE_OUT:-outputs/eval-controls-${STAMP}}"
GPU="${GPU:-1}"
CPU="${CPU:-8}"
MEMORY="${MEMORY:-100G}"
PROJECT="${PROJECT:?PROJECT is required; configure it in .env.runai or the environment}"
STAGE_B_CHECKPOINT="${STAGE_B_CHECKPOINT:?verified Stage-B checkpoint is required}"
STAGE_B_CHECKPOINT_SHA256="${STAGE_B_CHECKPOINT_SHA256:?verified Stage-B checkpoint SHA256 is required}"
if [[ ! "$STAGE_B_CHECKPOINT_SHA256" =~ ^[0-9a-fA-F]{64}$ ]]; then
  echo "Stage-B checkpoint SHA256 must be an exact 64-character digest" >&2
  exit 2
fi
JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-sme-eval}"
JOB_STAMP="${JOB_STAMP:-$STAMP}"
ONLY_LABELS="${ONLY_LABELS:-}"
FULL_250_QUERY_OFFSET="${FULL_250_QUERY_OFFSET:-125}"
FULL_250_QUERIES="${FULL_250_QUERIES:-125}"

mkdir -p "$BASE_OUT"

should_submit_label() {
  local label="$1"
  if [ -z "$ONLY_LABELS" ]; then
    return 0
  fi
  case ",$ONLY_LABELS," in
    *,"$label",*) return 0 ;;
    *) return 1 ;;
  esac
}

submit_eval() {
  local label="$1"
  local candidates="$2"
  local negatives="$3"
  local prefix_control="$4"
  local negative_mode="$5"
  local queries="${6:-${CONDITIONAL_QUERIES:-250}}"
  local eval_path="${7:-shared_prefix}"
  local query_offset="${8:-${QUERY_OFFSET:-0}}"
  local candidate_offset="${9:-${CANDIDATE_OFFSET:--1}}"
  if ! should_submit_label "$label"; then
    echo "Skipping ${label} due to ONLY_LABELS filter"
    return 0
  fi
  local job="${JOB_NAME_PREFIX}-${label}-${JOB_STAMP}"
  if [ "${#job}" -gt 55 ]; then
    echo "job name too long (${#job} > 55): ${job}" >&2
    return 1
  fi
  echo "Submitting ${job}: candidates=${candidates} negatives=${negatives} prefix=${prefix_control} negative_mode=${negative_mode} query_offset=${query_offset} candidate_offset=${candidate_offset}"
  JOB_NAME="$job" \
  MODE=conditional-eval \
  EVALUATION_SCOPE="${EVALUATION_SCOPE:?EVALUATION_SCOPE is required}" \
  OUT="${BASE_OUT}/${label}/metrics.json" \
  RUN_OUTPUT_DIR="$RUN_ROOT" \
  CHECKPOINT="$CHECKPOINT_PATH" \
  STAGE_B_CHECKPOINT="$STAGE_B_CHECKPOINT" \
  STAGE_B_CHECKPOINT_SHA256="$STAGE_B_CHECKPOINT_SHA256" \
  DATA_DIR="$DATA_DIR" \
  IMAGE_MANIFEST="${IMAGE_MANIFEST:-}" \
  SPEECH_MANIFEST="${SPEECH_MANIFEST:-}" \
  GPU="$GPU" CPU="$CPU" MEMORY="$MEMORY" PROJECT="$PROJECT" \
  CAPACITY_FACTOR="${CAPACITY_FACTOR:-6.0}" \
  AUX_COEF="${AUX_COEF:-0.01}" \
  IMAGE_PREFIX_TOKENS="${IMAGE_PREFIX_TOKENS:-50}" \
  AUDIO_PREFIX_TOKENS="${AUDIO_PREFIX_TOKENS:-64}" \
  ENCODER_FEATURE_TOKENS="${ENCODER_FEATURE_TOKENS:-100}" \
  IMAGE_EVAL_SAMPLES="${IMAGE_EVAL_SAMPLES:-250}" \
  SPEECH_EVAL_SAMPLES="${SPEECH_EVAL_SAMPLES:-250}" \
  CONDITIONAL_QUERIES="$queries" \
  CONDITIONAL_CANDIDATES="$candidates" \
  CONDITIONAL_NEGATIVES="$negatives" \
  CONDITIONAL_BATCH_SIZE="${CONDITIONAL_BATCH_SIZE:-16}" \
  PREFIX_CONTROL="$prefix_control" \
  NEGATIVE_MODE="$negative_mode" \
  EVAL_PATH="$eval_path" \
  CONTROL_SEED="${CONTROL_SEED:-42}" \
  EVAL_SPLIT_NAME="${EVAL_SPLIT_NAME:-eval_tail}" \
  QUERY_OFFSET="$query_offset" \
  CANDIDATE_OFFSET="$candidate_offset" \
  BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-1000}" \
  BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-12345}" \
  PER_QUERY_OUTPUT="${PER_QUERY_OUTPUT:-${BASE_OUT}/${label}/per_query.jsonl}" \
  bash scripts/submit_runai.sh
}

submit_encoder_baseline() {
  local label="$1"
  local candidates="$2"
  local negatives="$3"
  local negative_mode="$4"
  local queries="${5:-${CONDITIONAL_QUERIES:-250}}"
  local query_offset="${6:-${QUERY_OFFSET:-0}}"
  local candidate_offset="${7:-${CANDIDATE_OFFSET:--1}}"
  if ! should_submit_label "$label"; then
    echo "Skipping ${label} due to ONLY_LABELS filter"
    return 0
  fi
  local job="${JOB_NAME_PREFIX}-${label}-${JOB_STAMP}"
  if [ "${#job}" -gt 55 ]; then
    echo "job name too long (${#job} > 55): ${job}" >&2
    return 1
  fi
  echo "Submitting ${job}: encoder-only CLIP/Whisper baselines candidates=${candidates} negatives=${negatives} negative_mode=${negative_mode} query_offset=${query_offset} candidate_offset=${candidate_offset}"
  JOB_NAME="$job" \
  MODE=encoder-baseline-eval \
  OUT="${BASE_OUT}/${label}/metrics.json" \
  DATA_DIR="$DATA_DIR" \
  IMAGE_MANIFEST="${IMAGE_MANIFEST:-}" \
  SPEECH_MANIFEST="${SPEECH_MANIFEST:-}" \
  GPU="$GPU" CPU="$CPU" MEMORY="$MEMORY" PROJECT="$PROJECT" \
  VISION_MODEL="${VISION_MODEL:-openai/clip-vit-base-patch32}" \
  SPEECH_MODEL="${SPEECH_MODEL:-openai/whisper-base.en}" \
  FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR:-${BASE_OUT}/${label}/feature_cache}" \
  ENCODER_FEATURE_TOKENS="${ENCODER_FEATURE_TOKENS:-100}" \
  IMAGE_EVAL_SAMPLES="${IMAGE_EVAL_SAMPLES:-250}" \
  SPEECH_EVAL_SAMPLES="${SPEECH_EVAL_SAMPLES:-250}" \
  CONDITIONAL_QUERIES="$queries" \
  CONDITIONAL_CANDIDATES="$candidates" \
  CONDITIONAL_NEGATIVES="$negatives" \
  CONDITIONAL_BATCH_SIZE="${CONDITIONAL_BATCH_SIZE:-16}" \
  NEGATIVE_MODE="$negative_mode" \
  EVAL_SPLIT_NAME="${EVAL_SPLIT_NAME:-eval_tail}" \
  QUERY_OFFSET="$query_offset" \
  CANDIDATE_OFFSET="$candidate_offset" \
  BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-1000}" \
  BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-12345}" \
  PER_QUERY_OUTPUT="${BASE_OUT}/${label}/per_query.jsonl" \
  bash scripts/submit_runai.sh
}

# Local negative controls: complete 5-way and 10-way groups under stride negatives.
submit_eval "5way-real-stride" 250 4 real stride
submit_eval "5way-zero-stride" 250 4 zero stride
submit_eval "5way-random-stride" 250 4 random stride
submit_eval "5way-shuffled-stride" 250 4 shuffled stride
submit_eval "5way-noprefix-stride" 250 4 real stride 250 no_prefix_lm
submit_encoder_baseline "5way-encoder-stride" 250 4 stride 250

submit_eval "10way-real-stride" 250 9 real stride
submit_eval "10way-zero-stride" 250 9 zero stride
submit_eval "10way-random-stride" 250 9 random stride
submit_eval "10way-shuffled-stride" 250 9 shuffled stride
submit_eval "10way-noprefix-stride" 250 9 real stride 250 no_prefix_lm
submit_encoder_baseline "10way-encoder-stride" 250 9 stride 250

# Hard-text negatives are diagnostic; they are intentionally separate from the complete stride control groups.
submit_eval "10way-hardtext" 250 9 real hard_text

# Full-matrix controls: 50-way and 250-way complete groups. These are more expensive but still one GPU each.
submit_eval "50way-real-full" 50 -1 real stride 50
submit_eval "50way-zero-full" 50 -1 zero stride 50
submit_eval "50way-random-full" 50 -1 random stride 50
submit_eval "50way-shuffled-full" 50 -1 shuffled stride 50
submit_eval "50way-noprefix-full" 50 -1 real stride 50 no_prefix_lm
submit_encoder_baseline "50way-encoder-full" 50 -1 stride 50

submit_eval "250way-real-full" 250 -1 real stride "$FULL_250_QUERIES" shared_prefix "$FULL_250_QUERY_OFFSET" 0
submit_eval "250way-zero-full" 250 -1 zero stride "$FULL_250_QUERIES" shared_prefix "$FULL_250_QUERY_OFFSET" 0
submit_eval "250way-random-full" 250 -1 random stride "$FULL_250_QUERIES" shared_prefix "$FULL_250_QUERY_OFFSET" 0
submit_eval "250way-shuffled-full" 250 -1 shuffled stride "$FULL_250_QUERIES" shared_prefix "$FULL_250_QUERY_OFFSET" 0
submit_eval "250way-noprefix-full" 250 -1 real stride "$FULL_250_QUERIES" no_prefix_lm "$FULL_250_QUERY_OFFSET" 0
submit_encoder_baseline "250way-encoder-full" 250 -1 stride "$FULL_250_QUERIES" "$FULL_250_QUERY_OFFSET" 0
