#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
GIT=(git -c "safe.directory=$REPO_ROOT" -C "$REPO_ROOT")
SOURCE_COMMIT_SHA="${SOURCE_COMMIT_SHA:?SOURCE_COMMIT_SHA is required}"
if [[ ! "$SOURCE_COMMIT_SHA" =~ ^[0-9a-f]{40}$ ]]; then
  echo "SOURCE_COMMIT_SHA must be an exact lowercase 40-hex commit" >&2
  exit 2
fi
if [ "$("${GIT[@]}" rev-parse HEAD)" != "$SOURCE_COMMIT_SHA" ]; then
  echo "SOURCE_COMMIT_SHA does not match the Stage-B producer checkout" >&2
  exit 2
fi
if ! "${GIT[@]}" diff --quiet || ! "${GIT[@]}" diff --cached --quiet || \
  [ -n "$("${GIT[@]}" ls-files --others --exclude-standard)" ]; then
  echo "fresh Stage-B provenance requires a clean producer checkout" >&2
  exit 2
fi
export SOURCE_COMMIT_SHA

STAMP="${STAMP:-$(date +%y%m%d%H%M)}"
export JOB_NAME="${JOB_NAME:-sparse-moe-top2-distill-${STAMP}}"
export MODE="top2-distill-real"
PROJECT_ROOT="${PROJECT_ROOT:-$REPO_ROOT}"
export OUT="${OUT:-${PROJECT_ROOT}/outputs/${JOB_NAME}}"
export DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/real_subset_clean_260708b}"
export GPU="${GPU:-1}"
export CPU="${CPU:-8}"
export MEMORY="${MEMORY:-120G}"
export REPO_DIR="${REPO_DIR:-${REPO_ROOT}}"

export CAPACITY_FACTOR="${CAPACITY_FACTOR:-6.0}"
export AUX_COEF="${AUX_COEF:-0.01}"
export TEXT_EVAL_BLOCKS="${TEXT_EVAL_BLOCKS:-160}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
export DISTILL_STEPS="${DISTILL_STEPS:-800}"
export TRAIN_LM_HEAD="${TRAIN_LM_HEAD:-1}"
export TRAIN_ROUTER_GATES="${TRAIN_ROUTER_GATES:-1}"
export TRAIN_GAMMA_SCALE="${TRAIN_GAMMA_SCALE:-0}"
export LEARNING_RATE="${LEARNING_RATE:-0.00001}"
export ROUTER_LEARNING_RATE="${ROUTER_LEARNING_RATE:-0.00001}"
export GAMMA_LEARNING_RATE="${GAMMA_LEARNING_RATE:-0.0001}"
export DISTILL_LOGIT_COEF="${DISTILL_LOGIT_COEF:-0.1}"
export DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-2.0}"
export ROUTER_DISTILL_COEF="${ROUTER_DISTILL_COEF:-0.0}"
export ROUTER_DISTILL_TEMPERATURE="${ROUTER_DISTILL_TEMPERATURE:-2.0}"
export DISTILL_HIDDEN_COEF="${DISTILL_HIDDEN_COEF:-0.0}"
export DISTILL_HIDDEN_LAYERS="${DISTILL_HIDDEN_LAYERS:-last}"
export DISTILL_HIDDEN_MODE="${DISTILL_HIDDEN_MODE:-cosine}"
export MOE_RECONSTRUCTION_COEF="${MOE_RECONSTRUCTION_COEF:-0.1}"
export MOE_RECONSTRUCTION_LAYERS="${MOE_RECONSTRUCTION_LAYERS:-all}"
export TEXT_REPLAY_COEF="${TEXT_REPLAY_COEF:-0.2}"
export TEXT_REPLAY_MANIFEST="${TEXT_REPLAY_MANIFEST:-${DATA_DIR}/text_blocks_train.jsonl}"
export STUDENT_K_CURRICULUM="${STUDENT_K_CURRICULUM:-8,4,2}"
export RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
export CHECKPOINT_EVERY_STEPS="${CHECKPOINT_EVERY_STEPS:-100}"
export LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-50}"

bash scripts/submit_runai.sh
