#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
RUNAI_ENV_FILE="${RUNAI_ENV_FILE:-$REPO_ROOT/.env.runai}"

if [ -e "$RUNAI_ENV_FILE" ]; then
  if [ ! -f "$RUNAI_ENV_FILE" ] || [ -L "$RUNAI_ENV_FILE" ]; then
    echo "Refusing to source non-regular Run:AI environment file: $RUNAI_ENV_FILE" >&2
    exit 2
  fi
  restore_allexport=0
  case "$-" in
    *a*) ;;
    *) set -a; restore_allexport=1 ;;
  esac
  # shellcheck disable=SC1090
  source "$RUNAI_ENV_FILE"
  if [ "$restore_allexport" = "1" ]; then
    set +a
  fi
fi

REPO_DIR="${SUBMIT_REPO_DIR:-${REPO_DIR:-}}"

required_vars=(
  PROJECT IMAGE RUN_AS_UID RUN_AS_GID REPO_DIR
  SCRATCH_PVC SCRATCH_MOUNT HOME_PVC HOME_MOUNT
)
missing_vars=()
for name in "${required_vars[@]}"; do
  if [ -z "${!name:-}" ]; then
    missing_vars+=("$name")
  fi
done
if [ "${#missing_vars[@]}" -gt 0 ]; then
  printf 'Missing required Run:AI configuration: %s\n' "${missing_vars[*]}" >&2
  echo "Set these variables in .env.runai or the submission environment." >&2
  exit 2
fi

JOB_NAME="${JOB_NAME:-sparse-moe-smoke-$(date +%y%m%d%H%M%S)}"
MODE="${MODE:-smoke}"
GPU="${GPU:-0.25}"
CPU="${CPU:-4}"
MEMORY="${MEMORY:-32G}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
VENV_PATH="${VENV_PATH:-$REPO_DIR/.venv}"

if [ "$MODE" = "conditional-eval" ] && {
  { [ -n "${STAGE_B_CHECKPOINT:-}" ] && [ -z "${STAGE_B_CHECKPOINT_SHA256:-}" ]; } ||
    { [ -z "${STAGE_B_CHECKPOINT:-}" ] && [ -n "${STAGE_B_CHECKPOINT_SHA256:-}" ]; }
}; then
  echo "conditional-eval requires Stage-B checkpoint path and SHA256 together" >&2
  exit 2
fi

echo "Submitting $JOB_NAME to $PROJECT with GPU=$GPU CPU=$CPU MEMORY=$MEMORY"
runai submit --name "$JOB_NAME" \
  -p "$PROJECT" \
  -i "$IMAGE" \
  -g "$GPU" \
  --cpu "$CPU" \
  --memory "$MEMORY" \
  --run-as-uid "$RUN_AS_UID" \
  --run-as-gid "$RUN_AS_GID" \
  --environment OUT="${OUT:-}" \
  --environment MAX_STEPS="${MAX_STEPS:-}" \
  --environment TOP_K="${TOP_K:-}" \
  --environment TOP_K_LIST="${TOP_K_LIST:-}" \
  --environment TEACHER_TOP_K="${TEACHER_TOP_K:-}" \
  --environment STUDENT_TOP_K="${STUDENT_TOP_K:-}" \
  --environment HF_HOME="$HF_HOME" \
  --environment VENV_PATH="$VENV_PATH" \
  --environment GIT_CONFIG_COUNT=1 \
  --environment GIT_CONFIG_KEY_0=safe.directory \
  --environment GIT_CONFIG_VALUE_0="$REPO_DIR" \
  --environment RUNAI_JOB_NAME="$JOB_NAME" \
  --environment RUNAI_PROJECT="$PROJECT" \
  --environment SOURCE_COMMIT_SHA="${SOURCE_COMMIT_SHA:-}" \
  --environment SEED="${SEED:-}" \
  --environment BASE_MODEL="${BASE_MODEL:-}" \
  --environment VISION_MODEL="${VISION_MODEL:-}" \
  --environment SPEECH_MODEL="${SPEECH_MODEL:-}" \
  --environment SPEECH_TARGET_SPACE="${SPEECH_TARGET_SPACE:-}" \
  --environment IMAGE_ALIGNMENT_TARGET="${IMAGE_ALIGNMENT_TARGET:-}" \
  --environment IMAGE_BRIDGE_TYPE="${IMAGE_BRIDGE_TYPE:-}" \
  --environment AUDIO_BRIDGE_TYPE="${AUDIO_BRIDGE_TYPE:-}" \
  --environment BRIDGE_NUM_HEADS="${BRIDGE_NUM_HEADS:-}" \
  --environment ALIGNMENT_PREFIX_RESIDUAL="${ALIGNMENT_PREFIX_RESIDUAL:-}" \
  --environment CAPACITY_FACTOR="${CAPACITY_FACTOR:-}" \
  --environment AUX_COEF="${AUX_COEF:-}" \
  --environment ROUTER_Z_LOSS_COEF="${ROUTER_Z_LOSS_COEF:-}" \
  --environment EXPERT_DROPOUT_PROB="${EXPERT_DROPOUT_PROB:-}" \
  --environment DYNAMIC_EXPERT_BIAS_LR="${DYNAMIC_EXPERT_BIAS_LR:-}" \
  --environment DYNAMIC_EXPERT_BIAS_UPDATE_INTERVAL="${DYNAMIC_EXPERT_BIAS_UPDATE_INTERVAL:-}" \
  --environment DYNAMIC_EXPERT_BIAS_WARMUP_STEPS="${DYNAMIC_EXPERT_BIAS_WARMUP_STEPS:-}" \
  --environment DYNAMIC_EXPERT_BIAS_MAX_ABS="${DYNAMIC_EXPERT_BIAS_MAX_ABS:-}" \
  --environment FINAL_STEPS="${FINAL_STEPS:-}" \
  --environment ABLATION_STEPS="${ABLATION_STEPS:-}" \
  --environment CAPACITY_ABLATION_STEPS="${CAPACITY_ABLATION_STEPS:-}" \
  --environment ABLATION_EXPERIMENTS="${ABLATION_EXPERIMENTS:-}" \
  --environment RECOVER_EXISTING="${RECOVER_EXISTING:-}" \
  --environment EXPERT_ABLATION_STEPS="${EXPERT_ABLATION_STEPS:-}" \
  --environment DATA_DIR="${DATA_DIR:-}" \
  --environment DEVELOPMENT_SPLIT_MANIFEST="${DEVELOPMENT_SPLIT_MANIFEST:-}" \
  --environment DEVELOPMENT_SPEECH_SOURCE_SHA256="${DEVELOPMENT_SPEECH_SOURCE_SHA256:-}" \
  --environment FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR:-}" \
  --environment ALLOW_SHORT="${ALLOW_SHORT:-}" \
  --environment REAL_TEXT_SAMPLES="${REAL_TEXT_SAMPLES:-}" \
  --environment REAL_CODE_SAMPLES="${REAL_CODE_SAMPLES:-}" \
  --environment REAL_REASONING_SAMPLES="${REAL_REASONING_SAMPLES:-}" \
  --environment REAL_MATH_SAMPLES="${REAL_MATH_SAMPLES:-}" \
  --environment REAL_EDUCATION_SAMPLES="${REAL_EDUCATION_SAMPLES:-}" \
  --environment REAL_IMAGE_SAMPLES="${REAL_IMAGE_SAMPLES:-}" \
  --environment REAL_SPEECH_SAMPLES="${REAL_SPEECH_SAMPLES:-}" \
  --environment REAL_MAX_AUDIO_SECONDS="${REAL_MAX_AUDIO_SECONDS:-}" \
  --environment CAPTION_MIN_ASCII_RATIO="${CAPTION_MIN_ASCII_RATIO:-}" \
  --environment CAPTION_MIN_LETTERS="${CAPTION_MIN_LETTERS:-}" \
  --environment REAL_MAX_SOURCE_AUDIO_SECONDS="${REAL_MAX_SOURCE_AUDIO_SECONDS:-}" \
  --environment REAL_MAX_TRANSCRIPT_WORDS="${REAL_MAX_TRANSCRIPT_WORDS:-}" \
  --environment BLOCK_SIZE="${BLOCK_SIZE:-}" \
  --environment TEXT_TRAIN_BLOCKS="${TEXT_TRAIN_BLOCKS:-}" \
  --environment CODE_TRAIN_BLOCKS="${CODE_TRAIN_BLOCKS:-}" \
  --environment REASONING_TRAIN_BLOCKS="${REASONING_TRAIN_BLOCKS:-}" \
  --environment MATH_TRAIN_BLOCKS="${MATH_TRAIN_BLOCKS:-}" \
  --environment EDUCATION_TRAIN_BLOCKS="${EDUCATION_TRAIN_BLOCKS:-}" \
  --environment EVAL_BLOCKS_PER_TASK="${EVAL_BLOCKS_PER_TASK:-}" \
  --environment IMAGE_EVAL_SAMPLES="${IMAGE_EVAL_SAMPLES:-}" \
  --environment SPEECH_EVAL_SAMPLES="${SPEECH_EVAL_SAMPLES:-}" \
  --environment SPEECH_SPLIT_SEED="${SPEECH_SPLIT_SEED:-}" \
  --environment RETRIEVAL_EVAL_SAMPLES="${RETRIEVAL_EVAL_SAMPLES:-}" \
  --environment CONDITIONAL_EVAL_SAMPLES="${CONDITIONAL_EVAL_SAMPLES:-}" \
  --environment CONDITIONAL_NEGATIVES="${CONDITIONAL_NEGATIVES:-}" \
  --environment CONDITIONAL_RANKING_NEGATIVES="${CONDITIONAL_RANKING_NEGATIVES:-}" \
  --environment CONDITIONAL_RANKING_NEGATIVE_MODE="${CONDITIONAL_RANKING_NEGATIVE_MODE:-}" \
  --environment CONDITIONAL_RANKING_HARD_POOL_SIZE="${CONDITIONAL_RANKING_HARD_POOL_SIZE:-}" \
  --environment CONDITIONAL_RANKING_TEMPERATURE="${CONDITIONAL_RANKING_TEMPERATURE:-}" \
  --environment IMAGE_CONDITIONAL_RANKING_COEF="${IMAGE_CONDITIONAL_RANKING_COEF:-}" \
  --environment SPEECH_CONDITIONAL_RANKING_COEF="${SPEECH_CONDITIONAL_RANKING_COEF:-}" \
  --environment SPEECH_BEHAVIOR_KL_COEF="${SPEECH_BEHAVIOR_KL_COEF:-}" \
  --environment SPEECH_BEHAVIOR_KL_TEMPERATURE="${SPEECH_BEHAVIOR_KL_TEMPERATURE:-}" \
  --environment SPEECH_SHARED_CONTRASTIVE_COEF="${SPEECH_SHARED_CONTRASTIVE_COEF:-}" \
  --environment SPEECH_SHARED_CONTRASTIVE_TEMPERATURE="${SPEECH_SHARED_CONTRASTIVE_TEMPERATURE:-}" \
  --environment SPEECH_TEACHER_BANK_BATCH_SIZE="${SPEECH_TEACHER_BANK_BATCH_SIZE:-}" \
  --environment MAX_LENGTH="${MAX_LENGTH:-}" \
  --environment TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-}" \
  --environment EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-}" \
  --environment MODALITY_CYCLE="${MODALITY_CYCLE:-}" \
  --environment TEXT_EVAL_BLOCKS="${TEXT_EVAL_BLOCKS:-}" \
  --environment IMAGE_PREFIX_TOKENS="${IMAGE_PREFIX_TOKENS:-}" \
  --environment AUDIO_PREFIX_TOKENS="${AUDIO_PREFIX_TOKENS:-}" \
  --environment ENCODER_FEATURE_TOKENS="${ENCODER_FEATURE_TOKENS:-}" \
  --environment SAMPLE_RATE="${SAMPLE_RATE:-}" \
  --environment AUDIO_MAX_SECONDS="${AUDIO_MAX_SECONDS:-}" \
  --environment TRAIN_ROUTER_GATES="${TRAIN_ROUTER_GATES:-}" \
  --environment TRAIN_GAMMA_SCALE="${TRAIN_GAMMA_SCALE:-}" \
  --environment TRAIN_EXPERTS="${TRAIN_EXPERTS:-}" \
  --environment TRAIN_LM_HEAD="${TRAIN_LM_HEAD:-}" \
  --environment CAPACITY_ABLATION_FACTOR="${CAPACITY_ABLATION_FACTOR:-}" \
  --environment LEARNING_RATE="${LEARNING_RATE:-}" \
  --environment ROUTER_LEARNING_RATE="${ROUTER_LEARNING_RATE:-}" \
  --environment GAMMA_LEARNING_RATE="${GAMMA_LEARNING_RATE:-}" \
  --environment EXPERT_LEARNING_RATE="${EXPERT_LEARNING_RATE:-}" \
  --environment EXPERT_SELECTION_JSON="${EXPERT_SELECTION_JSON:-}" \
  --environment EXPERT_SELECTION_METHOD="${EXPERT_SELECTION_METHOD:-}" \
  --environment EXPERT_UPDATE_MODE="${EXPERT_UPDATE_MODE:-}" \
  --environment EXPERT_ANCHOR_COEFFICIENT="${EXPERT_ANCHOR_COEFFICIENT:-}" \
  --environment ALLOW_SELECTED_EXPERT_ROUTER_TUNING="${ALLOW_SELECTED_EXPERT_ROUTER_TUNING:-}" \
  --environment STAGE_B_CHECKPOINT="${STAGE_B_CHECKPOINT:-}" \
  --environment STAGE_B_CHECKPOINT_SHA256="${STAGE_B_CHECKPOINT_SHA256:-}" \
  --environment MULTIMODAL_INITIAL_CHECKPOINT="${MULTIMODAL_INITIAL_CHECKPOINT:-}" \
  --environment MULTIMODAL_INITIAL_CHECKPOINT_SHA256="${MULTIMODAL_INITIAL_CHECKPOINT_SHA256:-}" \
  --environment MULTIMODAL_INITIAL_MANIFEST="${MULTIMODAL_INITIAL_MANIFEST:-}" \
  --environment MULTIMODAL_INITIALIZATION_SCOPE="${MULTIMODAL_INITIALIZATION_SCOPE:-}" \
  --environment SPEECH_INITIAL_CHECKPOINT="${SPEECH_INITIAL_CHECKPOINT:-}" \
  --environment SPEECH_INITIAL_CHECKPOINT_SHA256="${SPEECH_INITIAL_CHECKPOINT_SHA256:-}" \
  --environment SPEECH_INITIAL_MANIFEST="${SPEECH_INITIAL_MANIFEST:-}" \
  --environment RETRIEVAL_HEAD_LEARNING_RATE="${RETRIEVAL_HEAD_LEARNING_RATE:-}" \
  --environment LM_HEAD_LEARNING_RATE="${LM_HEAD_LEARNING_RATE:-}" \
  --environment SPEECH_ENCODER_LEARNING_RATE="${SPEECH_ENCODER_LEARNING_RATE:-}" \
  --environment SPEECH_UNFREEZE_LAST_BLOCKS="${SPEECH_UNFREEZE_LAST_BLOCKS:-}" \
  --environment SPEECH_UNFREEZE_LAYER_NORM="${SPEECH_UNFREEZE_LAYER_NORM:-}" \
  --environment CONTRASTIVE_COEF="${CONTRASTIVE_COEF:-}" \
  --environment IMAGE_CONTRASTIVE_COEF="${IMAGE_CONTRASTIVE_COEF:-}" \
  --environment SPEECH_CONTRASTIVE_COEF="${SPEECH_CONTRASTIVE_COEF:-}" \
  --environment CENTER_POSITIVE_WEIGHT="${CENTER_POSITIVE_WEIGHT:-}" \
  --environment RAW_POSITIVE_WEIGHT="${RAW_POSITIVE_WEIGHT:-}" \
  --environment IMAGE_CENTER_POSITIVE_WEIGHT="${IMAGE_CENTER_POSITIVE_WEIGHT:-}" \
  --environment IMAGE_RAW_POSITIVE_WEIGHT="${IMAGE_RAW_POSITIVE_WEIGHT:-}" \
  --environment SPEECH_CENTER_POSITIVE_WEIGHT="${SPEECH_CENTER_POSITIVE_WEIGHT:-}" \
  --environment SPEECH_RAW_POSITIVE_WEIGHT="${SPEECH_RAW_POSITIVE_WEIGHT:-}" \
  --environment CONTRASTIVE_TEMPERATURE="${CONTRASTIVE_TEMPERATURE:-}" \
  --environment IMAGE_CONTRASTIVE_TEMPERATURE="${IMAGE_CONTRASTIVE_TEMPERATURE:-}" \
  --environment SPEECH_CONTRASTIVE_TEMPERATURE="${SPEECH_CONTRASTIVE_TEMPERATURE:-}" \
  --environment CONTRASTIVE_NEGATIVES="${CONTRASTIVE_NEGATIVES:-}" \
  --environment IMAGE_CONTRASTIVE_NEGATIVES="${IMAGE_CONTRASTIVE_NEGATIVES:-}" \
  --environment SPEECH_CONTRASTIVE_NEGATIVES="${SPEECH_CONTRASTIVE_NEGATIVES:-}" \
  --environment WEIGHT_DECAY="${WEIGHT_DECAY:-}" \
  --environment GRAD_CLIP="${GRAD_CLIP:-}" \
  --environment LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-}" \
  --environment SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-}" \
  --environment ALIGNMENT_PRETRAIN_STEPS="${ALIGNMENT_PRETRAIN_STEPS:-}" \
  --environment ALIGNMENT_PRETRAIN_LOG_EVERY="${ALIGNMENT_PRETRAIN_LOG_EVERY:-}" \
  --environment ALIGNMENT_PRETRAIN_MODALITIES="${ALIGNMENT_PRETRAIN_MODALITIES:-}" \
  --environment DISTILL_STEPS="${DISTILL_STEPS:-}" \
  --environment DISTILL_LOGIT_COEF="${DISTILL_LOGIT_COEF:-}" \
  --environment DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-}" \
  --environment ROUTER_DISTILL_COEF="${ROUTER_DISTILL_COEF:-}" \
  --environment ROUTER_DISTILL_TEMPERATURE="${ROUTER_DISTILL_TEMPERATURE:-}" \
  --environment DISTILL_HIDDEN_COEF="${DISTILL_HIDDEN_COEF:-}" \
  --environment DISTILL_HIDDEN_LAYERS="${DISTILL_HIDDEN_LAYERS:-}" \
  --environment DISTILL_HIDDEN_MODE="${DISTILL_HIDDEN_MODE:-}" \
  --environment MOE_RECONSTRUCTION_COEF="${MOE_RECONSTRUCTION_COEF:-}" \
  --environment MOE_RECONSTRUCTION_LAYERS="${MOE_RECONSTRUCTION_LAYERS:-}" \
  --environment TEXT_REPLAY_COEF="${TEXT_REPLAY_COEF:-}" \
  --environment TEXT_REPLAY_MANIFEST="${TEXT_REPLAY_MANIFEST:-}" \
  --environment STUDENT_K_CURRICULUM="${STUDENT_K_CURRICULUM:-}" \
  --environment RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}" \
  --environment CHECKPOINT_EVERY_STEPS="${CHECKPOINT_EVERY_STEPS:-}" \
  --environment GAMMA_MIN="${GAMMA_MIN:-}" \
  --environment GAMMA_MAX="${GAMMA_MAX:-}" \
  --environment EXPERT_AUDIT_DIR="${EXPERT_AUDIT_DIR:-}" \
  --environment POSTPROCESS_REQUIRED_RUNS="${POSTPROCESS_REQUIRED_RUNS:-}" \
  --environment SOURCE_OUTPUT_DIR="${SOURCE_OUTPUT_DIR:-}" \
  --environment MIRROR_SOURCE_ROOT="${MIRROR_SOURCE_ROOT:-}" \
  --environment RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-}" \
  --environment CHECKPOINT="${CHECKPOINT:-}" \
  --environment EXPECTED_CHECKPOINT_SHA256="${EXPECTED_CHECKPOINT_SHA256:-}" \
  --environment CONDITIONAL_QUERIES="${CONDITIONAL_QUERIES:-}" \
  --environment CONDITIONAL_CANDIDATES="${CONDITIONAL_CANDIDATES:-}" \
  --environment CONDITIONAL_BATCH_SIZE="${CONDITIONAL_BATCH_SIZE:-}" \
  --environment NEGATIVE_MODE="${NEGATIVE_MODE:-}" \
  --environment EVAL_PATH="${EVAL_PATH:-}" \
  --environment PREFIX_CONTROL="${PREFIX_CONTROL:-}" \
  --environment CONTROL_SEED="${CONTROL_SEED:-}" \
  --environment EVALUATION_SCOPE="${EVALUATION_SCOPE:-}" \
  --environment EVAL_SPLIT_NAME="${EVAL_SPLIT_NAME:-}" \
  --environment QUERY_OFFSET="${QUERY_OFFSET:-}" \
  --environment CANDIDATE_OFFSET="${CANDIDATE_OFFSET:-}" \
  --environment BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-}" \
  --environment BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-}" \
  --environment PER_QUERY_OUTPUT="${PER_QUERY_OUTPUT:-}" \
  --environment IMAGE_MANIFEST="${IMAGE_MANIFEST:-}" \
  --environment SPEECH_MANIFEST="${SPEECH_MANIFEST:-}" \
  --environment DEV_IMAGE_MANIFEST="${DEV_IMAGE_MANIFEST:-}" \
  --environment DEV_SPEECH_MANIFEST="${DEV_SPEECH_MANIFEST:-}" \
  --environment CANDIDATE_SEED="${CANDIDATE_SEED:-}" \
  --environment RANDOMIZE_POSITIVE_POSITION="${RANDOMIZE_POSITIVE_POSITION:-}" \
  --environment PROTOCOL_NAME="${PROTOCOL_NAME:-}" \
  --environment PROTOCOL_PATH="${PROTOCOL_PATH:-}" \
  --environment SEALED_OUTPUT_DIR="${SEALED_OUTPUT_DIR:-}" \
  --environment SEALED_INDEX_FILE="${SEALED_INDEX_FILE:-}" \
  --environment EXCLUDE_INDEX="${EXCLUDE_INDEX:-}" \
  --environment FORCE="${FORCE:-}" \
  --environment MATCHED_ROOT_1="${MATCHED_ROOT_1:-}" \
  --environment MATCHED_ROOT_2="${MATCHED_ROOT_2:-}" \
  --environment MATCHED_ROOT_3="${MATCHED_ROOT_3:-}" \
  --environment SOURCE_ROOT="${SOURCE_ROOT:-}" \
  --environment RUN_MANIFEST="${RUN_MANIFEST:-}" \
  --environment GAMMA_JSON="${GAMMA_JSON:-}" \
  --environment LOSS_WINDOW="${LOSS_WINDOW:-}" \
  --environment MATRIX_ROOT="${MATRIX_ROOT:-}" \
  --environment PERMUTATION_SAMPLES="${PERMUTATION_SAMPLES:-}" \
  --environment EVIDENCE_MANIFEST_SPEC="${EVIDENCE_MANIFEST_SPEC:-}" \
  --environment EVIDENCE_MANIFEST_OUTPUT="${EVIDENCE_MANIFEST_OUTPUT:-}" \
  --environment ROUTING_ANALYSIS_BATCH_SIZE="${ROUTING_ANALYSIS_BATCH_SIZE:-}" \
  --environment ROUTING_TEXT_BATCHES="${ROUTING_TEXT_BATCHES:-}" \
  --environment ROUTING_MODALITY_BATCHES="${ROUTING_MODALITY_BATCHES:-}" \
  --environment QUALITATIVE_EXAMPLES="${QUALITATIVE_EXAMPLES:-}" \
  --environment MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}" \
  --environment INTERVENTION_TOP_EXPERTS="${INTERVENTION_TOP_EXPERTS:-}" \
  --environment INTERVENTION_EXAMPLES="${INTERVENTION_EXAMPLES:-}" \
  --environment INTERVENTION_TEXT_BLOCKS="${INTERVENTION_TEXT_BLOCKS:-}" \
  --existing-pvc "claimname=$SCRATCH_PVC,path=$SCRATCH_MOUNT" \
  --existing-pvc "claimname=$HOME_PVC,path=$HOME_MOUNT" \
  --working-dir "$REPO_DIR" \
  --large-shm \
  --backoff-limit 0 \
  --command \
  -- /bin/bash run.sh "$MODE"
