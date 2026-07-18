#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$REPO_ROOT"

# The no-argument command is the portable release check required by the
# assignment. Formal data preparation and training remain explicit via
# `bash run.sh full`.
MODE="${1:-smoke}"

if [ -n "${SOURCE_COMMIT_SHA:-}" ]; then
  ACTUAL_SOURCE_SHA="$(git rev-parse HEAD)"
  if [ "$ACTUAL_SOURCE_SHA" != "$SOURCE_COMMIT_SHA" ]; then
    echo "source commit mismatch: expected $SOURCE_COMMIT_SHA, found $ACTUAL_SOURCE_SHA" >&2
    exit 2
  fi
  if ! git diff --quiet || ! git diff --cached --quiet || [ -n "$(git ls-files --others --exclude-standard)" ]; then
    echo "SOURCE_COMMIT_SHA runs require a clean worktree" >&2
    exit 2
  fi
fi

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$REPO_ROOT"
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

VENV_PATH="${VENV_PATH:-$REPO_ROOT/.venv}"
if [ -f "$VENV_PATH/bin/activate" ]; then
  source "$VENV_PATH/bin/activate"
fi

case "$MODE" in
  real-data-prepare)
    DATA_DIR="${DATA_DIR:-data/real_subset}"
    EXTRA_ALLOW_SHORT=()
    if [ "${ALLOW_SHORT:-0}" = "1" ]; then
      EXTRA_ALLOW_SHORT=(--allow-short)
    fi
    python datasets/build_real_subset.py \
      --output-dir "$DATA_DIR" \
      --tokenizer-model "${BASE_MODEL:-allenai/OLMoE-1B-7B-0924}" \
      --block-size "${BLOCK_SIZE:-512}" \
      --text-samples "${REAL_TEXT_SAMPLES:-30000}" \
      --code-samples "${REAL_CODE_SAMPLES:-1000}" \
      --reasoning-samples "${REAL_REASONING_SAMPLES:-651}" \
      --math-samples "${REAL_MATH_SAMPLES:-2500}" \
      --education-samples "${REAL_EDUCATION_SAMPLES:-900}" \
      --text-train-blocks "${TEXT_TRAIN_BLOCKS:-14086}" \
      --code-train-blocks "${CODE_TRAIN_BLOCKS:-1865}" \
      --reasoning-train-blocks "${REASONING_TRAIN_BLOCKS:-290}" \
      --math-train-blocks "${MATH_TRAIN_BLOCKS:-611}" \
      --education-train-blocks "${EDUCATION_TRAIN_BLOCKS:-424}" \
      --eval-blocks-per-task "${EVAL_BLOCKS_PER_TASK:-64}" \
      --image-samples "${REAL_IMAGE_SAMPLES:-5250}" \
      --speech-samples "${REAL_SPEECH_SAMPLES:-5250}" \
      --image-eval-samples "${IMAGE_EVAL_SAMPLES:-250}" \
      --speech-eval-samples "${SPEECH_EVAL_SAMPLES:-250}" \
      --speech-split-seed "${SPEECH_SPLIT_SEED:-0}" \
      --sample-rate "${SAMPLE_RATE:-16000}" \
      --max-audio-seconds "${REAL_MAX_AUDIO_SECONDS:-6.0}" \
      --caption-min-ascii-ratio "${CAPTION_MIN_ASCII_RATIO:-0.85}" \
      --caption-min-letters "${CAPTION_MIN_LETTERS:-8}" \
      --max-source-audio-seconds "${REAL_MAX_SOURCE_AUDIO_SECONDS:-0.0}" \
      --max-transcript-words "${REAL_MAX_TRANSCRIPT_WORDS:-0}" \
      "${EXTRA_ALLOW_SHORT[@]}"
    ;;
  data-quality-audit)
    DATA_DIR="${DATA_DIR:-data/real_subset}"
    python scripts/audit_data_quality.py \
      --data-dir "$DATA_DIR" \
      --min-image-rows "${REAL_IMAGE_SAMPLES:-5250}" \
      --min-speech-rows "${REAL_SPEECH_SAMPLES:-5250}" \
      --caption-min-ascii-ratio "${CAPTION_MIN_ASCII_RATIO:-0.85}" \
      --caption-min-letters "${CAPTION_MIN_LETTERS:-8}" \
      --max-source-audio-seconds "${REAL_MAX_SOURCE_AUDIO_SECONDS:-6.0}" \
      --max-transcript-words "${REAL_MAX_TRANSCRIPT_WORDS:-18}" \
      --output "${OUT:-$DATA_DIR/data_quality_audit.json}"
    ;;
  real-required-runs)
    OUT="${OUT:-outputs/real_required_runs}"
    DATA_DIR="${DATA_DIR:-data/real_subset}"
    REQUIRE_DATA_QUALITY_AUDIT="${REQUIRE_DATA_QUALITY_AUDIT:-1}"
    if [ "$REQUIRE_DATA_QUALITY_AUDIT" != "0" ]; then
      python scripts/audit_data_quality.py \
        --data-dir "$DATA_DIR" \
        --min-image-rows "${REAL_IMAGE_SAMPLES:-5250}" \
        --min-speech-rows "${REAL_SPEECH_SAMPLES:-5250}" \
        --caption-min-ascii-ratio "${CAPTION_MIN_ASCII_RATIO:-0.85}" \
        --caption-min-letters "${CAPTION_MIN_LETTERS:-8}" \
        --max-source-audio-seconds "${REAL_MAX_SOURCE_AUDIO_SECONDS:-6.0}" \
        --max-transcript-words "${REAL_MAX_TRANSCRIPT_WORDS:-18}" \
        --output "${OUT:-outputs/real_required_runs}/data_quality_audit.json"
    fi
    TRAIN_ROUTER_ARG=()
    TRAIN_EXPERT_ARG=()
    if [ "${TRAIN_ROUTER_GATES:-1}" = "1" ]; then
      TRAIN_ROUTER_ARG=(--train-router-gates)
    fi
    TRAIN_LM_HEAD_ARG=()
    ALIGNMENT_RESIDUAL_ARG=()
    SPEECH_LAYER_NORM_ARG=()
    EXPERT_SELECTION_ARG=()
    EXPERT_ROUTER_COMBINATION_ARG=()
    STAGE_B_INITIALIZATION_ARG=()
    MULTIMODAL_INITIALIZATION_ARG=()
    SPEECH_INITIALIZATION_ARG=()
    DEVELOPMENT_SPLIT_ARG=()
    if [ -n "${DEVELOPMENT_SPLIT_MANIFEST:-}" ]; then
      : "${DEVELOPMENT_SPEECH_SOURCE_SHA256:?DEVELOPMENT_SPEECH_SOURCE_SHA256 is required with DEVELOPMENT_SPLIT_MANIFEST}"
      DEVELOPMENT_SPLIT_ARG=(
        --development-split-manifest "$DEVELOPMENT_SPLIT_MANIFEST"
        --development-speech-source-sha256 "$DEVELOPMENT_SPEECH_SOURCE_SHA256"
      )
    fi
    if [ -n "${STAGE_B_CHECKPOINT:-}" ]; then
      STAGE_B_INITIALIZATION_ARG=(
        --stage-b-checkpoint "$STAGE_B_CHECKPOINT"
        --stage-b-checkpoint-sha256 "${STAGE_B_CHECKPOINT_SHA256:-}"
      )
    fi
    if [ -z "${MULTIMODAL_INITIAL_CHECKPOINT:-}" ] && {
      [ -n "${MULTIMODAL_INITIAL_CHECKPOINT_SHA256:-}" ] ||
        [ -n "${MULTIMODAL_INITIAL_MANIFEST:-}" ]
    }; then
      echo "multimodal initialization SHA/manifest requires a checkpoint" >&2
      exit 2
    fi
    if [ -n "${MULTIMODAL_INITIAL_CHECKPOINT:-}" ]; then
      MULTIMODAL_INITIALIZATION_ARG=(
        --multimodal-initial-checkpoint "$MULTIMODAL_INITIAL_CHECKPOINT"
        --multimodal-initial-checkpoint-sha256 "${MULTIMODAL_INITIAL_CHECKPOINT_SHA256:-}"
        --multimodal-initial-manifest "${MULTIMODAL_INITIAL_MANIFEST:-}"
        --multimodal-initialization-scope "${MULTIMODAL_INITIALIZATION_SCOPE:-both}"
      )
    fi
    if [ -z "${SPEECH_INITIAL_CHECKPOINT:-}" ] && {
      [ -n "${SPEECH_INITIAL_CHECKPOINT_SHA256:-}" ] ||
        [ -n "${SPEECH_INITIAL_MANIFEST:-}" ]
    }; then
      echo "speech initialization SHA/manifest requires a checkpoint" >&2
      exit 2
    fi
    if [ -n "${SPEECH_INITIAL_CHECKPOINT:-}" ]; then
      if [ -z "${SPEECH_INITIAL_CHECKPOINT_SHA256:-}" ] ||
        [ -z "${SPEECH_INITIAL_MANIFEST:-}" ]; then
        echo "speech initialization checkpoint requires an exact SHA and manifest" >&2
        exit 2
      fi
      SPEECH_INITIALIZATION_ARG=(
        --speech-initial-checkpoint "$SPEECH_INITIAL_CHECKPOINT"
        --speech-initial-checkpoint-sha256 "$SPEECH_INITIAL_CHECKPOINT_SHA256"
        --speech-initial-manifest "$SPEECH_INITIAL_MANIFEST"
      )
    fi
    if [ "${TRAIN_EXPERTS:-0}" = "1" ]; then
      TRAIN_EXPERT_ARG=(--train-experts)
    fi
    if [ "${TRAIN_LM_HEAD:-0}" = "1" ]; then
      TRAIN_LM_HEAD_ARG=(--train-lm-head)
    fi
    if [ "${ALIGNMENT_PREFIX_RESIDUAL:-0}" = "1" ]; then
      ALIGNMENT_RESIDUAL_ARG=(--alignment-prefix-residual)
    fi
    if [ "${SPEECH_UNFREEZE_LAYER_NORM:-0}" = "1" ]; then
      SPEECH_LAYER_NORM_ARG=(--speech-unfreeze-layer-norm)
    fi
    if [ -n "${EXPERT_SELECTION_JSON:-}" ]; then
      EXPERT_SELECTION_ARG=(
        --expert-selection-json "$EXPERT_SELECTION_JSON"
        --expert-selection-method "${EXPERT_SELECTION_METHOD:-ESFT-Gate}"
        --expert-update-mode "${EXPERT_UPDATE_MODE:-full}"
        --expert-anchor-coefficient "${EXPERT_ANCHOR_COEFFICIENT:-0.01}"
      )
    fi
    if [ "${ALLOW_SELECTED_EXPERT_ROUTER_TUNING:-0}" = "1" ]; then
      EXPERT_ROUTER_COMBINATION_ARG=(--allow-selected-expert-router-tuning)
    fi
    python -m training.olmoe_real_subset_runs \
      --data-dir "$DATA_DIR" \
      --output-dir "$OUT" \
      --base-model "${BASE_MODEL:-allenai/OLMoE-1B-7B-0924}" \
      --vision-model "${VISION_MODEL:-openai/clip-vit-base-patch32}" \
      --speech-model "${SPEECH_MODEL:-openai/whisper-base.en}" \
      --speech-target-space "${SPEECH_TARGET_SPACE:-olmoe_text_hidden}" \
      --image-alignment-target "${IMAGE_ALIGNMENT_TARGET:-clip_text}" \
      --image-bridge-type "${IMAGE_BRIDGE_TYPE:-query_resampler}" \
      --audio-bridge-type "${AUDIO_BRIDGE_TYPE:-query_resampler}" \
      --bridge-num-heads "${BRIDGE_NUM_HEADS:-4}" \
      --feature-cache-dir "${FEATURE_CACHE_DIR:-$OUT/feature_cache}" \
      --seed "${SEED:-42}" \
      --max-length "${MAX_LENGTH:-512}" \
      --capacity-factor "${CAPACITY_FACTOR:-4.0}" \
      --capacity-ablation-factor "${CAPACITY_ABLATION_FACTOR:-1.25}" \
      --aux-coef "${AUX_COEF:-0.01}" \
      --router-z-loss-coef "${ROUTER_Z_LOSS_COEF:-0.0}" \
      --expert-dropout-prob "${EXPERT_DROPOUT_PROB:-0.0}" \
      --dynamic-expert-bias-lr "${DYNAMIC_EXPERT_BIAS_LR:-0.0}" \
      --dynamic-expert-bias-update-interval "${DYNAMIC_EXPERT_BIAS_UPDATE_INTERVAL:-1}" \
      --dynamic-expert-bias-warmup-steps "${DYNAMIC_EXPERT_BIAS_WARMUP_STEPS:-0}" \
      --dynamic-expert-bias-max-abs "${DYNAMIC_EXPERT_BIAS_MAX_ABS:-2.0}" \
      --final-steps "${FINAL_STEPS:-4000}" \
      --ablation-steps "${ABLATION_STEPS:-1000}" \
      --capacity-ablation-steps "${CAPACITY_ABLATION_STEPS:-1000}" \
      --expert-ablation-steps "${EXPERT_ABLATION_STEPS:-0}" \
      --train-batch-size "${TRAIN_BATCH_SIZE:-1}" \
      --eval-batch-size "${EVAL_BATCH_SIZE:-4}" \
      --modality-cycle "${MODALITY_CYCLE:-text,image,speech}" \
      --text-eval-blocks "${TEXT_EVAL_BLOCKS:-250}" \
      --retrieval-eval-samples "${RETRIEVAL_EVAL_SAMPLES:-250}" \
      --conditional-eval-samples "${CONDITIONAL_EVAL_SAMPLES:-250}" \
      --conditional-negatives "${CONDITIONAL_NEGATIVES:-4}" \
      --conditional-batch-size "${CONDITIONAL_BATCH_SIZE:-16}" \
      --conditional-ranking-negatives "${CONDITIONAL_RANKING_NEGATIVES:-2}" \
      --conditional-ranking-negative-mode "${CONDITIONAL_RANKING_NEGATIVE_MODE:-stride}" \
      --conditional-ranking-hard-pool-size "${CONDITIONAL_RANKING_HARD_POOL_SIZE:-512}" \
      --conditional-ranking-temperature "${CONDITIONAL_RANKING_TEMPERATURE:-1.0}" \
      --image-conditional-ranking-coef "${IMAGE_CONDITIONAL_RANKING_COEF:-0.0}" \
      --speech-conditional-ranking-coef "${SPEECH_CONDITIONAL_RANKING_COEF:-0.0}" \
      --speech-behavior-kl-coef "${SPEECH_BEHAVIOR_KL_COEF:-0.0}" \
      --speech-behavior-kl-temperature "${SPEECH_BEHAVIOR_KL_TEMPERATURE:-1.0}" \
      --speech-shared-contrastive-coef "${SPEECH_SHARED_CONTRASTIVE_COEF:-0.0}" \
      --speech-shared-contrastive-temperature "${SPEECH_SHARED_CONTRASTIVE_TEMPERATURE:-0.07}" \
      --speech-teacher-bank-batch-size "${SPEECH_TEACHER_BANK_BATCH_SIZE:-64}" \
      --image-eval-samples "${IMAGE_EVAL_SAMPLES:-250}" \
      --speech-eval-samples "${SPEECH_EVAL_SAMPLES:-250}" \
      --image-prefix-tokens "${IMAGE_PREFIX_TOKENS:-50}" \
      --audio-prefix-tokens "${AUDIO_PREFIX_TOKENS:-64}" \
      --encoder-feature-tokens "${ENCODER_FEATURE_TOKENS:-100}" \
      --sample-rate "${SAMPLE_RATE:-16000}" \
      --audio-max-seconds "${AUDIO_MAX_SECONDS:-0.0}" \
      --learning-rate "${LEARNING_RATE:-0.0002}" \
      --router-learning-rate "${ROUTER_LEARNING_RATE:-0.00005}" \
      --expert-learning-rate "${EXPERT_LEARNING_RATE:-0.00005}" \
      --retrieval-head-learning-rate "${RETRIEVAL_HEAD_LEARNING_RATE:-0.0}" \
      --lm-head-learning-rate "${LM_HEAD_LEARNING_RATE:-0.00001}" \
      --speech-encoder-learning-rate "${SPEECH_ENCODER_LEARNING_RATE:-0.00001}" \
      --speech-unfreeze-last-blocks "${SPEECH_UNFREEZE_LAST_BLOCKS:-0}" \
      --contrastive-coef "${CONTRASTIVE_COEF:-0.1}" \
      --image-contrastive-coef "${IMAGE_CONTRASTIVE_COEF:--1.0}" \
      --speech-contrastive-coef "${SPEECH_CONTRASTIVE_COEF:--1.0}" \
      --center-positive-weight "${CENTER_POSITIVE_WEIGHT:-1.0}" \
      --raw-positive-weight "${RAW_POSITIVE_WEIGHT:-1.0}" \
      --image-center-positive-weight "${IMAGE_CENTER_POSITIVE_WEIGHT:--1.0}" \
      --image-raw-positive-weight "${IMAGE_RAW_POSITIVE_WEIGHT:--1.0}" \
      --speech-center-positive-weight "${SPEECH_CENTER_POSITIVE_WEIGHT:--1.0}" \
      --speech-raw-positive-weight "${SPEECH_RAW_POSITIVE_WEIGHT:--1.0}" \
      --contrastive-temperature "${CONTRASTIVE_TEMPERATURE:-0.07}" \
      --image-contrastive-temperature "${IMAGE_CONTRASTIVE_TEMPERATURE:--1.0}" \
      --speech-contrastive-temperature "${SPEECH_CONTRASTIVE_TEMPERATURE:--1.0}" \
      --contrastive-negatives "${CONTRASTIVE_NEGATIVES:-128}" \
      --image-contrastive-negatives "${IMAGE_CONTRASTIVE_NEGATIVES:--2}" \
      --speech-contrastive-negatives "${SPEECH_CONTRASTIVE_NEGATIVES:--2}" \
      --weight-decay "${WEIGHT_DECAY:-0.01}" \
      --grad-clip "${GRAD_CLIP:-1.0}" \
      --log-every-steps "${LOG_EVERY_STEPS:-20}" \
      --save-every-steps "${SAVE_EVERY_STEPS:-500}" \
      --alignment-pretrain-steps "${ALIGNMENT_PRETRAIN_STEPS:-0}" \
      --alignment-pretrain-log-every "${ALIGNMENT_PRETRAIN_LOG_EVERY:-100}" \
      --alignment-pretrain-modalities "${ALIGNMENT_PRETRAIN_MODALITIES:-image,speech}" \
      "${TRAIN_ROUTER_ARG[@]}" "${TRAIN_EXPERT_ARG[@]}" "${TRAIN_LM_HEAD_ARG[@]}" "${ALIGNMENT_RESIDUAL_ARG[@]}" "${SPEECH_LAYER_NORM_ARG[@]}" "${EXPERT_SELECTION_ARG[@]}" "${EXPERT_ROUTER_COMBINATION_ARG[@]}" "${STAGE_B_INITIALIZATION_ARG[@]}" "${MULTIMODAL_INITIALIZATION_ARG[@]}" "${SPEECH_INITIALIZATION_ARG[@]}" "${DEVELOPMENT_SPLIT_ARG[@]}"
    ;;
  debug-e0)
    OUT="${OUT:-outputs/e0_sanity}"
    DATA_DIR="${DATA_DIR:-data/real_subset_final}"
    python scripts/diagnose_e0_sanity.py \
      --data-dir "$DATA_DIR" \
      --output-dir "$OUT" \
      --base-model "${BASE_MODEL:-allenai/OLMoE-1B-7B-0924}" \
      --blocks "${DEBUG_E0_BLOCKS:-40}" \
      --max-length "${MAX_LENGTH:-512}" \
      --batch-size "${EVAL_BATCH_SIZE:-2}" \
      --aux-coef "${AUX_COEF:-0.0}" \
      --capacity-factor "${CAPACITY_FACTOR:-4.0}"
    ;;
  inspect-olmoe)
    OUT="${OUT:-outputs/inspect_olmoe_impl}"
    python scripts/inspect_olmoe_impl.py --output-dir "$OUT"
    ;;
  encoder-baseline-eval)
    OUT="${OUT:-outputs/encoder_baseline_eval/metrics.json}"
    DATA_DIR="${DATA_DIR:-data/real_subset_final}"
    python scripts/eval_encoder_baselines.py \
      --data-dir "$DATA_DIR" \
      --output "$OUT" \
      --vision-model "${VISION_MODEL:-openai/clip-vit-base-patch32}" \
      --speech-model "${SPEECH_MODEL:-openai/whisper-base.en}" \
      --feature-cache-dir "${FEATURE_CACHE_DIR:-}" \
      --sample-rate "${SAMPLE_RATE:-16000}" \
      --encoder-feature-tokens "${ENCODER_FEATURE_TOKENS:-100}" \
      --batch-size "${CONDITIONAL_BATCH_SIZE:-16}" \
      --image-eval-samples "${IMAGE_EVAL_SAMPLES:-250}" \
      --speech-eval-samples "${SPEECH_EVAL_SAMPLES:-250}" \
      --conditional-queries "${CONDITIONAL_QUERIES:-250}" \
      --conditional-candidates "${CONDITIONAL_CANDIDATES:-250}" \
      --conditional-negatives "${CONDITIONAL_NEGATIVES:--1}" \
      --negative-mode "${NEGATIVE_MODE:-stride}" \
      --eval-split-name "${EVAL_SPLIT_NAME:-eval_tail}" \
      --query-offset "${QUERY_OFFSET:-0}" \
      --candidate-offset "${CANDIDATE_OFFSET:--1}" \
      --bootstrap-samples "${BOOTSTRAP_SAMPLES:-1000}" \
      --bootstrap-seed "${BOOTSTRAP_SEED:-12345}" \
      --per-query-output "${PER_QUERY_OUTPUT:-}"
    ;;
  conditional-eval)
    OUT="${OUT:-outputs/conditional_eval/metrics.json}"
    DATA_DIR="${DATA_DIR:-data/real_subset_final}"
    ALIGNMENT_RESIDUAL_ARG=()
    IMAGE_MANIFEST_ARG=()
    SPEECH_MANIFEST_ARG=()
    PROTOCOL_MANIFEST_ARG=()
    RANDOMIZE_POSITIVE_ARG=()
    FEATURE_CACHE_ARG=()
    if [ "${ALIGNMENT_PREFIX_RESIDUAL:-0}" = "1" ]; then
      ALIGNMENT_RESIDUAL_ARG=(--alignment-prefix-residual)
    fi
    if [ -n "${IMAGE_MANIFEST:-}" ]; then
      IMAGE_MANIFEST_ARG=(--image-manifest "$IMAGE_MANIFEST")
    fi
    if [ -n "${SPEECH_MANIFEST:-}" ]; then
      SPEECH_MANIFEST_ARG=(--speech-manifest "$SPEECH_MANIFEST")
    fi
    if [ -n "${PROTOCOL_PATH:-}" ]; then
      python scripts/freeze_evaluation_protocol.py --verify "$PROTOCOL_PATH" --verify-git-state
      PROTOCOL_MANIFEST_ARG=(--protocol-manifest "$PROTOCOL_PATH")
    fi
    if [ "${RANDOMIZE_POSITIVE_POSITION:-0}" = "1" ]; then
      RANDOMIZE_POSITIVE_ARG=(--randomize-positive-position)
    fi
    if [ -n "${FEATURE_CACHE_DIR:-}" ]; then
      FEATURE_CACHE_ARG=(--feature-cache-dir "$FEATURE_CACHE_DIR")
    fi
    python scripts/eval_conditional_retrieval.py \
      --evaluation-scope "${EVALUATION_SCOPE:?EVALUATION_SCOPE is required}" \
      --data-dir "$DATA_DIR" \
      "${IMAGE_MANIFEST_ARG[@]}" "${SPEECH_MANIFEST_ARG[@]}" \
      --run-output-dir "${RUN_OUTPUT_DIR:?RUN_OUTPUT_DIR is required}" \
      --checkpoint "${CHECKPOINT:?CHECKPOINT is required}" \
      --stage-b-checkpoint "${STAGE_B_CHECKPOINT:-}" \
      --stage-b-checkpoint-sha256 "${STAGE_B_CHECKPOINT_SHA256:-}" \
      --output "$OUT" \
      --base-model "${BASE_MODEL:-allenai/OLMoE-1B-7B-0924}" \
      --vision-model "${VISION_MODEL:-openai/clip-vit-base-patch32}" \
      --speech-model "${SPEECH_MODEL:-openai/whisper-base.en}" \
      --speech-target-space "${SPEECH_TARGET_SPACE:-olmoe_text_hidden}" \
      --max-length "${MAX_LENGTH:-512}" \
      --capacity-factor "${CAPACITY_FACTOR:-4.0}" \
      --aux-coef "${AUX_COEF:-0.01}" \
      --top-k "${TOP_K:-2}" \
      --image-prefix-tokens "${IMAGE_PREFIX_TOKENS:-50}" \
      --audio-prefix-tokens "${AUDIO_PREFIX_TOKENS:-64}" \
      --encoder-feature-tokens "${ENCODER_FEATURE_TOKENS:-100}" \
      --sample-rate "${SAMPLE_RATE:-16000}" \
      --image-eval-samples "${IMAGE_EVAL_SAMPLES:-250}" \
      --speech-eval-samples "${SPEECH_EVAL_SAMPLES:-250}" \
      --conditional-queries "${CONDITIONAL_QUERIES:-250}" \
      --conditional-candidates "${CONDITIONAL_CANDIDATES:-250}" \
      --conditional-negatives "${CONDITIONAL_NEGATIVES:-4}" \
      --conditional-batch-size "${CONDITIONAL_BATCH_SIZE:-8}" \
      --negative-mode "${NEGATIVE_MODE:-stride}" \
      --eval-path "${EVAL_PATH:-shared_prefix}" \
      --prefix-control "${PREFIX_CONTROL:-real}" \
      --control-seed "${CONTROL_SEED:-42}" \
      --candidate-seed "${CANDIDATE_SEED:-314159}" \
      --protocol-name "${PROTOCOL_NAME:-conditional_matching_v2}" \
      "${PROTOCOL_MANIFEST_ARG[@]}" "${FEATURE_CACHE_ARG[@]}" \
      --eval-split-name "${EVAL_SPLIT_NAME:-eval_tail}" \
      --query-offset "${QUERY_OFFSET:-0}" \
      --candidate-offset "${CANDIDATE_OFFSET:--1}" \
      --bootstrap-samples "${BOOTSTRAP_SAMPLES:-1000}" \
      --bootstrap-seed "${BOOTSTRAP_SEED:-12345}" \
      --per-query-output "${PER_QUERY_OUTPUT:-}" \
      "${ALIGNMENT_RESIDUAL_ARG[@]}" "${RANDOMIZE_POSITIVE_ARG[@]}"
    ;;
  e3-refresh)
    OUT="${OUT:?OUT is required for e3-refresh}"
    SOURCE_OUTPUT_DIR="${SOURCE_OUTPUT_DIR:?SOURCE_OUTPUT_DIR is required for e3-refresh}"
    CHECKPOINT="${CHECKPOINT:?CHECKPOINT is required for e3-refresh}"
    CHECKPOINT_SHA256="${CHECKPOINT_SHA256:?CHECKPOINT_SHA256 is required for e3-refresh}"
    STAGE_B_CHECKPOINT="${STAGE_B_CHECKPOINT:?STAGE_B_CHECKPOINT is required for e3-refresh}"
    STAGE_B_CHECKPOINT_SHA256="${STAGE_B_CHECKPOINT_SHA256:?STAGE_B_CHECKPOINT_SHA256 is required for e3-refresh}"
    MIRROR_ARG=()
    if [ "${MIRROR_SOURCE_ROOT:-0}" = "1" ]; then
      MIRROR_ARG=(--mirror-source-root)
    fi
    FEATURE_CACHE_ARG=()
    if [ -n "${FEATURE_CACHE_DIR:-}" ]; then
      FEATURE_CACHE_ARG=(--feature-cache-dir "$FEATURE_CACHE_DIR")
    fi
    python scripts/refresh_e3_from_checkpoint.py \
      --source-output-dir "$SOURCE_OUTPUT_DIR" \
      --output-dir "$OUT" \
      --data-dir "${DATA_DIR:-data/real_subset_final}" \
      --checkpoint "$CHECKPOINT" \
      --checkpoint-sha256 "$CHECKPOINT_SHA256" \
      --stage-b-checkpoint "$STAGE_B_CHECKPOINT" \
      --stage-b-checkpoint-sha256 "$STAGE_B_CHECKPOINT_SHA256" \
      "${FEATURE_CACHE_ARG[@]}" "${MIRROR_ARG[@]}"
    python scripts/rebuild_refresh_summary.py --source-output-dir "$SOURCE_OUTPUT_DIR" --output-dir "$OUT"
    ;;
  text-eval-refresh)
    OUT="${OUT:?OUT is required for text-eval-refresh}"
    SOURCE_OUTPUT_DIR="${SOURCE_OUTPUT_DIR:?SOURCE_OUTPUT_DIR is required for text-eval-refresh}"
    MIRROR_ARG=()
    if [ "${MIRROR_SOURCE_ROOT:-0}" = "1" ]; then
      MIRROR_ARG=(--mirror-source-root)
    fi
    python scripts/refresh_text_eval_from_checkpoints.py       --source-output-dir "$SOURCE_OUTPUT_DIR"       --output-dir "$OUT"       --data-dir "${DATA_DIR:-data/real_subset_final}"       --base-model "${BASE_MODEL:-allenai/OLMoE-1B-7B-0924}"       --experiments "${TEXT_REFRESH_EXPERIMENTS:-E4_no_aux_load_balance_ablation,E5_capacity_1p25_ablation}"       --text-eval-blocks "${TEXT_EVAL_BLOCKS:-160}"       --eval-batch-size "${EVAL_BATCH_SIZE:-8}"       --max-length "${MAX_LENGTH:-512}"       "${MIRROR_ARG[@]}"
    ;;
  specialization-analysis)
    OUT="${OUT:-outputs/sparse-moe-final-lmheadrank-260708m-refresh2}"
    RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-$OUT}"
    CHECKPOINT="${CHECKPOINT:-$OUT/E3_final_multimodal_top2/checkpoint_final.pt}"
    IMAGE_MANIFEST_ARG=()
    SPEECH_MANIFEST_ARG=()
    if [ -n "${IMAGE_MANIFEST:-}" ]; then
      IMAGE_MANIFEST_ARG=(--image-manifest "$IMAGE_MANIFEST")
    fi
    if [ -n "${SPEECH_MANIFEST:-}" ]; then
      SPEECH_MANIFEST_ARG=(--speech-manifest "$SPEECH_MANIFEST")
    fi
    python scripts/analyze_specialization_and_quality.py \
      --run-output-dir "$RUN_OUTPUT_DIR" \
      --output-dir "$OUT" \
      --data-dir "${DATA_DIR:-data/real_subset_final}" \
      --checkpoint "$CHECKPOINT" \
      --stage-b-checkpoint "${STAGE_B_CHECKPOINT:?STAGE_B_CHECKPOINT is required}" \
      --stage-b-checkpoint-sha256 "${STAGE_B_CHECKPOINT_SHA256:?STAGE_B_CHECKPOINT_SHA256 is required}" \
      --feature-cache-dir "${FEATURE_CACHE_DIR:-$OUT/feature_cache}" \
      --base-model "${BASE_MODEL:-allenai/OLMoE-1B-7B-0924}" \
      --vision-model "${VISION_MODEL:-openai/clip-vit-base-patch32}" \
      --speech-model "${SPEECH_MODEL:-openai/whisper-base.en}" \
      --speech-target-space "${SPEECH_TARGET_SPACE:-olmoe_text_hidden}" \
      --capacity-factor "${CAPACITY_FACTOR:-4.0}" \
      --aux-coef "${AUX_COEF:-0.01}" \
      --max-length "${MAX_LENGTH:-512}" \
      --batch-size "${ROUTING_ANALYSIS_BATCH_SIZE:-4}" \
      --text-batches "${ROUTING_TEXT_BATCHES:-8}" \
      --modality-batches "${ROUTING_MODALITY_BATCHES:-8}" \
      --image-eval-count "${IMAGE_EVAL_SAMPLES:-250}" \
      --speech-eval-count "${SPEECH_EVAL_SAMPLES:-250}" \
      --image-prefix-tokens "${IMAGE_PREFIX_TOKENS:-50}" \
      --audio-prefix-tokens "${AUDIO_PREFIX_TOKENS:-64}" \
      --encoder-feature-tokens "${ENCODER_FEATURE_TOKENS:-100}" \
      --sample-rate "${SAMPLE_RATE:-16000}" \
      --qualitative-examples "${QUALITATIVE_EXAMPLES:-6}" \
      --max-new-tokens "${MAX_NEW_TOKENS:-28}" \
      --intervention-top-experts "${INTERVENTION_TOP_EXPERTS:-5}" \
      --intervention-examples "${INTERVENTION_EXAMPLES:-24}" \
      --intervention-text-blocks "${INTERVENTION_TEXT_BLOCKS:-16}" \
      "${IMAGE_MANIFEST_ARG[@]}" "${SPEECH_MANIFEST_ARG[@]}"
    ;;
  ablation-only)
    OUT="${OUT:?OUT is required for ablation-only}"
    SOURCE_OUTPUT_DIR="${SOURCE_OUTPUT_DIR:-$OUT}"
    RECOVER_EXISTING_ARG=()
    if [ "${RECOVER_EXISTING:-0}" = "1" ]; then
      RECOVER_EXISTING_ARG=(--skip-existing)
    fi
    python scripts/run_missing_ablations.py \
      --source-output-dir "$SOURCE_OUTPUT_DIR" \
      --output-dir "$OUT" \
      --ablation-steps "${ABLATION_STEPS:-300}" \
      --capacity-ablation-steps "${CAPACITY_ABLATION_STEPS:-300}" \
      --feature-cache-dir "${FEATURE_CACHE_DIR:-$OUT/feature_cache}" \
      --experiments "${ABLATION_EXPERIMENTS:-E4,E5}" \
      "${RECOVER_EXISTING_ARG[@]}"
    ;;
  e6-feasibility)
    python scripts/run_e6_feasibility.py \
      --source-output-dir "${RUN_OUTPUT_DIR:?RUN_OUTPUT_DIR is required for e6-feasibility}" \
      --output-dir "${OUT:?OUT is required for e6-feasibility}" \
      --checkpoint "${CHECKPOINT:?CHECKPOINT is required for e6-feasibility}" \
      --expected-checkpoint-sha256 "${EXPECTED_CHECKPOINT_SHA256:?EXPECTED_CHECKPOINT_SHA256 is required for e6-feasibility}" \
      --data-dir "${DATA_DIR:?DATA_DIR is required for e6-feasibility}" \
      --feature-cache-dir "${FEATURE_CACHE_DIR:?FEATURE_CACHE_DIR is required for e6-feasibility}"
    ;;
  top2-distill-real)
    OUT="${OUT:-outputs/top2_distill_real}"
    DATA_DIR="${DATA_DIR:-data/real_subset_clean_260708b}"
    TRAIN_LM_HEAD_ARG=()
    TRAIN_ROUTER_ARG=()
    TRAIN_GAMMA_ARG=()
    if [ "${TRAIN_LM_HEAD:-1}" = "1" ]; then
      TRAIN_LM_HEAD_ARG=(--train-lm-head)
    fi
    if [ "${TRAIN_ROUTER_GATES:-0}" = "1" ]; then
      TRAIN_ROUTER_ARG=(--train-router-gates)
    fi
    if [ "${TRAIN_GAMMA_SCALE:-0}" = "1" ]; then
      TRAIN_GAMMA_ARG=(--train-gamma-scale)
    fi
    python -m training.distill_olmoe_top2_real \
      --data-dir "$DATA_DIR" \
      --output-dir "$OUT" \
      --base-model "${BASE_MODEL:-allenai/OLMoE-1B-7B-0924}" \
      --seed "${SEED:-42}" \
      --teacher-top-k "${TEACHER_TOP_K:-8}" \
      --student-top-k "${STUDENT_TOP_K:-2}" \
      --capacity-factor "${CAPACITY_FACTOR:-6.0}" \
      --aux-coef "${AUX_COEF:-0.01}" \
      --gamma-min "${GAMMA_MIN:-0.25}" \
      --gamma-max "${GAMMA_MAX:-2.0}" \
      --max-length "${MAX_LENGTH:-512}" \
      --train-batch-size "${TRAIN_BATCH_SIZE:-4}" \
      --eval-batch-size "${EVAL_BATCH_SIZE:-8}" \
      --text-eval-blocks "${TEXT_EVAL_BLOCKS:-160}" \
      --distill-steps "${DISTILL_STEPS:-800}" \
      --learning-rate "${LEARNING_RATE:-0.00001}" \
      --router-learning-rate "${ROUTER_LEARNING_RATE:-0.00001}" \
      --gamma-learning-rate "${GAMMA_LEARNING_RATE:-0.0001}" \
      --weight-decay "${WEIGHT_DECAY:-0.0}" \
      --grad-clip "${GRAD_CLIP:-1.0}" \
      --distill-logit-coef "${DISTILL_LOGIT_COEF:-0.1}" \
      --distill-temperature "${DISTILL_TEMPERATURE:-2.0}" \
      --router-distill-coef "${ROUTER_DISTILL_COEF:-0.0}" \
      --router-distill-temperature "${ROUTER_DISTILL_TEMPERATURE:-2.0}" \
      --distill-hidden-coef "${DISTILL_HIDDEN_COEF:-0.0}" \
      --distill-hidden-layers "${DISTILL_HIDDEN_LAYERS:-last}" \
      --distill-hidden-mode "${DISTILL_HIDDEN_MODE:-cosine}" \
      --moe-reconstruction-coef "${MOE_RECONSTRUCTION_COEF:-0.0}" \
      --moe-reconstruction-layers "${MOE_RECONSTRUCTION_LAYERS:-all}" \
      --text-replay-coef "${TEXT_REPLAY_COEF:-0.0}" \
      --text-replay-manifest "${TEXT_REPLAY_MANIFEST:-}" \
      --student-k-curriculum "${STUDENT_K_CURRICULUM:-none}" \
      --resume-checkpoint "${RESUME_CHECKPOINT:-}" \
      --checkpoint-every-steps "${CHECKPOINT_EVERY_STEPS:-0}" \
      --log-every-steps "${LOG_EVERY_STEPS:-50}" \
      "${TRAIN_LM_HEAD_ARG[@]}" "${TRAIN_ROUTER_ARG[@]}" "${TRAIN_GAMMA_ARG[@]}"
    ;;
  integrity-tests)
    python -m unittest discover -s tests -p 'test_*.py' -v
    ;;
  sealed-data-prepare)
    SEALED_OUTPUT_DIR="${SEALED_OUTPUT_DIR:-data/sealed_eval_v1}"
    FORCE_ARG=()
    EXCLUDE_ARG=()
    if [ "${FORCE:-0}" = "1" ]; then
      FORCE_ARG=(--force)
    fi
    if [ -n "${EXCLUDE_INDEX:-}" ]; then
      EXCLUDE_ARG=(--exclude-index "$EXCLUDE_INDEX")
    fi
    python datasets/build_sealed_eval.py \
      --output-dir "$SEALED_OUTPUT_DIR" \
      --image-samples "${IMAGE_EVAL_SAMPLES:-250}" \
      --speech-samples "${SPEECH_EVAL_SAMPLES:-250}" \
      --speech-duration-seconds "${REAL_MAX_AUDIO_SECONDS:-6.0}" \
      --index-file "${SEALED_INDEX_FILE:-sealed_eval_index.jsonl}" \
      "${EXCLUDE_ARG[@]}" "${FORCE_ARG[@]}"
    ;;
  matched-ablation)
    python scripts/run_matched_ablations.py \
      --source-root "${SOURCE_ROOT:?SOURCE_ROOT is required}" \
      --output-root "${OUT:?OUT is required}" \
      --seed "${SEED:?SEED is required}"
    ;;
  functional-audit)
    FUNCTIONAL_ARGS=()
    if [ -n "${RUN_MANIFEST:-}" ]; then
      FUNCTIONAL_ARGS+=(--run-manifest "$RUN_MANIFEST")
    fi
    if [ -n "${GAMMA_JSON:-}" ]; then
      FUNCTIONAL_ARGS+=(--gamma-json "$GAMMA_JSON")
    fi
    python scripts/audit_functional_paths.py \
      --checkpoint "${CHECKPOINT:?CHECKPOINT is required}" \
      --data-dir "${DATA_DIR:-data/real_subset_clean_260708b}" \
      --output "${OUT:?OUT is required}" \
      --seed "${SEED:-42}" \
      "${FUNCTIONAL_ARGS[@]}"
    ;;
  matched-summary)
    python scripts/summarize_matched_ablations.py \
      --seed-roots "${MATCHED_ROOT_1:?MATCHED_ROOT_1 is required}" \
        "${MATCHED_ROOT_2:?MATCHED_ROOT_2 is required}" \
        "${MATCHED_ROOT_3:?MATCHED_ROOT_3 is required}" \
      --output-dir "${OUT:?OUT is required}"
    ;;
  checkpoint-provenance)
    python scripts/extract_checkpoint_provenance.py \
      --checkpoint "${CHECKPOINT:?CHECKPOINT is required}" \
      --run-manifest "${RUN_MANIFEST:?RUN_MANIFEST is required}" \
      --output-dir "${OUT:?OUT is required}"
    ;;
  representation-funnel)
    python scripts/freeze_evaluation_protocol.py \
      --verify "${PROTOCOL_PATH:?PROTOCOL_PATH is required}" \
      --verify-git-state
    python scripts/eval_representation_funnel.py \
      --checkpoint "${CHECKPOINT:?CHECKPOINT is required}" \
      --stage-b-checkpoint "${STAGE_B_CHECKPOINT:?STAGE_B_CHECKPOINT is required}" \
      --stage-b-checkpoint-sha256 "${STAGE_B_CHECKPOINT_SHA256:?STAGE_B_CHECKPOINT_SHA256 is required}" \
      --protocol-manifest "$PROTOCOL_PATH" \
      --gamma-json "${GAMMA_JSON:?GAMMA_JSON is required}" \
      --image-dev-manifest "${DEV_IMAGE_MANIFEST:?DEV_IMAGE_MANIFEST is required}" \
      --speech-dev-manifest "${DEV_SPEECH_MANIFEST:?DEV_SPEECH_MANIFEST is required}" \
      --image-test-manifest "${IMAGE_MANIFEST:?IMAGE_MANIFEST is required}" \
      --speech-test-manifest "${SPEECH_MANIFEST:?SPEECH_MANIFEST is required}" \
      --output-dir "${OUT:?OUT is required}" \
      --batch-size "${EVAL_BATCH_SIZE:-4}"
    ;;
  development-representation-diagnostics)
    python scripts/eval_development_representation_diagnostics.py \
      --checkpoint "${CHECKPOINT:?CHECKPOINT is required}" \
      --gamma-json "${GAMMA_JSON:?GAMMA_JSON is required}" \
      --image-dev-manifest "${DEV_IMAGE_MANIFEST:?DEV_IMAGE_MANIFEST is required}" \
      --speech-dev-manifest "${DEV_SPEECH_MANIFEST:?DEV_SPEECH_MANIFEST is required}" \
      --output-dir "${OUT:?OUT is required}" \
      --batch-size "${EVAL_BATCH_SIZE:-4}"
    ;;
  sealed-matrix-analysis)
    python scripts/analyze_sealed_matrix.py \
      --protocol "${PROTOCOL_PATH:?PROTOCOL_PATH is required}" \
      --matrix-root "${MATRIX_ROOT:?MATRIX_ROOT is required}" \
      --output-dir "${OUT:?OUT is required}" \
      --bootstrap-samples "${BOOTSTRAP_SAMPLES:-10000}" \
      --permutation-samples "${PERMUTATION_SAMPLES:-10000}" \
      --seed "${SEED:-20260709}"
    ;;
  evidence-verifier)
    python scripts/build_evaluation_result_manifest.py \
      --spec "${EVIDENCE_MANIFEST_SPEC:?EVIDENCE_MANIFEST_SPEC is required}" \
      --output "${EVIDENCE_MANIFEST_OUTPUT:?EVIDENCE_MANIFEST_OUTPUT is required}"
    ;;
  loss-reconcile)
    python scripts/reconcile_training_losses.py \
      --run-root "${RUN_OUTPUT_DIR:?RUN_OUTPUT_DIR is required}" \
      --window "${LOSS_WINDOW:-100}" \
      --output-dir "${OUT:-$RUN_OUTPUT_DIR/review_repair}"
    ;;

  full)
    DATA_DIR="${DATA_DIR:-data/real_subset}"
    if [ ! -f "$DATA_DIR/manifest.json" ]; then
      bash "$0" real-data-prepare
    fi
    bash "$0" real-required-runs
    ;;
  required-runs)
    OUT="${OUT:-outputs/required_runs}"
    python -m training.olmoe_required_runs \
      --output-dir "$OUT" \
      --base-model "${BASE_MODEL:-allenai/OLMoE-1B-7B-0924}" \
      --vision-model "${VISION_MODEL:-openai/clip-vit-base-patch32}" \
      --speech-model "${SPEECH_MODEL:-openai/whisper-tiny.en}" \
      --final-steps "${FINAL_STEPS:-20}" \
      --ablation-steps "${ABLATION_STEPS:-10}" \
      --capacity-factor "${CAPACITY_FACTOR:-1.25}" \
      --aux-coef "${AUX_COEF:-0.01}"
    ;;
  olmoe-probe)
    OUT="${OUT:-outputs/olmoe_probe}"
    python -m training.olmoe_short_run --mode probe --top-k-list "${TOP_K_LIST:-8,2}" --output-dir "$OUT"
    ;;
  olmoe-short-train)
    OUT="${OUT:-outputs/olmoe_short_train}"
    python -m training.olmoe_short_run --mode train --top-k "${TOP_K:-2}" --max-steps "${MAX_STEPS:-2}" --output-dir "$OUT"
    ;;
  bootstrap-env)
    bash scripts/bootstrap_env.sh
    ;;
  env-check)
    OUT="${OUT:-outputs/env_check}"
    python scripts/env_check.py
    ;;
  smoke)
    CONFIG="training/config_smoke.yaml"
    OUT="${OUT:-outputs/smoke}"
    python -m training.calibrate_top2 --config "$CONFIG" --output-dir "$OUT"
    python -m training.train --config "$CONFIG" --output-dir "$OUT"
    python -m evaluation.evaluate --config "$CONFIG" --output-dir "$OUT"
    python scripts/make_plots.py --output-dir "$OUT"
    ;;
  distill)
    CONFIG="training/config_smoke.yaml"
    OUT="${OUT:-outputs/smoke}"
    python -m training.distill_top2 --config "$CONFIG" --output-dir "$OUT"
    ;;
  synthetic-demo)
    CONFIG="${CONFIG:-training/config_top2_1week.yaml}"
    OUT="${OUT:-outputs/top2_1week}"
    python -m training.calibrate_top2 --config "$CONFIG" --output-dir "$OUT"
    python -m training.train --config "$CONFIG" --output-dir "$OUT"
    python -m evaluation.evaluate --config "$CONFIG" --output-dir "$OUT"
    python scripts/make_plots.py --output-dir "$OUT"
    ;;
  final)
    echo "final is disabled; freeze the protocol, then use scripts/submit_sealed_control_matrix.sh and scripts/submit_sealed_analysis.sh" >&2
    exit 2
    ;;
  *)
    echo "usage: bash run.sh [full|real-data-prepare|sealed-data-prepare|real-required-runs|integrity-tests|matched-ablation|matched-summary|functional-audit|checkpoint-provenance|representation-funnel|development-representation-diagnostics|sealed-matrix-analysis|evidence-verifier|loss-reconcile|ablation-only|e6-feasibility|specialization-analysis|text-eval-refresh|e3-refresh|conditional-eval|top2-distill-real|debug-e0|inspect-olmoe|bootstrap-env|env-check|required-runs|olmoe-probe|olmoe-short-train|smoke|distill|synthetic-demo]" >&2
    exit 2
    ;;
esac
