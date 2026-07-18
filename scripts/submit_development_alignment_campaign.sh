#!/usr/bin/env bash
set -euo pipefail

# Development-only Stage A bridge/alignment screening. Each arm uses one GPU.
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$REPO_ROOT"
GIT=(git -c "safe.directory=$REPO_ROOT")

SOURCE_COMMIT_SHA="${SOURCE_COMMIT_SHA:?SOURCE_COMMIT_SHA is required}"
ACTUAL_SOURCE_SHA="$("${GIT[@]}" rev-parse HEAD)"
if [ "$ACTUAL_SOURCE_SHA" != "$SOURCE_COMMIT_SHA" ]; then
  echo "source commit mismatch: expected $SOURCE_COMMIT_SHA, found $ACTUAL_SOURCE_SHA" >&2
  exit 2
fi
if ! "${GIT[@]}" diff --quiet || ! "${GIT[@]}" diff --cached --quiet || [ -n "$("${GIT[@]}" ls-files --others --exclude-standard)" ]; then
  if [ "${DRY_RUN:-0}" != "1" ] || [ "${ALLOW_DIRTY_DRY_RUN:-0}" != "1" ]; then
    echo "alignment campaign requires a clean source worktree" >&2
    exit 2
  fi
fi

DATA_DIR="${DATA_DIR:-data/real_subset_clean_260708b}"
DEVELOPMENT_SPLIT_MANIFEST="${DEVELOPMENT_SPLIT_MANIFEST:?DEVELOPMENT_SPLIT_MANIFEST is required}"
DEVELOPMENT_SPEECH_SOURCE_SHA256="${DEVELOPMENT_SPEECH_SOURCE_SHA256:?DEVELOPMENT_SPEECH_SOURCE_SHA256 is required}"
export DEVELOPMENT_SPEECH_SOURCE_SHA256
BASE_OUT="${BASE_OUT:-outputs/development_alignment}"
STAMP="${STAMP:-$(date +%y%m%d%H%M)}"
SEED="${SEED:-42}"
SCREEN_STEPS="${SCREEN_STEPS:-500}"
ALIGNMENT_PRETRAIN_STEPS="${ALIGNMENT_PRETRAIN_STEPS:-400}"
ONLY=",${ONLY:-I_QUERY,I_LINEAR,I_NORM,S_QUERY6,S_ATTN6,S_TEMP6,S_ATTN10,S_ATTN_LAST1_LN},"
DRY_RUN="${DRY_RUN:-0}"
SPEECH_TEACHER_BANK_BATCH_SIZE="${SPEECH_TEACHER_BANK_BATCH_SIZE:-64}"

case "$DATA_DIR:$DEVELOPMENT_SPLIT_MANIFEST:$BASE_OUT" in
  *sealed*) echo "refusing sealed path: $DATA_DIR:$BASE_OUT" >&2; exit 2 ;;
esac

reject_symlink_path_components() {
  local value="$1" absolute current component
  local -a components
  if [[ "$value" = /* ]]; then
    absolute="$value"
  else
    absolute="$PWD/$value"
  fi
  current="/"
  IFS='/' read -r -a components <<<"$absolute"
  for component in "${components[@]}"; do
    case "$component" in
      ""|.) continue ;;
      ..) current="${current%/*}"; [ -n "$current" ] || current="/" ;;
      *)
        if [ "$current" = "/" ]; then
          current="/$component"
        else
          current="$current/$component"
        fi
        [ ! -L "$current" ] || {
          echo "refusing unsafe DATA_DIR symlink component: $current" >&2
          exit 2
        }
        ;;
    esac
  done
}

reject_symlink_path_components "$DATA_DIR"
[ -d "$DATA_DIR" ] || { echo "missing development data: $DATA_DIR" >&2; exit 2; }
DATA_DIR="$(realpath -- "$DATA_DIR")"
[[ "$DATA_DIR" != *sealed* ]] || { echo "refusing sealed path: $DATA_DIR" >&2; exit 2; }
[ -f "$DEVELOPMENT_SPLIT_MANIFEST" ] && [ ! -L "$DEVELOPMENT_SPLIT_MANIFEST" ] || {
  echo "missing or unsafe development split manifest" >&2
  exit 2
}
DEVELOPMENT_SPLIT_MANIFEST="$(realpath -- "$DEVELOPMENT_SPLIT_MANIFEST")"
if [ "$DRY_RUN" != "1" ]; then
  python3 -c 'import sys; from scripts.audit_requirements import validate_development_multimodal_runtime_manifest as validate; result = validate(sys.argv[1], expected_source_commit_sha=sys.argv[2]); assert result["passed"], result["errors"]' "$DEVELOPMENT_SPLIT_MANIFEST" "$SOURCE_COMMIT_SHA"
fi
[[ "$SCREEN_STEPS" =~ ^[0-9]+$ && "$ALIGNMENT_PRETRAIN_STEPS" =~ ^[0-9]+$ ]] || {
  echo "screen/alignment steps must be integers" >&2; exit 2;
}
[[ "$SPEECH_TEACHER_BANK_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || {
  echo "speech teacher bank batch size must be a positive integer" >&2; exit 2;
}
if [ "${ALLOW_SMOKE:-0}" = "1" ]; then
  [ "$SCREEN_STEPS" -ge 2 ] && [ "$SCREEN_STEPS" -le 20 ] || {
    echo "smoke runs require 2-20 steps" >&2; exit 2;
  }
elif [ "$SCREEN_STEPS" -lt 500 ] || [ "$SCREEN_STEPS" -gt 1000 ]; then
  echo "screening runs require 500-1000 steps" >&2; exit 2
fi
[ "$ALIGNMENT_PRETRAIN_STEPS" -ge 1 ] && [ "$ALIGNMENT_PRETRAIN_STEPS" -lt "$SCREEN_STEPS" ] || {
  echo "alignment pretraining must be positive and shorter than the screen" >&2; exit 2;
}

COMMON_GIT_DIR="$("${GIT[@]}" rev-parse --path-format=absolute --git-common-dir)"
DEFAULT_RUNAI_ENV="$(dirname "$COMMON_GIT_DIR")/.env.runai"
if [ -z "${RUNAI_ENV_FILE:-}" ] && [ -f "$DEFAULT_RUNAI_ENV" ]; then
  export RUNAI_ENV_FILE="$DEFAULT_RUNAI_ENV"
fi

# arm|modality|image_bridge|audio_bridge|feature_tokens|image_prefix|audio_prefix|audio_seconds|speech_blocks|speech_ln
SPECS=(
  "I_QUERY|image|query_resampler|query_resampler|50|50|50|6.0|0|0"
  "I_LINEAR|image|linear_projector|query_resampler|50|50|50|6.0|0|0"
  "I_NORM|image|linear_projector_norm|query_resampler|50|50|50|6.0|0|0"
  "S_QUERY6|speech|query_resampler|query_resampler|100|50|64|6.0|0|0"
  "S_ATTN6|speech|query_resampler|attention_pool|100|50|64|6.0|0|0"
  "S_TEMP6|speech|query_resampler|temporal_resample|100|50|64|6.0|0|0"
  "S_ATTN10|speech|query_resampler|attention_pool|100|50|64|10.0|0|0"
  "S_ATTN_LAST1_LN|speech|query_resampler|attention_pool|100|50|64|6.0|1|1"
)

submit_arm() {
  local arm="$1" modality="$2" image_bridge="$3" audio_bridge="$4"
  local feature_tokens="$5" image_prefix="$6" audio_prefix="$7" audio_seconds="$8"
  local speech_blocks="$9" speech_ln="${10}"
  local lower="${arm,,}"
  local out="$BASE_OUT/${lower}_seed${SEED}"
  local job="sme-a-${lower//_/-}-s${SEED}-${STAMP}"
  [[ "$ONLY" == *",${arm},"* ]] || return 0
  [ ! -e "$out" ] || { echo "refusing overwrite: $out" >&2; return 1; }
  [ "${#job}" -le 55 ] || { echo "job name too long: $job" >&2; return 1; }

  if [ "$DRY_RUN" = "1" ]; then
    printf '%s\n' "arm=$arm job=$job modality=$modality image_bridge=$image_bridge audio_bridge=$audio_bridge feature_tokens=$feature_tokens image_prefix=$image_prefix audio_prefix=$audio_prefix audio_seconds=$audio_seconds speech_blocks=$speech_blocks speech_ln=$speech_ln data_dir=$DATA_DIR development_split_manifest=$DEVELOPMENT_SPLIT_MANIFEST source=$SOURCE_COMMIT_SHA"
    return 0
  fi

  env JOB_NAME="$job" OUT="$out" DATA_DIR="$DATA_DIR" DEVELOPMENT_SPLIT_MANIFEST="$DEVELOPMENT_SPLIT_MANIFEST" SUBMIT_REPO_DIR="$REPO_ROOT" \
    GPU="${GPU:-1}" CPU="${CPU:-8}" MEMORY="${MEMORY:-120G}" SEED="$SEED" \
    SOURCE_COMMIT_SHA="$SOURCE_COMMIT_SHA" FINAL_STEPS="$SCREEN_STEPS" \
    ALIGNMENT_PRETRAIN_STEPS="$ALIGNMENT_PRETRAIN_STEPS" \
    ALIGNMENT_PRETRAIN_MODALITIES="$modality" MODALITY_CYCLE="$modality" \
    ABLATION_STEPS=0 CAPACITY_ABLATION_STEPS=0 EXPERT_ABLATION_STEPS=0 \
    POSTPROCESS_REQUIRED_RUNS=0 CAPACITY_FACTOR=8.0 AUX_COEF=0.02 \
    TRAIN_BATCH_SIZE=1 EVAL_BATCH_SIZE=1 \
    SPEECH_TEACHER_BANK_BATCH_SIZE="$SPEECH_TEACHER_BANK_BATCH_SIZE" TEXT_EVAL_BLOCKS=160 \
    IMAGE_EVAL_SAMPLES=137 SPEECH_EVAL_SAMPLES=137 RETRIEVAL_EVAL_SAMPLES=137 \
    CONDITIONAL_EVAL_SAMPLES=137 CONDITIONAL_NEGATIVES=9 CONDITIONAL_BATCH_SIZE=1 \
    IMAGE_ALIGNMENT_TARGET=olmoe_caption_hidden SPEECH_TARGET_SPACE=olmoe_text_hidden \
    IMAGE_BRIDGE_TYPE="$image_bridge" AUDIO_BRIDGE_TYPE="$audio_bridge" \
    IMAGE_PREFIX_TOKENS="$image_prefix" AUDIO_PREFIX_TOKENS="$audio_prefix" \
    ENCODER_FEATURE_TOKENS="$feature_tokens" AUDIO_MAX_SECONDS="$audio_seconds" \
    SPEECH_UNFREEZE_LAST_BLOCKS="$speech_blocks" SPEECH_UNFREEZE_LAYER_NORM="$speech_ln" \
    SPEECH_ENCODER_LEARNING_RATE=0.000005 TRAIN_ROUTER_GATES=0 TRAIN_EXPERTS=0 \
    TRAIN_LM_HEAD=0 LEARNING_RATE=0.0005 RETRIEVAL_HEAD_LEARNING_RATE=0.001 \
    CONTRASTIVE_COEF=0.2 IMAGE_CONTRASTIVE_COEF=0.2 SPEECH_CONTRASTIVE_COEF=0.2 \
    CONTRASTIVE_NEGATIVES=256 WEIGHT_DECAY=0.0 GRAD_CLIP=5.0 \
    LOG_EVERY_STEPS=25 ALIGNMENT_PRETRAIN_LOG_EVERY=25 SAVE_EVERY_STEPS="$SCREEN_STEPS" \
    bash scripts/submit_e3_candidate_runai.sh
}

for spec in "${SPECS[@]}"; do
  IFS='|' read -r arm modality image_bridge audio_bridge feature_tokens image_prefix audio_prefix audio_seconds speech_blocks speech_ln <<<"$spec"
  submit_arm "$arm" "$modality" "$image_bridge" "$audio_bridge" "$feature_tokens" "$image_prefix" "$audio_prefix" "$audio_seconds" "$speech_blocks" "$speech_ln"
done
