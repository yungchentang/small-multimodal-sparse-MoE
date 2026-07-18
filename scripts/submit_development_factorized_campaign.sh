#!/usr/bin/env bash
set -euo pipefail

# Development-only, single-factor A0-A5 screening campaign. A4 is a
# fail-closed LoRA fallback until an expert-LoRA path is implemented.
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
    echo "development campaign requires a clean source worktree" >&2
    exit 2
  fi
fi

BASE_OUT="${BASE_OUT:-outputs/development_factorized}"
DATA_DIR="${DATA_DIR:-data/real_subset_clean_260708b}"
DEVELOPMENT_SPLIT_MANIFEST="${DEVELOPMENT_SPLIT_MANIFEST:?DEVELOPMENT_SPLIT_MANIFEST is required}"
DEVELOPMENT_SPEECH_SOURCE_SHA256="${DEVELOPMENT_SPEECH_SOURCE_SHA256:?DEVELOPMENT_SPEECH_SOURCE_SHA256 is required}"
export DEVELOPMENT_SPEECH_SOURCE_SHA256
STAMP="${STAMP:-$(date +%y%m%d%H%M)}"
SEED="${SEED:-42}"
SCREEN_STEPS="${SCREEN_STEPS:-500}"
ALIGNMENT_PRETRAIN_STEPS="${ALIGNMENT_PRETRAIN_STEPS:-100}"
ONLY=",${ONLY:-A0,A1,A2,A3,A5},"
DRY_RUN="${DRY_RUN:-0}"

case "${BASE_OUT}:${DATA_DIR}:${EXPERT_SELECTION_JSON:-}" in
  *sealed*|*synthetic*) echo "refusing sealed/synthetic path" >&2; exit 2 ;;
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
  IFS="/" read -r -a components <<<"$absolute"
  for component in "${components[@]}"; do
    case "$component" in
      ""|.) continue ;;
      ..) current="${current%/*}"; [ -n "$current" ] || current="/" ;;
      *)
        if [ "$current" = "/" ]; then current="/$component"; else current="$current/$component"; fi
        [ ! -L "$current" ] || {
          echo "refusing unsafe DATA_DIR symlink component: $current" >&2
          exit 2
        }
        ;;
    esac
  done
}

reject_symlink_path_components "$DATA_DIR"
[ -d "$DATA_DIR" ] && [ -f "$DATA_DIR/manifest.json" ] || {
  echo "missing or unsafe real development data: $DATA_DIR" >&2
  exit 2
}
DATA_DIR="$(realpath -- "$DATA_DIR")"
[[ "$DATA_DIR" != *sealed* && "$DATA_DIR" != *synthetic* ]] || {
  echo "refusing sealed/synthetic canonical data path: $DATA_DIR" >&2
  exit 2
}
[ -f "$DEVELOPMENT_SPLIT_MANIFEST" ] && [ ! -L "$DEVELOPMENT_SPLIT_MANIFEST" ] || {
  echo "missing or unsafe development split manifest" >&2
  exit 2
}
[[ "$DEVELOPMENT_SPLIT_MANIFEST" = /* ]] && [ "$(realpath -- "$DEVELOPMENT_SPLIT_MANIFEST")" = "$DEVELOPMENT_SPLIT_MANIFEST" ] || {
  echo "development split manifest must be an exact canonical absolute path" >&2
  exit 2
}
if [ "$DRY_RUN" != "1" ]; then
  python3 -c 'import sys; from scripts.audit_requirements import validate_development_multimodal_runtime_manifest as validate; result = validate(sys.argv[1], expected_source_commit_sha=sys.argv[2]); assert result["passed"], result["errors"]' "$DEVELOPMENT_SPLIT_MANIFEST" "$SOURCE_COMMIT_SHA"
fi

[[ "$SCREEN_STEPS" =~ ^[0-9]+$ && "$ALIGNMENT_PRETRAIN_STEPS" =~ ^[0-9]+$ ]] || {
  echo "screen/alignment steps must be integers" >&2; exit 2;
}
if [ "${ALLOW_SMOKE:-0}" = "1" ]; then
  [ "$SCREEN_STEPS" -ge 1 ] && [ "$SCREEN_STEPS" -le 20 ] || {
    echo "smoke runs require 1-20 steps" >&2; exit 2;
  }
elif [ "$SCREEN_STEPS" -lt 500 ] || [ "$SCREEN_STEPS" -gt 1000 ]; then
  echo "screening runs require 500-1000 steps" >&2; exit 2
fi
[ "$ALIGNMENT_PRETRAIN_STEPS" -ge 1 ] && [ "$ALIGNMENT_PRETRAIN_STEPS" -lt "$SCREEN_STEPS" ] || {
  echo "alignment pretraining must be positive and shorter than the screen" >&2; exit 2;
}

# arm|dynamic_bias_lr|train_router|selection_method|speech_blocks|speech_ln|expert_mode
SPECS=(
  "A0|0.0|0|none|0|0|none"
  "A1|0.001|0|none|0|0|none"
  "A2|0.0|1|none|0|0|none"
  "A3|0.0|0|ESFT-Gate|0|0|full"
  "A4|0.0|0|ESFT-Gate|0|0|lora"
  "A5|0.0|0|none|1|1|none"
)

submit_arm() {
  local arm="$1" bias="$2" router="$3" selection_method="$4"
  local speech_blocks="$5" speech_ln="$6" expert_mode="$7"
  local lower="${arm,,}" out="${BASE_OUT}/${arm,,}_seed${SEED}"
  local job="sme-dev-${lower}-s${SEED}-${STAMP}"
  [[ "$ONLY" == *",${arm},"* ]] || return 0
  [ ! -e "$out" ] || { echo "refusing overwrite: $out" >&2; return 1; }
  [ "${#job}" -le 55 ] || { echo "job name too long: $job" >&2; return 1; }

  local selection_json=""
  if [ "$selection_method" != "none" ]; then
    selection_json="${EXPERT_SELECTION_JSON:?EXPERT_SELECTION_JSON is required for A3/A4}"
    [ -s "$selection_json" ] || { echo "missing expert selection: $selection_json" >&2; return 1; }
  fi
  if [ "$expert_mode" = "lora" ]; then
    echo "A4 expert LoRA is unavailable and fails closed; implement and test it only if A3 full selected-expert training OOMs" >&2
    return 2
  fi

  if [ "$DRY_RUN" = "1" ]; then
    printf '%s\n' "arm=$arm job=$job output=$out bias=$bias router=$router selection=$selection_method speech_blocks=$speech_blocks speech_ln=$speech_ln expert_mode=$expert_mode source=$SOURCE_COMMIT_SHA"
    return 0
  fi

  env JOB_NAME="$job" OUT="$out" DATA_DIR="$DATA_DIR" DEVELOPMENT_SPLIT_MANIFEST="$DEVELOPMENT_SPLIT_MANIFEST" DEVELOPMENT_SPEECH_SOURCE_SHA256="$DEVELOPMENT_SPEECH_SOURCE_SHA256" SUBMIT_REPO_DIR="$REPO_ROOT" \
    GPU="${GPU:-1}" CPU="${CPU:-8}" MEMORY="${MEMORY:-120G}" SEED="$SEED" \
    SOURCE_COMMIT_SHA="$SOURCE_COMMIT_SHA" FINAL_STEPS="$SCREEN_STEPS" \
    ALIGNMENT_PRETRAIN_STEPS="$ALIGNMENT_PRETRAIN_STEPS" ABLATION_STEPS=0 \
    CAPACITY_ABLATION_STEPS=0 EXPERT_ABLATION_STEPS=0 POSTPROCESS_REQUIRED_RUNS=0 \
    CAPACITY_FACTOR=8.0 AUX_COEF=0.02 MODALITY_CYCLE=text,text,image,speech \
    TRAIN_BATCH_SIZE=4 EVAL_BATCH_SIZE=8 TEXT_EVAL_BLOCKS=160 \
    IMAGE_EVAL_SAMPLES=137 SPEECH_EVAL_SAMPLES=137 RETRIEVAL_EVAL_SAMPLES=137 \
    CONDITIONAL_EVAL_SAMPLES=137 IMAGE_PREFIX_TOKENS=50 AUDIO_PREFIX_TOKENS=64 \
    ENCODER_FEATURE_TOKENS=100 IMAGE_ALIGNMENT_TARGET=olmoe_caption_hidden \
    IMAGE_BRIDGE_TYPE=query_resampler AUDIO_BRIDGE_TYPE=query_resampler \
    AUDIO_MAX_SECONDS=6.0 TRAIN_ROUTER_GATES="$router" TRAIN_EXPERTS=0 TRAIN_LM_HEAD=1 \
    ROUTER_LEARNING_RATE=0.00001 DYNAMIC_EXPERT_BIAS_LR="$bias" \
    EXPERT_SELECTION_JSON="$selection_json" EXPERT_SELECTION_METHOD="$selection_method" \
    EXPERT_UPDATE_MODE="$expert_mode" EXPERT_ANCHOR_COEFFICIENT=0.01 \
    SPEECH_UNFREEZE_LAST_BLOCKS="$speech_blocks" SPEECH_UNFREEZE_LAYER_NORM="$speech_ln" \
    SPEECH_ENCODER_LEARNING_RATE=0.000005 LEARNING_RATE=0.0005 \
    RETRIEVAL_HEAD_LEARNING_RATE=0.0015 LM_HEAD_LEARNING_RATE=0.00001 \
    EXPERT_LEARNING_RATE=0.000001 WEIGHT_DECAY=0.0 GRAD_CLIP=5.0 \
    LOG_EVERY_STEPS=25 SAVE_EVERY_STEPS="$SCREEN_STEPS" \
    bash scripts/submit_e3_candidate_runai.sh
}

for spec in "${SPECS[@]}"; do
  IFS='|' read -r arm bias router selection_method speech_blocks speech_ln expert_mode <<<"$spec"
  submit_arm "$arm" "$bias" "$router" "$selection_method" "$speech_blocks" "$speech_ln" "$expert_mode"
done
