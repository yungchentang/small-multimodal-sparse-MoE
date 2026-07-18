#!/usr/bin/env bash
set -euo pipefail

# Development-only A3/C1/C2 screening with Stage B and Stage A initialization.
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$REPO_ROOT"

SOURCE_COMMIT_SHA="${SOURCE_COMMIT_SHA:?SOURCE_COMMIT_SHA is required}"
GIT=(git -c "safe.directory=$REPO_ROOT")
[ "$("${GIT[@]}" rev-parse HEAD)" = "$SOURCE_COMMIT_SHA" ] || {
  echo "source commit mismatch" >&2
  exit 2
}
if ! "${GIT[@]}" diff --quiet || ! "${GIT[@]}" diff --cached --quiet || [ -n "$("${GIT[@]}" ls-files --others --exclude-standard)" ]; then
  if [ "${DRY_RUN:-0}" != "1" ] || [ "${ALLOW_DIRTY_DRY_RUN:-0}" != "1" ]; then
    echo "selected-expert campaign requires a clean source worktree" >&2
    exit 2
  fi
fi

BASE_OUT="${BASE_OUT:-outputs/development_selected_expert_v1}"
DATA_DIR="${DATA_DIR:-data/real_subset_clean_260708b}"
DEVELOPMENT_SPLIT_MANIFEST="${DEVELOPMENT_SPLIT_MANIFEST:?DEVELOPMENT_SPLIT_MANIFEST is required}"
DEVELOPMENT_SPEECH_SOURCE_SHA256="${DEVELOPMENT_SPEECH_SOURCE_SHA256:?DEVELOPMENT_SPEECH_SOURCE_SHA256 is required}"
export DEVELOPMENT_SPEECH_SOURCE_SHA256
STAMP="${STAMP:-$(date +%y%m%d%H%M)}"
SEED="${SEED:-42}"
SCREEN_STEPS="${SCREEN_STEPS:-500}"
ALIGNMENT_PRETRAIN_STEPS="${ALIGNMENT_PRETRAIN_STEPS:-400}"
ONLY_RAW="${ONLY:-A0,A3}"
ONLY=",${ONLY_RAW},"
DRY_RUN="${DRY_RUN:-0}"

EXPERT_SELECTION_JSON="${EXPERT_SELECTION_JSON:?EXPERT_SELECTION_JSON is required}"
STAGE_B_CHECKPOINT="${STAGE_B_CHECKPOINT:?STAGE_B_CHECKPOINT is required}"
STAGE_B_CHECKPOINT_SHA256="${STAGE_B_CHECKPOINT_SHA256:?STAGE_B_CHECKPOINT_SHA256 is required}"
MULTIMODAL_INITIAL_CHECKPOINT="${MULTIMODAL_INITIAL_CHECKPOINT:?MULTIMODAL_INITIAL_CHECKPOINT is required}"
MULTIMODAL_INITIAL_CHECKPOINT_SHA256="${MULTIMODAL_INITIAL_CHECKPOINT_SHA256:?MULTIMODAL_INITIAL_CHECKPOINT_SHA256 is required}"
MULTIMODAL_INITIAL_MANIFEST="${MULTIMODAL_INITIAL_MANIFEST:?MULTIMODAL_INITIAL_MANIFEST is required}"

case "${BASE_OUT}:${DATA_DIR}:${EXPERT_SELECTION_JSON}:${STAGE_B_CHECKPOINT}:${MULTIMODAL_INITIAL_CHECKPOINT}:${MULTIMODAL_INITIAL_MANIFEST}" in
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

for path in "$DATA_DIR" "$EXPERT_SELECTION_JSON" "$STAGE_B_CHECKPOINT" "$MULTIMODAL_INITIAL_CHECKPOINT" "$MULTIMODAL_INITIAL_MANIFEST"; do
  [ -e "$path" ] || { echo "missing required path: $path" >&2; exit 2; }
done
IFS=',' read -r -a requested_arms <<<"$ONLY_RAW"
for arm in "${requested_arms[@]}"; do
  case "$arm" in
    A0|A3|C1|C2) ;;
    *) echo "unknown development arm: $arm" >&2; exit 2 ;;
  esac
done
[[ "$SCREEN_STEPS" =~ ^[0-9]+$ && "$ALIGNMENT_PRETRAIN_STEPS" =~ ^[0-9]+$ ]] || {
  echo "screen/alignment steps must be integers" >&2; exit 2;
}
[ "$ALIGNMENT_PRETRAIN_STEPS" -ge 1 ] && [ "$ALIGNMENT_PRETRAIN_STEPS" -lt "$SCREEN_STEPS" ] || {
  echo "alignment pretraining must be positive and shorter than the screen" >&2; exit 2;
}
if [ "${ALLOW_SMOKE:-0}" = "1" ]; then
  [ "$SCREEN_STEPS" -ge 1 ] && [ "$SCREEN_STEPS" -le 20 ] || {
    echo "smoke runs require 1-20 steps" >&2; exit 2;
  }
elif [ "$SCREEN_STEPS" -lt 500 ] || [ "$SCREEN_STEPS" -gt 1000 ]; then
  echo "screening runs require 500-1000 steps" >&2
  exit 2
fi

# arm|dynamic_bias_lr|train_router|selected_experts
SPECS=(
  "A0|0.0|0|0"
  "A3|0.0|0|1"
  "C1|0.001|0|1"
  "C2|0.0|1|1"
)

submit_arm() {
  local arm="$1" bias="$2" router="$3" selected="$4"
  [[ "$ONLY" == *",${arm},"* ]] || return 0
  local lower="${arm,,}"
  local out="${BASE_OUT}/${lower}_seed${SEED}"
  local job="sme-esft-${lower}-s${SEED}-${STAMP}"
  local selection_json=""
  [ "$selected" = "0" ] || selection_json="$EXPERT_SELECTION_JSON"
  [ ! -e "$out" ] || { echo "refusing overwrite: $out" >&2; return 1; }

  if [ "$DRY_RUN" = "1" ]; then
    printf 'arm=%s job=%s output=%s bias=%s router=%s selected=%s lm_head=0 stage_b=%s multimodal_scope=image source=%s\n' \
      "$arm" "$job" "$out" "$bias" "$router" "$selected" "$STAGE_B_CHECKPOINT" "$SOURCE_COMMIT_SHA"
    return 0
  fi

  env JOB_NAME="$job" OUT="$out" DATA_DIR="$DATA_DIR" DEVELOPMENT_SPLIT_MANIFEST="$DEVELOPMENT_SPLIT_MANIFEST" DEVELOPMENT_SPEECH_SOURCE_SHA256="$DEVELOPMENT_SPEECH_SOURCE_SHA256" SUBMIT_REPO_DIR="$REPO_ROOT" \
    GPU=1 CPU="${CPU:-8}" MEMORY="${MEMORY:-120G}" SEED="$SEED" \
    SOURCE_COMMIT_SHA="$SOURCE_COMMIT_SHA" FINAL_STEPS="$SCREEN_STEPS" \
    ALIGNMENT_PRETRAIN_STEPS="$ALIGNMENT_PRETRAIN_STEPS" ALIGNMENT_PRETRAIN_LOG_EVERY=25 \
    ALIGNMENT_PRETRAIN_MODALITIES=speech ABLATION_STEPS=0 CAPACITY_ABLATION_STEPS=0 \
    EXPERT_ABLATION_STEPS=0 POSTPROCESS_REQUIRED_RUNS=0 CAPACITY_FACTOR=8.0 AUX_COEF=0.02 \
    MODALITY_CYCLE=text,speech,speech TRAIN_BATCH_SIZE=2 EVAL_BATCH_SIZE=4 \
    TEXT_EVAL_BLOCKS=160 IMAGE_EVAL_SAMPLES=137 SPEECH_EVAL_SAMPLES=137 \
    RETRIEVAL_EVAL_SAMPLES=137 CONDITIONAL_EVAL_SAMPLES=137 CONDITIONAL_NEGATIVES=9 \
    CONDITIONAL_RANKING_NEGATIVES=9 CONDITIONAL_RANKING_NEGATIVE_MODE=stride \
    CONDITIONAL_RANKING_HARD_POOL_SIZE=512 CONDITIONAL_RANKING_TEMPERATURE=0.7 \
    IMAGE_PREFIX_TOKENS=50 AUDIO_PREFIX_TOKENS=64 ENCODER_FEATURE_TOKENS=100 \
    IMAGE_ALIGNMENT_TARGET=olmoe_caption_hidden IMAGE_BRIDGE_TYPE=linear_projector \
    AUDIO_BRIDGE_TYPE=attention_pool AUDIO_MAX_SECONDS=6.0 \
    TRAIN_ROUTER_GATES="$router" TRAIN_EXPERTS=0 TRAIN_LM_HEAD=0 \
    ALLOW_SELECTED_EXPERT_ROUTER_TUNING="$router" \
    ROUTER_LEARNING_RATE=0.000002 DYNAMIC_EXPERT_BIAS_LR="$bias" \
    EXPERT_SELECTION_JSON="$selection_json" EXPERT_SELECTION_METHOD=ESFT-Gate \
    EXPERT_UPDATE_MODE=full EXPERT_ANCHOR_COEFFICIENT=0.01 EXPERT_LEARNING_RATE=0.000001 \
    STAGE_B_CHECKPOINT="$STAGE_B_CHECKPOINT" STAGE_B_CHECKPOINT_SHA256="$STAGE_B_CHECKPOINT_SHA256" \
    MULTIMODAL_INITIAL_CHECKPOINT="$MULTIMODAL_INITIAL_CHECKPOINT" \
    MULTIMODAL_INITIAL_CHECKPOINT_SHA256="$MULTIMODAL_INITIAL_CHECKPOINT_SHA256" \
    MULTIMODAL_INITIAL_MANIFEST="$MULTIMODAL_INITIAL_MANIFEST" \
    MULTIMODAL_INITIALIZATION_SCOPE=image SPEECH_UNFREEZE_LAST_BLOCKS=1 \
    SPEECH_UNFREEZE_LAYER_NORM=1 SPEECH_ENCODER_LEARNING_RATE=0.000005 \
    LEARNING_RATE=0.0005 RETRIEVAL_HEAD_LEARNING_RATE=0.0 LM_HEAD_LEARNING_RATE=0.00001 \
    CONTRASTIVE_COEF=0.2 CONTRASTIVE_NEGATIVES=128 CONTRASTIVE_TEMPERATURE=0.07 \
    SPEECH_CONTRASTIVE_COEF=0.2 SPEECH_CONTRASTIVE_TEMPERATURE=0.04 \
    SPEECH_CENTER_POSITIVE_WEIGHT=5.0 SPEECH_RAW_POSITIVE_WEIGHT=0.0 \
    SPEECH_CONDITIONAL_RANKING_COEF=3.0 WEIGHT_DECAY=0.0 GRAD_CLIP=5.0 \
    LOG_EVERY_STEPS=25 SAVE_EVERY_STEPS="$SCREEN_STEPS" \
    bash scripts/submit_e3_candidate_runai.sh
}

for spec in "${SPECS[@]}"; do
  IFS='|' read -r arm bias router selected <<<"$spec"
  submit_arm "$arm" "$bias" "$router" "$selected"
done
