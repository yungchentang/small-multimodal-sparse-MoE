#!/usr/bin/env bash
set -euo pipefail

# One-GPU development A0+KD arm matched to the provenance-aware A0 configuration.
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$REPO_ROOT"
GIT=(git -c "safe.directory=$REPO_ROOT")

SOURCE_COMMIT_SHA="${SOURCE_COMMIT_SHA:?SOURCE_COMMIT_SHA is required}"
[ "${#SOURCE_COMMIT_SHA}" -eq 40 ] && [[ "$SOURCE_COMMIT_SHA" =~ ^[0-9a-fA-F]+$ ]] || {
  echo "SOURCE_COMMIT_SHA must be an exact 40-character hex commit" >&2
  exit 2
}
[ "$("${GIT[@]}" rev-parse HEAD)" = "$SOURCE_COMMIT_SHA" ] || {
  echo "source commit mismatch" >&2
  exit 2
}
if ! "${GIT[@]}" diff --quiet || ! "${GIT[@]}" diff --cached --quiet || [ -n "$("${GIT[@]}" ls-files --others --exclude-standard)" ]; then
  echo "speech KD campaign requires a clean source worktree" >&2
  exit 2
fi

DATA_DIR="${DATA_DIR:-data/real_subset_clean_260708b}"
DEVELOPMENT_SPLIT_MANIFEST="${DEVELOPMENT_SPLIT_MANIFEST:?DEVELOPMENT_SPLIT_MANIFEST is required}"
DEVELOPMENT_SPEECH_SOURCE_SHA256="${DEVELOPMENT_SPEECH_SOURCE_SHA256:?DEVELOPMENT_SPEECH_SOURCE_SHA256 is required}"
export DEVELOPMENT_SPEECH_SOURCE_SHA256
BASE_OUT="${BASE_OUT:-outputs/development_speech_kd}"
STAMP="${STAMP:-$(date +%y%m%d%H%M)}"
SEED="${SEED:-42}"
MAIN_STEPS="${MAIN_STEPS:-500}"
ALIGNMENT_PRETRAIN_STEPS="${ALIGNMENT_PRETRAIN_STEPS:-400}"
SPEECH_BEHAVIOR_KL_COEF="${SPEECH_BEHAVIOR_KL_COEF:-1.0}"
SPEECH_BEHAVIOR_KL_TEMPERATURE="${SPEECH_BEHAVIOR_KL_TEMPERATURE:-2.0}"
DRY_RUN="${DRY_RUN:-0}"

STAGE_B_CHECKPOINT="${STAGE_B_CHECKPOINT:?STAGE_B_CHECKPOINT is required}"
STAGE_B_CHECKPOINT_SHA256="${STAGE_B_CHECKPOINT_SHA256:?STAGE_B_CHECKPOINT_SHA256 is required}"
MULTIMODAL_INITIAL_CHECKPOINT="${MULTIMODAL_INITIAL_CHECKPOINT:?MULTIMODAL_INITIAL_CHECKPOINT is required}"
MULTIMODAL_INITIAL_CHECKPOINT_SHA256="${MULTIMODAL_INITIAL_CHECKPOINT_SHA256:?MULTIMODAL_INITIAL_CHECKPOINT_SHA256 is required}"
MULTIMODAL_INITIAL_MANIFEST="${MULTIMODAL_INITIAL_MANIFEST:?MULTIMODAL_INITIAL_MANIFEST is required}"

case "${BASE_OUT}:${DATA_DIR}:${STAGE_B_CHECKPOINT}:${MULTIMODAL_INITIAL_CHECKPOINT}:${MULTIMODAL_INITIAL_MANIFEST}" in
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

for path in "$STAGE_B_CHECKPOINT" "$MULTIMODAL_INITIAL_CHECKPOINT" "$MULTIMODAL_INITIAL_MANIFEST"; do
  [ -f "$path" ] && [ ! -L "$path" ] || {
    echo "missing or unsafe required file: $path" >&2
    exit 2
  }
done
for hash in "$STAGE_B_CHECKPOINT_SHA256" "$MULTIMODAL_INITIAL_CHECKPOINT_SHA256"; do
  [ "${#hash}" -eq 64 ] && [[ "$hash" =~ ^[0-9a-fA-F]+$ ]] || {
    echo "checkpoint hashes must be exact 64-character SHA-256 values" >&2
    exit 2
  }
done
[ "$(sha256sum "$STAGE_B_CHECKPOINT" | awk '{print $1}')" = "${STAGE_B_CHECKPOINT_SHA256,,}" ] || {
  echo "Stage B checkpoint SHA-256 mismatch" >&2
  exit 2
}
[ "$(sha256sum "$MULTIMODAL_INITIAL_CHECKPOINT" | awk '{print $1}')" = "${MULTIMODAL_INITIAL_CHECKPOINT_SHA256,,}" ] || {
  echo "Stage A checkpoint SHA-256 mismatch" >&2
  exit 2
}
[[ "$MAIN_STEPS" =~ ^[0-9]+$ && "$ALIGNMENT_PRETRAIN_STEPS" =~ ^[0-9]+$ ]] || {
  echo "main/alignment steps must be integers" >&2
  exit 2
}
if [ "${ALLOW_SMOKE:-0}" = "1" ]; then
  [ "$MAIN_STEPS" -ge 2 ] && [ "$MAIN_STEPS" -le 20 ] || {
    echo "smoke runs require 2-20 main steps" >&2
    exit 2
  }
else
  [ "$MAIN_STEPS" -eq 500 ] && [ "$ALIGNMENT_PRETRAIN_STEPS" -eq 400 ] || {
    echo "speech KD campaign requires exactly 500 main + 400 alignment steps" >&2
    exit 2
  }
fi
[ "$ALIGNMENT_PRETRAIN_STEPS" -ge 1 ] && [ "$ALIGNMENT_PRETRAIN_STEPS" -lt "$MAIN_STEPS" ] || {
  echo "alignment pretraining must be positive and shorter than main training" >&2
  exit 2
}
[[ "$SPEECH_BEHAVIOR_KL_COEF" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || {
  echo "speech behavior KL coefficient must be a positive number" >&2
  exit 2
}
[[ "$SPEECH_BEHAVIOR_KL_TEMPERATURE" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || {
  echo "speech behavior KL temperature must be a positive number" >&2
  exit 2
}
awk -v value="$SPEECH_BEHAVIOR_KL_COEF" 'BEGIN { exit !(value > 0) }' || {
  echo "speech behavior KL coefficient must be positive for the KD arm" >&2
  exit 2
}
awk -v value="$SPEECH_BEHAVIOR_KL_TEMPERATURE" 'BEGIN { exit !(value > 0) }' || {
  echo "speech behavior KL temperature must be positive" >&2
  exit 2
}

OUT="${BASE_OUT}/a0_kd_seed${SEED}"
JOB_NAME="sme-speech-kd-s${SEED}-${STAMP}"
[ ! -e "$OUT" ] || { echo "refusing overwrite: $OUT" >&2; exit 2; }
[ "${#JOB_NAME}" -le 55 ] || { echo "job name too long: $JOB_NAME" >&2; exit 2; }

if [ "$DRY_RUN" = "1" ]; then
  printf 'arm=A0_KD job=%s output=%s gpu=1 main_steps=%s alignment_steps=%s kl_coef=%s kl_temperature=%s router=0 experts=0 lm_head=0 stage_b_sha256=%s stage_a_sha256=%s stage_a_manifest=%s multimodal_scope=image source=%s\n' \
    "$JOB_NAME" "$OUT" "$MAIN_STEPS" "$ALIGNMENT_PRETRAIN_STEPS" \
    "$SPEECH_BEHAVIOR_KL_COEF" "$SPEECH_BEHAVIOR_KL_TEMPERATURE" \
    "$STAGE_B_CHECKPOINT_SHA256" "$MULTIMODAL_INITIAL_CHECKPOINT_SHA256" \
    "$MULTIMODAL_INITIAL_MANIFEST" "$SOURCE_COMMIT_SHA"
  exit 0
fi

env JOB_NAME="$JOB_NAME" OUT="$OUT" DATA_DIR="$DATA_DIR" DEVELOPMENT_SPLIT_MANIFEST="$DEVELOPMENT_SPLIT_MANIFEST" DEVELOPMENT_SPEECH_SOURCE_SHA256="$DEVELOPMENT_SPEECH_SOURCE_SHA256" SUBMIT_REPO_DIR="$REPO_ROOT" \
  GPU=1 CPU="${CPU:-8}" MEMORY="${MEMORY:-120G}" SEED="$SEED" \
  SOURCE_COMMIT_SHA="$SOURCE_COMMIT_SHA" FINAL_STEPS="$MAIN_STEPS" \
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
  TRAIN_ROUTER_GATES=0 TRAIN_EXPERTS=0 TRAIN_LM_HEAD=0 DYNAMIC_EXPERT_BIAS_LR=0.0 \
  ROUTER_LEARNING_RATE=0.000002 ALLOW_SELECTED_EXPERT_ROUTER_TUNING=0 \
  EXPERT_ANCHOR_COEFFICIENT=0.01 EXPERT_LEARNING_RATE=0.000001 EXPERT_UPDATE_MODE=full \
  EXPERT_SELECTION_JSON= STAGE_B_CHECKPOINT="$STAGE_B_CHECKPOINT" \
  STAGE_B_CHECKPOINT_SHA256="$STAGE_B_CHECKPOINT_SHA256" \
  MULTIMODAL_INITIAL_CHECKPOINT="$MULTIMODAL_INITIAL_CHECKPOINT" \
  MULTIMODAL_INITIAL_CHECKPOINT_SHA256="$MULTIMODAL_INITIAL_CHECKPOINT_SHA256" \
  MULTIMODAL_INITIAL_MANIFEST="$MULTIMODAL_INITIAL_MANIFEST" \
  MULTIMODAL_INITIALIZATION_SCOPE=image SPEECH_UNFREEZE_LAST_BLOCKS=1 \
  SPEECH_UNFREEZE_LAYER_NORM=1 SPEECH_ENCODER_LEARNING_RATE=0.000005 \
  LEARNING_RATE=0.0005 RETRIEVAL_HEAD_LEARNING_RATE=0.0 LM_HEAD_LEARNING_RATE=0.00001 \
  CONTRASTIVE_COEF=0.2 CONTRASTIVE_NEGATIVES=128 CONTRASTIVE_TEMPERATURE=0.07 \
  SPEECH_CONTRASTIVE_COEF=0.2 SPEECH_CONTRASTIVE_TEMPERATURE=0.04 \
  SPEECH_CENTER_POSITIVE_WEIGHT=5.0 SPEECH_RAW_POSITIVE_WEIGHT=0.0 \
  SPEECH_CONDITIONAL_RANKING_COEF=3.0 \
  SPEECH_BEHAVIOR_KL_COEF="$SPEECH_BEHAVIOR_KL_COEF" \
  SPEECH_BEHAVIOR_KL_TEMPERATURE="$SPEECH_BEHAVIOR_KL_TEMPERATURE" \
  WEIGHT_DECAY=0.0 GRAD_CLIP=5.0 LOG_EVERY_STEPS=25 SAVE_EVERY_STEPS="$MAIN_STEPS" \
  bash scripts/submit_e3_candidate_runai.sh
