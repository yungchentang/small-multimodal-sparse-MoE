#!/usr/bin/env bash
set -euo pipefail

# Hash-bound reproduction launcher for sme-selected-k2-1k-s42-final260713a.
# Scientific settings are constants below. Machine and Run:AI settings are
# supplied by the caller or an ignored environment file.

SOURCE_COMMIT_SHA="519ecacfce49afba137298a036dd9e9c9f6d9bc8"

# These manifest fields have no CLI override in real-required-runs. The exact
# source commit above binds their implementation and prevents default drift.
readonly CONDITIONAL_CANDIDATE_PERMUTATION_CONTRACT="query_identity_seeded"
readonly CONDITIONAL_CONTROL_SEED_CONTRACT="42"
readonly CONDITIONAL_TIE_EPSILON_CONTRACT="1e-08"
readonly SAVE_EXPERT_WEIGHTS_CONTRACT="false"
readonly SPEECH_FEATURE_CACHE_POLICY_CONTRACT="strict_recompute_no_persistent_cache"

ARTIFACT_ROOT="${ARTIFACT_ROOT:?set the absolute root containing data/ and outputs/}"
SOURCE_WORKTREE="${SOURCE_WORKTREE:?set the absolute clean checkout at $SOURCE_COMMIT_SHA}"
CHECK_ONLY=0

case "${1:-}" in
  --check-only) CHECK_ONLY=1; shift ;;
  "") ;;
  *) echo "usage: $0 [--check-only]" >&2; exit 2 ;;
esac
[ "$#" -eq 0 ] || { echo "usage: $0 [--check-only]" >&2; exit 2; }

for path in "$ARTIFACT_ROOT" "$SOURCE_WORKTREE"; do
  [[ "$path" = /* ]] || { echo "path must be absolute: $path" >&2; exit 2; }
done

GIT=(git -c "safe.directory=$SOURCE_WORKTREE" -C "$SOURCE_WORKTREE")
[ "$("${GIT[@]}" rev-parse HEAD)" = "$SOURCE_COMMIT_SHA" ] || {
  echo "source worktree must be at $SOURCE_COMMIT_SHA" >&2
  exit 2
}
[ -z "$("${GIT[@]}" status --porcelain)" ] || {
  echo "source worktree must be clean: $SOURCE_WORKTREE" >&2
  exit 2
}

verify_sha256() {
  local expected="$1" path="$2" actual
  [ -f "$path" ] && [ ! -L "$path" ] || {
    echo "missing or unsafe exact-final artifact: $path" >&2
    exit 2
  }
  actual="$(sha256sum -- "$path")"
  actual="${actual%% *}"
  [ "$actual" = "$expected" ] || {
    echo "SHA-256 mismatch for $path" >&2
    echo "expected: $expected" >&2
    echo "actual:   $actual" >&2
    exit 2
  }
}

DATA_DIR="$ARTIFACT_ROOT/data/real_subset_final_s42_260713a"
DEVELOPMENT_SPLIT_MANIFEST="$ARTIFACT_ROOT/outputs/development_splits_final_s42_260713a/manifest.json"
EXPERT_SELECTION_JSON="$ARTIFACT_ROOT/outputs/final_esft_s42_260713a/esft_gate_k2.json"
STAGE_B_CHECKPOINT="$ARTIFACT_ROOT/outputs/final_stage_b_distill_s42_260713a/E2D_logits_kl/checkpoint_final.pt"
MULTIMODAL_INITIAL_CHECKPOINT="$ARTIFACT_ROOT/outputs/final_speech_focus_s42_260713a/speechheavy3k_seed42/E3_final_multimodal_top2/checkpoint_final.pt"
MULTIMODAL_INITIAL_MANIFEST="$ARTIFACT_ROOT/outputs/final_speech_focus_s42_260713a/speechheavy3k_seed42/manifest.json"

# Inputs and initialization artifacts.
verify_sha256 f1ff9e12c1bc10ba2c1e863728be6067ebf9553ce3e1d693c38363ec6ab86e9a "$DATA_DIR/manifest.json"
verify_sha256 a9a9a0fba654d7191e996719713cd15d8a8d65e6e04e7917e9c9bd21ec69aaf8 "$DATA_DIR/text_blocks_train.jsonl"
verify_sha256 7169596792c0955878f21e0af27cabc1fddacca7daed607aa46f23c638667aa0 "$DATA_DIR/text_blocks_eval.jsonl"
verify_sha256 480d7be39ee4c7359973c7e2ee2bf6dbdcdc72d406f6d99ade2a3de7c3a10754 "$DATA_DIR/image_captions.jsonl"
verify_sha256 7cbed039c45279dcb4c09e782c3173c033a98d0c912333e517a2feacb4a5af7f "$DATA_DIR/speech_transcripts.jsonl"
verify_sha256 d74924addd18921c85271092dbafeda906364acfa2f81c8a92e779ab2c171a82 "$DEVELOPMENT_SPLIT_MANIFEST"
verify_sha256 50ec76da11836c0845e4c4142ca6915e62e29a9f0f916561b522e93212c82a7c "$EXPERT_SELECTION_JSON"
verify_sha256 dd3da666b0307e390e6463321551f2195e4d61f4e4320a5a3e30831bb6c2a5dd "$STAGE_B_CHECKPOINT"
verify_sha256 504f9dfe35ebdcf88b5e8be8b9b8a75cb0d91472c64ff25792034db03c20f9c1 "$MULTIMODAL_INITIAL_CHECKPOINT"
verify_sha256 adfcbd86f370353cd6221c40703e4a7e985fc8cff37e9b19730915248189f350 "$MULTIMODAL_INITIAL_MANIFEST"

# Canonical result and Run:AI provenance used to recover this contract.
verify_sha256 bd0d6bed11b94cd05ecb05743abb08869a9cf78d15b9318585f386c3b598cb8d "$ARTIFACT_ROOT/outputs/final_selected_stage_b_s42_260713a/manifest.json"
verify_sha256 9da9d1ca1f7b33ef8477614862597993bdc907300809991679cb9bf83ba221ef "$ARTIFACT_ROOT/outputs/final_runai_raw_evidence_s42_260713a/sme-selected-k2-1k-s42-final260713a.describe.txt"
verify_sha256 2f6632ed84a697e7e9918ad76c4e801db7204bb15f716dec1a0912aa09ca1efa "$ARTIFACT_ROOT/outputs/final_runai_success_ledger_s42_260713a/runai_success_ledger.json"

if [ "$CHECK_ONLY" = "1" ]; then
  echo "Exact-final E3 preflight passed for source $SOURCE_COMMIT_SHA"
  echo "Source-bound contracts: permutation=$CONDITIONAL_CANDIDATE_PERMUTATION_CONTRACT control_seed=$CONDITIONAL_CONTROL_SEED_CONTRACT tie_epsilon=$CONDITIONAL_TIE_EPSILON_CONTRACT save_expert_weights=$SAVE_EXPERT_WEIGHTS_CONTRACT speech_cache=$SPEECH_FEATURE_CACHE_POLICY_CONTRACT"
  exit 0
fi

JOB_NAME="${JOB_NAME:?set a new Run:AI JOB_NAME}"
OUT="${OUT:?set a new absolute output root}"
: "${PROJECT:?set the Run:AI project}"
: "${IMAGE:?set the container image}"
: "${RUN_AS_UID:?set the container user ID}"
: "${RUN_AS_GID:?set the container group ID}"
: "${SCRATCH_PVC:?set the scratch PVC name}"
: "${SCRATCH_MOUNT:?set the scratch mount path}"
: "${HOME_PVC:?set the home PVC name}"
: "${HOME_MOUNT:?set the home mount path}"
: "${HF_HOME:?set the Hugging Face cache path}"
: "${VENV_PATH:?set the cluster virtual environment path}"
[[ "$OUT" = /* ]] || { echo "OUT must be absolute" >&2; exit 2; }
[ ! -e "$OUT" ] || { echo "refusing to overwrite OUT: $OUT" >&2; exit 2; }
[ "$JOB_NAME" != "sme-selected-k2-1k-s42-final260713a" ] || {
  echo "use a new JOB_NAME; the canonical job identity is immutable" >&2
  exit 2
}

# The source commit and Stage-B checkpoint enforce Top-8 teacher / Top-2
# student routing. All remaining effective runner arguments are explicit here.
export JOB_NAME OUT
export MODE="real-required-runs"
export GPU="1" CPU="12" MEMORY="160G"
export PROJECT IMAGE RUN_AS_UID RUN_AS_GID
export SCRATCH_PVC SCRATCH_MOUNT HOME_PVC HOME_MOUNT HF_HOME VENV_PATH
export SUBMIT_REPO_DIR="$SOURCE_WORKTREE"
export SOURCE_COMMIT_SHA

export SEED="42" TOP_K="2" TEACHER_TOP_K="8" STUDENT_TOP_K="2"
export BASE_MODEL="allenai/OLMoE-1B-7B-0924"
export VISION_MODEL="openai/clip-vit-base-patch32"
export SPEECH_MODEL="openai/whisper-base.en"
export DATA_DIR DEVELOPMENT_SPLIT_MANIFEST EXPERT_SELECTION_JSON STAGE_B_CHECKPOINT
export MULTIMODAL_INITIAL_CHECKPOINT MULTIMODAL_INITIAL_MANIFEST
export DEVELOPMENT_SPEECH_SOURCE_SHA256="7cbed039c45279dcb4c09e782c3173c033a98d0c912333e517a2feacb4a5af7f"
export FEATURE_CACHE_DIR="$OUT/feature_cache"

export SPEECH_TARGET_SPACE="olmoe_text_hidden"
export IMAGE_ALIGNMENT_TARGET="olmoe_caption_hidden"
export IMAGE_BRIDGE_TYPE="linear_projector" AUDIO_BRIDGE_TYPE="attention_pool"
export BRIDGE_NUM_HEADS="4" ALIGNMENT_PREFIX_RESIDUAL="0"
export IMAGE_PREFIX_TOKENS="50" AUDIO_PREFIX_TOKENS="64" ENCODER_FEATURE_TOKENS="100"
export SAMPLE_RATE="16000" AUDIO_MAX_SECONDS="6.0" MAX_LENGTH="512"

export FINAL_STEPS="1000" ABLATION_STEPS="0" CAPACITY_ABLATION_STEPS="0" EXPERT_ABLATION_STEPS="0"
export TRAIN_BATCH_SIZE="1" EVAL_BATCH_SIZE="1"
export MODALITY_CYCLE="text,image,speech,speech"
export TEXT_EVAL_BLOCKS="160" RETRIEVAL_EVAL_SAMPLES="137"
export CONDITIONAL_EVAL_SAMPLES="137" IMAGE_EVAL_SAMPLES="137" SPEECH_EVAL_SAMPLES="137"
export CONDITIONAL_NEGATIVES="9" CONDITIONAL_BATCH_SIZE="16"
export CONDITIONAL_RANKING_NEGATIVES="9" CONDITIONAL_RANKING_NEGATIVE_MODE="stride"
export CONDITIONAL_RANKING_HARD_POOL_SIZE="512" CONDITIONAL_RANKING_TEMPERATURE="0.7"
export IMAGE_CONDITIONAL_RANKING_COEF="0.5" SPEECH_CONDITIONAL_RANKING_COEF="3.0"

export CAPACITY_FACTOR="8.0" CAPACITY_ABLATION_FACTOR="1.25" AUX_COEF="0.02"
export ROUTER_Z_LOSS_COEF="0.0" EXPERT_DROPOUT_PROB="0.0"
export DYNAMIC_EXPERT_BIAS_LR="0.0" DYNAMIC_EXPERT_BIAS_UPDATE_INTERVAL="1"
export DYNAMIC_EXPERT_BIAS_WARMUP_STEPS="0" DYNAMIC_EXPERT_BIAS_MAX_ABS="2.0"
export TRAIN_ROUTER_GATES="0" TRAIN_EXPERTS="0" TRAIN_LM_HEAD="0"
export EXPERT_SELECTION_METHOD="ESFT-Gate" EXPERT_UPDATE_MODE="full"
export EXPERT_ANCHOR_COEFFICIENT="0.01" ALLOW_SELECTED_EXPERT_ROUTER_TUNING="0"
export STAGE_B_CHECKPOINT_SHA256="dd3da666b0307e390e6463321551f2195e4d61f4e4320a5a3e30831bb6c2a5dd"
export MULTIMODAL_INITIAL_CHECKPOINT_SHA256="504f9dfe35ebdcf88b5e8be8b9b8a75cb0d91472c64ff25792034db03c20f9c1"
export MULTIMODAL_INITIALIZATION_SCOPE="image"
export SPEECH_INITIAL_CHECKPOINT="$MULTIMODAL_INITIAL_CHECKPOINT"
export SPEECH_INITIAL_CHECKPOINT_SHA256="504f9dfe35ebdcf88b5e8be8b9b8a75cb0d91472c64ff25792034db03c20f9c1"
export SPEECH_INITIAL_MANIFEST="$MULTIMODAL_INITIAL_MANIFEST"

export LEARNING_RATE="0.0005" ROUTER_LEARNING_RATE="0.000002" EXPERT_LEARNING_RATE="0.000001"
export RETRIEVAL_HEAD_LEARNING_RATE="0.0" LM_HEAD_LEARNING_RATE="0.00001"
export SPEECH_ENCODER_LEARNING_RATE="0.000005"
export SPEECH_UNFREEZE_LAST_BLOCKS="1" SPEECH_UNFREEZE_LAYER_NORM="1"
export WEIGHT_DECAY="0.0" GRAD_CLIP="5.0"
export CONTRASTIVE_COEF="0.2" IMAGE_CONTRASTIVE_COEF="0.0" SPEECH_CONTRASTIVE_COEF="0.2"
export CONTRASTIVE_TEMPERATURE="0.07" IMAGE_CONTRASTIVE_TEMPERATURE="0.07"
export SPEECH_CONTRASTIVE_TEMPERATURE="0.04"
export CONTRASTIVE_NEGATIVES="128" IMAGE_CONTRASTIVE_NEGATIVES="-1" SPEECH_CONTRASTIVE_NEGATIVES="-1"
export CENTER_POSITIVE_WEIGHT="1.0" RAW_POSITIVE_WEIGHT="0.0"
export IMAGE_CENTER_POSITIVE_WEIGHT="1.5" IMAGE_RAW_POSITIVE_WEIGHT="0.05"
export SPEECH_CENTER_POSITIVE_WEIGHT="5.0" SPEECH_RAW_POSITIVE_WEIGHT="0.0"
export SPEECH_BEHAVIOR_KL_COEF="0.0" SPEECH_BEHAVIOR_KL_TEMPERATURE="1.0"
export SPEECH_SHARED_CONTRASTIVE_COEF="0.0" SPEECH_SHARED_CONTRASTIVE_TEMPERATURE="0.07"
export SPEECH_TEACHER_BANK_BATCH_SIZE="64"
export ALIGNMENT_PRETRAIN_STEPS="0" ALIGNMENT_PRETRAIN_LOG_EVERY="100"
export ALIGNMENT_PRETRAIN_MODALITIES="image,speech"
export LOG_EVERY_STEPS="25" SAVE_EVERY_STEPS="500" POSTPROCESS_REQUIRED_RUNS="0"
export GAMMA_MIN="0.25" GAMMA_MAX="2.0"
export REAL_IMAGE_SAMPLES="5250" REAL_SPEECH_SAMPLES="5250"
export CAPTION_MIN_ASCII_RATIO="0.85" CAPTION_MIN_LETTERS="8"
export REAL_MAX_SOURCE_AUDIO_SECONDS="6.0" REAL_MAX_TRANSCRIPT_WORDS="18"

# Prevent a local .env.runai from overriding the constants after this point.
EMPTY_ENV_FILE="$(mktemp "${TMPDIR:-/tmp}/sparse-moe-exact-e3-env.XXXXXX")"
trap 'rm -f "$EMPTY_ENV_FILE"' EXIT
export RUNAI_ENV_FILE="$EMPTY_ENV_FILE"

bash "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/submit_runai.sh"
