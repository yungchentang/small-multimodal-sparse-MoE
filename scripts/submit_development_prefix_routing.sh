#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
RUNAI_ENV_FILE="${RUNAI_ENV_FILE:-$REPO_ROOT/.env.runai}"

if [ -e "$RUNAI_ENV_FILE" ]; then
  if [ ! -f "$RUNAI_ENV_FILE" ] || [ -L "$RUNAI_ENV_FILE" ]; then
    echo "refusing non-regular Run:AI environment file: $RUNAI_ENV_FILE" >&2
    exit 2
  fi
  set -a
  # shellcheck disable=SC1090
  source "$RUNAI_ENV_FILE"
  set +a
fi

SOURCE_COMMIT_SHA="${SOURCE_COMMIT_SHA:?SOURCE_COMMIT_SHA is required}"
CHECKPOINT="${CHECKPOINT:?CHECKPOINT is required}"
EXPECTED_CHECKPOINT_SHA256="${EXPECTED_CHECKPOINT_SHA256:?EXPECTED_CHECKPOINT_SHA256 is required}"
CHECKPOINT_MANIFEST="${CHECKPOINT_MANIFEST:?CHECKPOINT_MANIFEST is required}"
EXPECTED_CHECKPOINT_MANIFEST_SHA256="${EXPECTED_CHECKPOINT_MANIFEST_SHA256:?EXPECTED_CHECKPOINT_MANIFEST_SHA256 is required}"
STAGE_B_CHECKPOINT="${STAGE_B_CHECKPOINT:?STAGE_B_CHECKPOINT is required}"
EXPECTED_STAGE_B_CHECKPOINT_SHA256="${EXPECTED_STAGE_B_CHECKPOINT_SHA256:?EXPECTED_STAGE_B_CHECKPOINT_SHA256 is required}"
STAGE_B_COMPANION_MANIFEST="${STAGE_B_COMPANION_MANIFEST:?STAGE_B_COMPANION_MANIFEST is required}"
EXPECTED_STAGE_B_COMPANION_MANIFEST_SHA256="${EXPECTED_STAGE_B_COMPANION_MANIFEST_SHA256:?EXPECTED_STAGE_B_COMPANION_MANIFEST_SHA256 is required}"
DEVELOPMENT_SPLIT_MANIFEST="${DEVELOPMENT_SPLIT_MANIFEST:?DEVELOPMENT_SPLIT_MANIFEST is required}"
EXPECTED_DEVELOPMENT_SPLIT_MANIFEST_SHA256="${EXPECTED_DEVELOPMENT_SPLIT_MANIFEST_SHA256:?EXPECTED_DEVELOPMENT_SPLIT_MANIFEST_SHA256 is required}"
TRAIN_IMAGE_MANIFEST="${TRAIN_IMAGE_MANIFEST:?TRAIN_IMAGE_MANIFEST is required}"
TRAIN_SPEECH_MANIFEST="${TRAIN_SPEECH_MANIFEST:?TRAIN_SPEECH_MANIFEST is required}"
DEV_IMAGE_MANIFEST="${DEV_IMAGE_MANIFEST:-}"
DEV_SPEECH_MANIFEST="${DEV_SPEECH_MANIFEST:-}"
COLLECTION_SPLIT="${COLLECTION_SPLIT:-train}"
OUT="${OUT:?OUT is required}"
SAMPLE_COUNT="${SAMPLE_COUNT:?SAMPLE_COUNT is required}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GAMMA_JSON="${GAMMA_JSON:-}"
JOB_NAME="${JOB_NAME:-sme-dev-prefix-routing-$(date +%y%m%d%H%M%S)}"

case "$SOURCE_COMMIT_SHA" in
  *[!0-9a-f]*|'') echo "SOURCE_COMMIT_SHA must be lowercase hexadecimal" >&2; exit 2 ;;
esac
if [ "${#SOURCE_COMMIT_SHA}" -ne 40 ] && [ "${#SOURCE_COMMIT_SHA}" -ne 64 ]; then
  echo "SOURCE_COMMIT_SHA must be an exact 40- or 64-hex commit" >&2
  exit 2
fi
if [ "$(git -C "$REPO_ROOT" rev-parse HEAD)" != "$SOURCE_COMMIT_SHA" ]; then
  echo "SOURCE_COMMIT_SHA does not match the submission checkout" >&2
  exit 2
fi
if [ -n "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all)" ]; then
  echo "refusing submission from a dirty source checkout" >&2
  exit 2
fi
case "$COLLECTION_SPLIT" in
  train)
    if [ -n "$DEV_IMAGE_MANIFEST" ] || [ -n "$DEV_SPEECH_MANIFEST" ]; then
      echo "train-only collection rejects DEV_IMAGE_MANIFEST/DEV_SPEECH_MANIFEST" >&2
      exit 2
    fi
    ;;
  train-dev)
    : "${DEV_IMAGE_MANIFEST:?DEV_IMAGE_MANIFEST is required for train-dev}"
    : "${DEV_SPEECH_MANIFEST:?DEV_SPEECH_MANIFEST is required for train-dev}"
    ;;
  *) echo "COLLECTION_SPLIT must be train or train-dev" >&2; exit 2 ;;
esac
required_inputs=(
  "$CHECKPOINT" "$CHECKPOINT_MANIFEST" "$STAGE_B_CHECKPOINT"
  "$STAGE_B_COMPANION_MANIFEST" "$DEVELOPMENT_SPLIT_MANIFEST"
  "$TRAIN_IMAGE_MANIFEST" "$TRAIN_SPEECH_MANIFEST"
)
if [ "$COLLECTION_SPLIT" = "train-dev" ]; then
  required_inputs+=("$DEV_IMAGE_MANIFEST" "$DEV_SPEECH_MANIFEST")
fi
for path in "${required_inputs[@]}"; do
  if [ ! -s "$path" ]; then
    echo "missing required development prefix-routing input: $path" >&2
    exit 2
  fi
done
for path in "$CHECKPOINT" "$CHECKPOINT_MANIFEST" "$STAGE_B_CHECKPOINT" "$STAGE_B_COMPANION_MANIFEST" "$DEVELOPMENT_SPLIT_MANIFEST" "$GAMMA_JSON" "$TRAIN_IMAGE_MANIFEST" "$TRAIN_SPEECH_MANIFEST" "$DEV_IMAGE_MANIFEST" "$DEV_SPEECH_MANIFEST" "$OUT"; do
  case "${path,,}" in
    *sealed*|*synthetic*) echo "development-only launcher rejects path: $path" >&2; exit 2 ;;
  esac
done
if [ -e "$OUT" ]; then
  echo "refusing to overwrite development prefix-routing output: $OUT" >&2
  exit 2
fi
if [ "${#JOB_NAME}" -gt 55 ]; then
  echo "Run:AI job name exceeds 55 characters: $JOB_NAME" >&2
  exit 2
fi

required_runai=(PROJECT IMAGE RUN_AS_UID RUN_AS_GID REPO_DIR SCRATCH_PVC SCRATCH_MOUNT HOME_PVC HOME_MOUNT)
missing=()
for name in "${required_runai[@]}"; do
  if [ -z "${!name:-}" ]; then
    missing+=("$name")
  fi
done
if [ "${#missing[@]}" -gt 0 ]; then
  printf 'Missing required Run:AI configuration: %s\n' "${missing[*]}" >&2
  exit 2
fi

HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
VENV_PATH="${VENV_PATH:-$REPO_DIR/.venv}"
CPU="${CPU:-8}"
MEMORY="${MEMORY:-120G}"

echo "Submitting $JOB_NAME to $PROJECT with exactly one GPU"
runai submit --name "$JOB_NAME" \
  -p "$PROJECT" \
  -i "$IMAGE" \
  -g 1 \
  --cpu "$CPU" \
  --memory "$MEMORY" \
  --run-as-uid "$RUN_AS_UID" \
  --run-as-gid "$RUN_AS_GID" \
  --environment SOURCE_COMMIT_SHA="$SOURCE_COMMIT_SHA" \
  --environment CHECKPOINT="$CHECKPOINT" \
  --environment EXPECTED_CHECKPOINT_SHA256="$EXPECTED_CHECKPOINT_SHA256" \
  --environment CHECKPOINT_MANIFEST="$CHECKPOINT_MANIFEST" \
  --environment EXPECTED_CHECKPOINT_MANIFEST_SHA256="$EXPECTED_CHECKPOINT_MANIFEST_SHA256" \
  --environment STAGE_B_CHECKPOINT="$STAGE_B_CHECKPOINT" \
  --environment EXPECTED_STAGE_B_CHECKPOINT_SHA256="$EXPECTED_STAGE_B_CHECKPOINT_SHA256" \
  --environment STAGE_B_COMPANION_MANIFEST="$STAGE_B_COMPANION_MANIFEST" \
  --environment EXPECTED_STAGE_B_COMPANION_MANIFEST_SHA256="$EXPECTED_STAGE_B_COMPANION_MANIFEST_SHA256" \
  --environment DEVELOPMENT_SPLIT_MANIFEST="$DEVELOPMENT_SPLIT_MANIFEST" \
  --environment EXPECTED_DEVELOPMENT_SPLIT_MANIFEST_SHA256="$EXPECTED_DEVELOPMENT_SPLIT_MANIFEST_SHA256" \
  --environment TRAIN_IMAGE_MANIFEST="$TRAIN_IMAGE_MANIFEST" \
  --environment TRAIN_SPEECH_MANIFEST="$TRAIN_SPEECH_MANIFEST" \
  --environment DEV_IMAGE_MANIFEST="$DEV_IMAGE_MANIFEST" \
  --environment DEV_SPEECH_MANIFEST="$DEV_SPEECH_MANIFEST" \
  --environment COLLECTION_SPLIT="$COLLECTION_SPLIT" \
  --environment GAMMA_JSON="$GAMMA_JSON" \
  --environment OUT="$OUT" \
  --environment SAMPLE_COUNT="$SAMPLE_COUNT" \
  --environment BATCH_SIZE="$BATCH_SIZE" \
  --environment HF_HOME="$HF_HOME" \
  --environment VENV_PATH="$VENV_PATH" \
  --environment RUNAI_JOB_NAME="$JOB_NAME" \
  --environment RUNAI_PROJECT="$PROJECT" \
  --environment GIT_CONFIG_COUNT=1 \
  --environment GIT_CONFIG_KEY_0=safe.directory \
  --environment GIT_CONFIG_VALUE_0="$REPO_DIR" \
  --existing-pvc "claimname=$SCRATCH_PVC,path=$SCRATCH_MOUNT" \
  --existing-pvc "claimname=$HOME_PVC,path=$HOME_MOUNT" \
  --working-dir "$REPO_DIR" \
  --large-shm \
  --backoff-limit 0 \
  --command \
  -- /bin/bash -lc '
set -euo pipefail
test "$(git rev-parse HEAD)" = "$SOURCE_COMMIT_SHA"
test -z "$(git status --porcelain --untracked-files=all)"
gamma_args=()
if [ -n "$GAMMA_JSON" ]; then
  gamma_args=(--gamma-json "$GAMMA_JSON")
fi
routing_args=(
  --checkpoint "$CHECKPOINT"
  --expected-checkpoint-sha256 "$EXPECTED_CHECKPOINT_SHA256"
  --checkpoint-manifest "$CHECKPOINT_MANIFEST"
  --expected-checkpoint-manifest-sha256 "$EXPECTED_CHECKPOINT_MANIFEST_SHA256"
  --stage-b-checkpoint "$STAGE_B_CHECKPOINT"
  --expected-stage-b-checkpoint-sha256 "$EXPECTED_STAGE_B_CHECKPOINT_SHA256"
  --stage-b-companion-manifest "$STAGE_B_COMPANION_MANIFEST"
  --expected-stage-b-companion-manifest-sha256 "$EXPECTED_STAGE_B_COMPANION_MANIFEST_SHA256"
  --source-commit-sha "$SOURCE_COMMIT_SHA"
  --development-split-manifest "$DEVELOPMENT_SPLIT_MANIFEST"
  --expected-development-split-manifest-sha256 "$EXPECTED_DEVELOPMENT_SPLIT_MANIFEST_SHA256"
  --train-image-manifest "$TRAIN_IMAGE_MANIFEST"
  --train-speech-manifest "$TRAIN_SPEECH_MANIFEST"
  --collection-split "$COLLECTION_SPLIT"
  --sample-count "$SAMPLE_COUNT"
  --batch-size "$BATCH_SIZE"
  --output-dir "$OUT"
)
if [ "$COLLECTION_SPLIT" = "train-dev" ]; then
  routing_args+=(
    --dev-image-manifest "$DEV_IMAGE_MANIFEST"
    --dev-speech-manifest "$DEV_SPEECH_MANIFEST"
  )
fi
exec "$VENV_PATH/bin/python" -m scripts.collect_development_prefix_routing "${routing_args[@]}" "${gamma_args[@]}"
'
