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
  set -a
  # shellcheck disable=SC1090
  source "$RUNAI_ENV_FILE"
  set +a
fi

: "${PROJECT:?PROJECT is required}"
: "${IMAGE:?IMAGE is required}"
: "${RUN_AS_UID:?RUN_AS_UID is required}"
: "${RUN_AS_GID:?RUN_AS_GID is required}"
: "${SCRATCH_PVC:?SCRATCH_PVC is required}"
: "${SCRATCH_MOUNT:?SCRATCH_MOUNT is required}"
: "${HOME_PVC:?HOME_PVC is required}"
: "${HOME_MOUNT:?HOME_MOUNT is required}"
: "${DATA_DIR:?canonical DATA_DIR is required}"
: "${DEVELOPMENT_SPLIT_MANIFEST:?DEVELOPMENT_SPLIT_MANIFEST is required}"
: "${DEVELOPMENT_SPEECH_SOURCE_SHA256:?DEVELOPMENT_SPEECH_SOURCE_SHA256 is required}"
: "${VENV_PATH:?shared VENV_PATH is required}"
if [ ! -f "$VENV_PATH/bin/activate" ]; then
  echo "VENV_PATH must contain bin/activate: $VENV_PATH" >&2
  exit 2
fi

REPO_DIR="${SUBMIT_REPO_DIR:-$REPO_ROOT}"
SOURCE_COMMIT_SHA="${SOURCE_COMMIT_SHA:-$(git -c safe.directory="$REPO_DIR" -C "$REPO_DIR" rev-parse HEAD)}"
ACTUAL_SOURCE_SHA="$(git -c safe.directory="$REPO_DIR" -C "$REPO_DIR" rev-parse HEAD)"
if [ "$ACTUAL_SOURCE_SHA" != "$SOURCE_COMMIT_SHA" ]; then
  echo "source commit mismatch: expected $SOURCE_COMMIT_SHA, found $ACTUAL_SOURCE_SHA" >&2
  exit 2
fi
if [ -n "$(git -c safe.directory="$REPO_DIR" -C "$REPO_DIR" status --porcelain --untracked-files=all)" ]; then
  echo "native Whisper WER submission requires a clean source worktree" >&2
  exit 2
fi

MODEL_ID="openai/whisper-base.en"
MODEL_REVISION="911407f4214e0e1d82085af863093ec0b66f9cd6"
SEED=42
GPU="${GPU:-1}"
if [ "$GPU" != "1" ]; then
  echo "native Whisper WER launcher requires exactly one GPU" >&2
  exit 2
fi

STAMP="$(date +%y%m%d%H%M%S)"
JOB_NAME="${JOB_NAME:-sparse-moe-whisper-native-wer-${STAMP}}"
CPU="${CPU:-8}"
MEMORY="${MEMORY:-64G}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH_MOUNT/$JOB_NAME}"
HF_HOME="${HF_HOME:-$HOME_MOUNT/.cache/huggingface}"
BATCH_SIZE="${BATCH_SIZE:-8}"

echo "Submitting $JOB_NAME: speech_dev native WER, $MODEL_ID@$MODEL_REVISION, seed=$SEED"
runai submit --name "$JOB_NAME" \
  -p "$PROJECT" \
  -i "$IMAGE" \
  -g "$GPU" \
  --cpu "$CPU" \
  --memory "$MEMORY" \
  --run-as-uid "$RUN_AS_UID" \
  --run-as-gid "$RUN_AS_GID" \
  --existing-pvc "claimname=$SCRATCH_PVC,path=$SCRATCH_MOUNT" \
  --existing-pvc "claimname=$HOME_PVC,path=$HOME_MOUNT" \
  --working-dir "$REPO_DIR" \
  --large-shm \
  --backoff-limit 0 \
  --environment GIT_CONFIG_COUNT=1 \
  --environment GIT_CONFIG_KEY_0=safe.directory \
  --environment GIT_CONFIG_VALUE_0="$REPO_DIR" \
  --environment SOURCE_COMMIT_SHA="$SOURCE_COMMIT_SHA" \
  --environment RUNAI_JOB_NAME="$JOB_NAME" \
  --environment RUNAI_PROJECT="$PROJECT" \
  --environment REPO_DIR="$REPO_DIR" \
  --environment DATA_DIR="$DATA_DIR" \
  --environment DEVELOPMENT_SPLIT_MANIFEST="$DEVELOPMENT_SPLIT_MANIFEST" \
  --environment DEVELOPMENT_SPEECH_SOURCE_SHA256="$DEVELOPMENT_SPEECH_SOURCE_SHA256" \
  --environment OUTPUT_DIR="$OUTPUT_DIR" \
  --environment HF_HOME="$HF_HOME" \
  --environment VENV_PATH="$VENV_PATH" \
  --environment BATCH_SIZE="$BATCH_SIZE" \
  --environment MODEL_ID="$MODEL_ID" \
  --environment MODEL_REVISION="$MODEL_REVISION" \
  --environment SEED="$SEED" \
  --command \
  -- /bin/bash -c '
set -euo pipefail
cd "$REPO_DIR"
test "$(git rev-parse HEAD)" = "$SOURCE_COMMIT_SHA"
test -z "$(git status --porcelain --untracked-files=all)"
test "$MODEL_ID" = "openai/whisper-base.en"
test "$MODEL_REVISION" = "911407f4214e0e1d82085af863093ec0b66f9cd6"
test "$SEED" = "42"
source "$VENV_PATH/bin/activate"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$REPO_DIR"
python scripts/eval_whisper_native_wer.py \
  --data-dir "$DATA_DIR" \
  --development-split-manifest "$DEVELOPMENT_SPLIT_MANIFEST" \
  --trusted-speech-source-sha256 "$DEVELOPMENT_SPEECH_SOURCE_SHA256" \
  --source-commit-sha "$SOURCE_COMMIT_SHA" \
  --output-dir "$OUTPUT_DIR" \
  --expected-rows 137 \
  --batch-size "$BATCH_SIZE" \
  --max-seconds 6.0 \
  --device cuda
'
