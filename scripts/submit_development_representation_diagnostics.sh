#!/usr/bin/env bash
set -euo pipefail

# Development-only representation geometry and retention diagnostics, one GPU.
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$REPO_ROOT"

SOURCE_COMMIT_SHA="${SOURCE_COMMIT_SHA:?SOURCE_COMMIT_SHA is required}"
ACTUAL_SOURCE_SHA="$(git rev-parse HEAD)"
if [ "$ACTUAL_SOURCE_SHA" != "$SOURCE_COMMIT_SHA" ]; then
  echo "source commit mismatch: expected $SOURCE_COMMIT_SHA, found $ACTUAL_SOURCE_SHA" >&2
  exit 2
fi
if ! git diff --quiet || ! git diff --cached --quiet || [ -n "$(git ls-files --others --exclude-standard)" ]; then
  echo "development diagnostics require a clean source worktree" >&2
  exit 2
fi

CHECKPOINT="${CHECKPOINT:?CHECKPOINT is required}"
GAMMA_JSON="${GAMMA_JSON:?GAMMA_JSON is required}"
DEV_IMAGE_MANIFEST="${DEV_IMAGE_MANIFEST:-outputs/review_repair/development_eval_v1/image_val.jsonl}"
DEV_SPEECH_MANIFEST="${DEV_SPEECH_MANIFEST:-outputs/review_repair/development_eval_v1/speech_val.jsonl}"
OUT="${OUT:?OUT is required}"
JOB_NAME="${JOB_NAME:?JOB_NAME is required}"
case "${CHECKPOINT}:${GAMMA_JSON}:${DEV_IMAGE_MANIFEST}:${DEV_SPEECH_MANIFEST}:${OUT}" in
  *sealed*|*synthetic*) echo "refusing sealed/synthetic development diagnostics input" >&2; exit 2 ;;
esac
for required in "$CHECKPOINT" "$GAMMA_JSON" "$DEV_IMAGE_MANIFEST" "$DEV_SPEECH_MANIFEST"; do
  [ -s "$required" ] || { echo "missing development diagnostics input: $required" >&2; exit 2; }
done
[ ! -e "$OUT" ] || { echo "refusing overwrite: $OUT" >&2; exit 2; }
[ "${#JOB_NAME}" -le 55 ] || { echo "job name too long: $JOB_NAME" >&2; exit 2; }

COMMON_GIT_DIR="$(git rev-parse --path-format=absolute --git-common-dir)"
DEFAULT_RUNAI_ENV="$(dirname "$COMMON_GIT_DIR")/.env.runai"
if [ -z "${RUNAI_ENV_FILE:-}" ] && [ -f "$DEFAULT_RUNAI_ENV" ]; then
  export RUNAI_ENV_FILE="$DEFAULT_RUNAI_ENV"
fi

env MODE=development-representation-diagnostics \
  SOURCE_COMMIT_SHA="$SOURCE_COMMIT_SHA" SUBMIT_REPO_DIR="$REPO_ROOT" \
  JOB_NAME="$JOB_NAME" GPU="${GPU:-1}" CPU="${CPU:-8}" MEMORY="${MEMORY:-120G}" \
  CHECKPOINT="$CHECKPOINT" GAMMA_JSON="$GAMMA_JSON" \
  DEV_IMAGE_MANIFEST="$DEV_IMAGE_MANIFEST" DEV_SPEECH_MANIFEST="$DEV_SPEECH_MANIFEST" \
  OUT="$OUT" EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}" \
  bash scripts/submit_runai.sh
