#!/usr/bin/env bash
set -euo pipefail

EVIDENCE_MANIFEST_SPEC="${EVIDENCE_MANIFEST_SPEC:?EVIDENCE_MANIFEST_SPEC is required}"
EVIDENCE_MANIFEST_OUTPUT="${EVIDENCE_MANIFEST_OUTPUT:?EVIDENCE_MANIFEST_OUTPUT is required}"
JOB_NAME="${JOB_NAME:-sme-final-evidence-verifier}"

if [ ! -s "$EVIDENCE_MANIFEST_SPEC" ]; then
  echo "missing evidence manifest spec: $EVIDENCE_MANIFEST_SPEC" >&2
  exit 2
fi
if [ -e "$EVIDENCE_MANIFEST_OUTPUT" ]; then
  echo "refusing to overwrite evidence result manifest: $EVIDENCE_MANIFEST_OUTPUT" >&2
  exit 2
fi

env \
  MODE=evidence-verifier GPU=0 CPU="${CPU:-4}" MEMORY="${MEMORY:-32G}" \
  JOB_NAME="$JOB_NAME" EVIDENCE_MANIFEST_SPEC="$EVIDENCE_MANIFEST_SPEC" \
  EVIDENCE_MANIFEST_OUTPUT="$EVIDENCE_MANIFEST_OUTPUT" \
  bash scripts/submit_runai.sh
