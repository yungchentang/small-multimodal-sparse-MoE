#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
PROJECT="${PROJECT:?PROJECT is required}"
CAPACITY_FACTOR="${CAPACITY_FACTOR:?CAPACITY_FACTOR is required}"
AUX_COEF="${AUX_COEF:?AUX_COEF is required}"
DATA_DIR="${DATA_DIR:-data/real_subset_clean_260708b}"
IMAGE_MANIFEST="${IMAGE_MANIFEST:?IMAGE_MANIFEST is required}"
SPEECH_MANIFEST="${SPEECH_MANIFEST:?SPEECH_MANIFEST is required}"
CHECKPOINT="${CHECKPOINT:-$RUN_ROOT/E3_final_multimodal_top2/checkpoint_final.pt}"
OUT="${OUT:-outputs/review_repair/final_specialization}"
JOB_NAME="${JOB_NAME:-sme-final-specialization}"

for required in "$CHECKPOINT" "$RUN_ROOT/manifest.json" "$IMAGE_MANIFEST" "$SPEECH_MANIFEST"; do
  if [ ! -s "$required" ]; then
    echo "missing specialization input: $required" >&2
    exit 2
  fi
done
if [ -e "$OUT" ]; then
  echo "refusing to overwrite specialization output: $OUT" >&2
  exit 2
fi
python -c 'import json,sys; p,c,a=sys.argv[1:]; m=json.load(open(p)); d=m["args"]; x=m.get("completion",{}); assert x.get("status")=="completed"; assert int(x.get("e3_steps",0))==int(d["final_steps"])>0; assert float(d["capacity_factor"])==float(c); assert float(d["aux_coef"])==float(a)' \
  "$RUN_ROOT/manifest.json" "$CAPACITY_FACTOR" "$AUX_COEF"

env \
  PROJECT="$PROJECT" JOB_NAME="$JOB_NAME" \
  MODE=specialization-analysis GPU="${GPU:-1}" CPU="${CPU:-8}" MEMORY="${MEMORY:-120G}" \
  OUT="$OUT" RUN_OUTPUT_DIR="$RUN_ROOT" CHECKPOINT="$CHECKPOINT" \
  DATA_DIR="$DATA_DIR" FEATURE_CACHE_DIR="$OUT/feature_cache" \
  IMAGE_MANIFEST="$IMAGE_MANIFEST" SPEECH_MANIFEST="$SPEECH_MANIFEST" \
  CAPACITY_FACTOR="$CAPACITY_FACTOR" AUX_COEF="$AUX_COEF" \
  IMAGE_PREFIX_TOKENS=50 AUDIO_PREFIX_TOKENS=64 ENCODER_FEATURE_TOKENS=100 \
  IMAGE_EVAL_SAMPLES="${IMAGE_EVAL_SAMPLES:-137}" SPEECH_EVAL_SAMPLES="${SPEECH_EVAL_SAMPLES:-137}" \
  ROUTING_ANALYSIS_BATCH_SIZE=4 ROUTING_TEXT_BATCHES=8 ROUTING_MODALITY_BATCHES=8 \
  QUALITATIVE_EXAMPLES="${QUALITATIVE_EXAMPLES:-12}" MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}" \
  INTERVENTION_TOP_EXPERTS=5 INTERVENTION_EXAMPLES=24 INTERVENTION_TEXT_BLOCKS=16 \
  bash scripts/submit_runai.sh
