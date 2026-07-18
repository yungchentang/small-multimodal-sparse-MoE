#!/usr/bin/env bash
set -euo pipefail

# Compare completed corrected E3 candidates on one fixed development protocol.
BASE_OUT="${BASE_OUT:-outputs/review_repair/candidate_development_eval}"
JOB_STAMP="${JOB_STAMP:-260709}"
PROJECT="${PROJECT:?PROJECT is required; configure it in .env.runai or the environment}"
STAGE_B_CHECKPOINT="${STAGE_B_CHECKPOINT:?verified Stage-B checkpoint is required}"
STAGE_B_CHECKPOINT_SHA256="${STAGE_B_CHECKPOINT_SHA256:?verified Stage-B checkpoint SHA256 is required}"
if [[ ! "$STAGE_B_CHECKPOINT_SHA256" =~ ^[0-9a-fA-F]{64}$ ]]; then
  echo "Stage-B checkpoint SHA256 must be an exact 64-character digest" >&2
  exit 2
fi
IMAGE_MANIFEST="${IMAGE_MANIFEST:-outputs/review_repair/development_eval_v1/image_val.jsonl}"
SPEECH_MANIFEST="${SPEECH_MANIFEST:-outputs/review_repair/development_eval_v1/speech_val.jsonl}"
ONLY_CANDIDATES=",${ONLY_CANDIDATES:-cap7a01,cap7a02,cap7a04,cap8a02,cap6a02},"
ONLY_CELLS=",${ONLY_CELLS:-r5,r10},"

SPECS=(
  "cap7a01|outputs/review_repair/corrected_cap7a01_seed42|7.0|0.01"
  "cap7a02|outputs/review_repair/corrected_cap7a02_seed42|7.0|0.02"
  "cap7a04|outputs/review_repair/corrected_cap7a04_seed42|7.0|0.04"
  "cap8a02|outputs/review_repair/corrected_cap8a02_seed42|8.0|0.02"
  "cap6a02|outputs/review_repair/corrected_cap6a02_seed42|6.0|0.02"
)

for required in "$IMAGE_MANIFEST" "$SPEECH_MANIFEST"; do
  if [ ! -s "$required" ]; then
    echo "missing development manifest: $required" >&2
    exit 2
  fi
done

submit_cell() {
  local label="$1" root="$2" capacity="$3" aux="$4" cell="$5" candidates="$6"
  local output_dir="${BASE_OUT}/${label}/${cell}"
  local checkpoint="${root}/E3_final_multimodal_top2/checkpoint_final.pt"
  local job_name="sme-dev-${label}-${cell}-${JOB_STAMP}"

  if [[ "$ONLY_CANDIDATES" != *",${label},"* ]] || [[ "$ONLY_CELLS" != *",${cell},"* ]]; then
    return 0
  fi
  for required in "$checkpoint" "$root/manifest.json"; do
    if [ ! -s "$required" ]; then
      echo "candidate is incomplete: $required" >&2
      return 1
    fi
  done
  python -c 'import json,sys; p,c,a=sys.argv[1:]; d=json.load(open(p))["args"]; assert float(d["capacity_factor"])==float(c); assert float(d["aux_coef"])==float(a); assert int(d["final_steps"])==6000' \
    "$root/manifest.json" "$capacity" "$aux"
  if [ -e "$output_dir" ]; then
    echo "refusing to overwrite development evaluation: $output_dir" >&2
    return 1
  fi

  env \
    PROJECT="$PROJECT" JOB_NAME="$job_name" MODE=conditional-eval \
    GPU="${GPU:-1}" CPU="${CPU:-8}" MEMORY="${MEMORY:-100G}" \
    OUT="${output_dir}/metrics.json" PER_QUERY_OUTPUT="${output_dir}/per_query.jsonl" \
    FEATURE_CACHE_DIR="${output_dir}/feature_cache" \
    RUN_OUTPUT_DIR="$root" CHECKPOINT="$checkpoint" DATA_DIR=data/real_subset_clean_260708b \
    STAGE_B_CHECKPOINT="$STAGE_B_CHECKPOINT" STAGE_B_CHECKPOINT_SHA256="$STAGE_B_CHECKPOINT_SHA256" \
    IMAGE_MANIFEST="$IMAGE_MANIFEST" SPEECH_MANIFEST="$SPEECH_MANIFEST" \
    EVAL_SPLIT_NAME=development_selection RANDOMIZE_POSITIVE_POSITION=1 \
    EVALUATION_SCOPE=development \
    PROTOCOL_NAME=development_conditional_v1 CANDIDATE_SEED=271828 CONTROL_SEED=42 \
    CAPACITY_FACTOR="$capacity" AUX_COEF="$aux" \
    IMAGE_PREFIX_TOKENS=50 AUDIO_PREFIX_TOKENS=64 ENCODER_FEATURE_TOKENS=100 \
    IMAGE_EVAL_SAMPLES=250 SPEECH_EVAL_SAMPLES=250 CONDITIONAL_QUERIES=250 \
    CONDITIONAL_CANDIDATES="$candidates" CONDITIONAL_NEGATIVES="$((candidates - 1))" \
    CONDITIONAL_BATCH_SIZE=16 NEGATIVE_MODE=random EVAL_PATH=shared_prefix PREFIX_CONTROL=real \
    BOOTSTRAP_SAMPLES=2000 BOOTSTRAP_SEED=12345 \
    bash scripts/submit_runai.sh
}

for spec in "${SPECS[@]}"; do
  IFS='|' read -r label root capacity aux <<<"$spec"
  submit_cell "$label" "$root" "$capacity" "$aux" r5 5
  submit_cell "$label" "$root" "$capacity" "$aux" r10 10
done
