#!/usr/bin/env bash
set -euo pipefail

BASE_OUT="${BASE_OUT:-outputs/review_repair/followup_development_eval_v2}"
JOB_STAMP="${JOB_STAMP:-260710b}"
PROJECT="${PROJECT:?PROJECT is required; configure it in .env.runai or the environment}"
STAGE_B_CHECKPOINT="${STAGE_B_CHECKPOINT:?verified Stage-B checkpoint is required}"
STAGE_B_CHECKPOINT_SHA256="${STAGE_B_CHECKPOINT_SHA256:?verified Stage-B checkpoint SHA256 is required}"
if [[ ! "$STAGE_B_CHECKPOINT_SHA256" =~ ^[0-9a-fA-F]{64}$ ]]; then
  echo "Stage-B checkpoint SHA256 must be an exact 64-character digest" >&2
  exit 2
fi
IMAGE_MANIFEST="${IMAGE_MANIFEST:-outputs/review_repair/development_eval_v1/image_val.jsonl}"
SPEECH_MANIFEST="${SPEECH_MANIFEST:-outputs/review_repair/development_eval_v1/speech_val.jsonl}"
ONLY_CANDIDATES=",${ONLY_CANDIDATES:-baseline_cap8,speech3rank49,speech3rank19,hard19,balanced19,speechlight49},"
ONLY_CELLS=",${ONLY_CELLS:-r5,r10,h10},"

SPECS=(
  "baseline_cap8|outputs/review_repair/corrected_cap8a02_seed42"
  "baseline_cap8_clean|outputs/review_repair/corrected_cap8a02_seed42_cleanrerun"
  "baseline_cap8_retry2|outputs/review_repair/corrected_cap8a02_seed42_cleanrerun2"
  "baseline_cap8_retry3|outputs/review_repair/corrected_cap8a02_seed42_cleanrerun3"
  "speech3rank49|outputs/review_repair/followup_speech3rank49_seed42"
  "speech3rank19|outputs/review_repair/followup_speech3rank19_seed42"
  "hard19|outputs/review_repair/followup_hard19_seed42"
  "balanced19|outputs/review_repair/followup_balanced19_seed42"
  "speechlight49|outputs/review_repair/followup_speechlight49_seed42"
)

for required in "$IMAGE_MANIFEST" "$SPEECH_MANIFEST"; do
  if [ ! -s "$required" ]; then
    echo "missing development manifest: $required" >&2
    exit 2
  fi
done

submit_cell() {
  local label="$1"
  local root="$2"
  local cell="$3"
  local candidates="$4"
  local negative_mode="$5"
  local output_dir="${BASE_OUT}/${label}/${cell}"
  local checkpoint="${root}/E3_final_multimodal_top2/checkpoint_final.pt"
  local job_label="${label//_/-}"
  local job_name="sme-fdev-${job_label}-${cell}-${JOB_STAMP}"

  if [[ "$ONLY_CANDIDATES" != *",${label},"* ]] || [[ "$ONLY_CELLS" != *",${cell},"* ]]; then
    return 0
  fi
  for required in "$checkpoint" "$root/manifest.json"; do
    if [ ! -s "$required" ]; then
      echo "candidate is incomplete: $required" >&2
      return 1
    fi
  done
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
    PROTOCOL_NAME=development_conditional_v2 CANDIDATE_SEED=271828 CONTROL_SEED=42 \
    CAPACITY_FACTOR=8.0 AUX_COEF=0.02 \
    IMAGE_PREFIX_TOKENS=50 AUDIO_PREFIX_TOKENS=64 ENCODER_FEATURE_TOKENS=100 \
    IMAGE_EVAL_SAMPLES=250 SPEECH_EVAL_SAMPLES=250 CONDITIONAL_QUERIES=250 \
    CONDITIONAL_CANDIDATES="$candidates" CONDITIONAL_NEGATIVES="$((candidates - 1))" \
    CONDITIONAL_BATCH_SIZE=16 NEGATIVE_MODE="$negative_mode" \
    EVAL_PATH=shared_prefix PREFIX_CONTROL=real \
    BOOTSTRAP_SAMPLES=2000 BOOTSTRAP_SEED=12345 \
    bash scripts/submit_runai.sh
}

for spec in "${SPECS[@]}"; do
  IFS='|' read -r label root <<<"$spec"
  submit_cell "$label" "$root" r5 5 random
  submit_cell "$label" "$root" r10 10 random
  submit_cell "$label" "$root" h10 10 hard_text
done
