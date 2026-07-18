#!/usr/bin/env bash
set -euo pipefail

# One-GPU E3-only candidate launcher for iterative sweeps.
# It runs E0/E1/E2 plus one E3 multimodal candidate, then skips final report/audit.
# Use scripts/submit_final_runai.sh for the selected full E3/E4/E5/E6 evidence root.

STAMP="$(date +%y%m%d%H%M)"
export JOB_NAME="${JOB_NAME:-sparse-moe-clean-e3-candidate-${STAMP}}"
export OUT="${OUT:-outputs/${JOB_NAME}}"
export DATA_DIR="${DATA_DIR:-data/real_subset_final}"
export FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR:-${OUT}/feature_cache}"

export POSTPROCESS_REQUIRED_RUNS=0
export ABLATION_STEPS=0
export CAPACITY_ABLATION_STEPS=0
export EXPERT_ABLATION_STEPS=0

bash scripts/submit_final_runai.sh
