#!/usr/bin/env bash
set -euo pipefail

RUN_BASE="${RUN_BASE:-outputs/review_repair}"
DEV_BASE="${DEV_BASE:-outputs/review_repair/candidate_development_eval}"
OUT="${OUT:-outputs/review_repair/corrected_candidate_selection}"

python scripts/select_corrected_candidate.py \
  --run-root "cap7a01=$RUN_BASE/corrected_cap7a01_seed42" \
  --run-root "cap7a02=$RUN_BASE/corrected_cap7a02_seed42" \
  --run-root "cap7a04=$RUN_BASE/corrected_cap7a04_seed42" \
  --run-root "cap8a02=$RUN_BASE/corrected_cap8a02_seed42" \
  --run-root "cap6a02=$RUN_BASE/corrected_cap6a02_seed42" \
  --dev-eval-dir "cap7a01=$DEV_BASE/cap7a01" \
  --dev-eval-dir "cap7a02=$DEV_BASE/cap7a02" \
  --dev-eval-dir "cap7a04=$DEV_BASE/cap7a04" \
  --dev-eval-dir "cap8a02=$DEV_BASE/cap8a02" \
  --dev-eval-dir "cap6a02=$DEV_BASE/cap6a02" \
  --output-dir "$OUT"
