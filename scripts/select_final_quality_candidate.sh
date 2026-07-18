#!/usr/bin/env bash
set -euo pipefail

RUN_BASE="${RUN_BASE:-outputs/review_repair}"
DEV_BASE="${DEV_BASE:-outputs/review_repair/followup_development_eval_v2}"
OUT="${OUT:-outputs/review_repair/final_quality_selection_v2}"
BASELINE_RUN_ROOT="${BASELINE_RUN_ROOT:-$RUN_BASE/corrected_cap8a02_seed42_cleanrerun}"
BASELINE_DEV_ROOT="${BASELINE_DEV_ROOT:-$DEV_BASE/baseline_cap8_clean}"

python -m scripts.select_multimodal_quality_followup \
  --run-root "baseline_cap8=$BASELINE_RUN_ROOT" \
  --run-root "speech3rank49=$RUN_BASE/followup_speech3rank49_seed42" \
  --run-root "speech3rank19=$RUN_BASE/followup_speech3rank19_seed42" \
  --run-root "hard19=$RUN_BASE/followup_hard19_seed42" \
  --run-root "balanced19=$RUN_BASE/followup_balanced19_seed42" \
  --run-root "speechlight49=$RUN_BASE/followup_speechlight49_seed42" \
  --dev-eval-dir "baseline_cap8=$BASELINE_DEV_ROOT" \
  --dev-eval-dir "speech3rank49=$DEV_BASE/speech3rank49" \
  --dev-eval-dir "speech3rank19=$DEV_BASE/speech3rank19" \
  --dev-eval-dir "hard19=$DEV_BASE/hard19" \
  --dev-eval-dir "balanced19=$DEV_BASE/balanced19" \
  --dev-eval-dir "speechlight49=$DEV_BASE/speechlight49" \
  --output-dir "$OUT"
