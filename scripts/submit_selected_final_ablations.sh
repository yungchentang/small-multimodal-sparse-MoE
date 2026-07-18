#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
OUT="${OUT:?OUT is required and must be external to RUN_ROOT}"
PROJECT="${PROJECT:?PROJECT is required}"
CAPACITY_FACTOR="${CAPACITY_FACTOR:?CAPACITY_FACTOR is required}"
AUX_COEF="${AUX_COEF:?AUX_COEF is required}"
DATA_DIR="${DATA_DIR:-data/real_subset_clean_260708b}"
JOB_NAME="${JOB_NAME:-sme-final-ablations}"
ABLATION_EXPERIMENTS="${ABLATION_EXPERIMENTS:-E4,E5}"

for required in \
  "$RUN_ROOT/manifest.json" \
  "$RUN_ROOT/E3_final_multimodal_top2/checkpoint_final.pt"; do
  if [ ! -s "$required" ]; then
    echo "missing final-ablation input: $required" >&2
    exit 2
  fi
done
run_root_abs="$(realpath -m "$RUN_ROOT")"
out_abs="$(realpath -m "$OUT")"
if [[ "$out_abs" == "$run_root_abs" || "$out_abs" == "$run_root_abs/"* ]]; then
  echo "final ablation OUT must be external to RUN_ROOT" >&2
  exit 2
fi
outputs=()
[[ ",$ABLATION_EXPERIMENTS," == *",E4,"* ]] && outputs+=("$OUT/E4_no_aux_load_balance_ablation")
[[ ",$ABLATION_EXPERIMENTS," == *",E5,"* ]] && outputs+=("$OUT/E5_capacity_1p25_ablation")
if [ "${#outputs[@]}" -eq 0 ]; then
  echo "ABLATION_EXPERIMENTS must include E4 and/or E5" >&2
  exit 2
fi
for output in "${outputs[@]}"; do
  if [ -e "$output" ]; then
    echo "refusing to overwrite final-ablation output: $output" >&2
    exit 2
  fi
done

python -c 'import json,sys; p,c,a=sys.argv[1:]; m=json.load(open(p)); d=m["args"]; done=m["completion"]; assert int(d["final_steps"])>0; assert done["status"]=="completed"; assert int(done["e3_steps"])==int(d["final_steps"]); assert float(d["capacity_factor"])==float(c); assert float(d["aux_coef"])==float(a)' \
  "$RUN_ROOT/manifest.json" "$CAPACITY_FACTOR" "$AUX_COEF"

env \
  PROJECT="$PROJECT" JOB_NAME="$JOB_NAME" MODE=ablation-only \
  GPU="${GPU:-1}" CPU="${CPU:-10}" MEMORY="${MEMORY:-120G}" \
  OUT="$OUT" SOURCE_OUTPUT_DIR="$RUN_ROOT" DATA_DIR="$DATA_DIR" \
  FEATURE_CACHE_DIR="$OUT/feature_cache" \
  ABLATION_STEPS="${ABLATION_STEPS:-300}" \
  CAPACITY_ABLATION_STEPS="${CAPACITY_ABLATION_STEPS:-300}" \
  ABLATION_EXPERIMENTS="$ABLATION_EXPERIMENTS" \
  bash scripts/submit_runai.sh
