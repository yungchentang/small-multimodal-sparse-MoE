#!/usr/bin/env bash
set -euo pipefail
CONFIG="${CONFIG:-training/config_smoke.yaml}"
OUT="${OUT:-outputs/smoke}"
python -m training.calibrate_top2 --config "$CONFIG" --output-dir "$OUT"
