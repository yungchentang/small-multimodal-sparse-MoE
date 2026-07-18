#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
DATA_DIR="${DATA_DIR:-data/real_subset_clean_260708b}"
BEST_ROOT="${BEST_ROOT:-outputs/sparse-moe-mm-rank49-fullbank-cap7-260709next}"
E6_ROOT="${E6_ROOT:-outputs/sparse-moe-clean-e6-feasibility-260709a}"
REPORT_DIR="${REPORT_DIR:-report}"
STAMP="${STAMP:-260709next}"
CHECK_RUNAI="${CHECK_RUNAI:-0}"

cd "$PROJECT_ROOT"

require_file() {
  local path="$1"
  if [ ! -s "$path" ]; then
    echo "missing file: $path" >&2
    return 1
  fi
}

require_dir() {
  local path="$1"
  if [ ! -d "$path" ]; then
    echo "missing dir: $path" >&2
    return 1
  fi
}

require_dir "$DATA_DIR"
require_file "$DATA_DIR/manifest.json"
require_file "$DATA_DIR/text_blocks_train.jsonl"
require_file "$DATA_DIR/image_captions.jsonl"
require_file "$DATA_DIR/speech_transcripts.jsonl"
require_dir "$BEST_ROOT"
require_file "$BEST_ROOT/E3_final_multimodal_top2/checkpoint_final.pt"
require_file "$BEST_ROOT/summary.json"
require_file "$BEST_ROOT/requirement_audit.json"
require_dir "$E6_ROOT"
require_file "$E6_ROOT/summary.json"
require_file "$REPORT_DIR/report.md"
require_file "$REPORT_DIR/report.pdf"

python - "$DATA_DIR" "$BEST_ROOT" <<'PYCOUNTS'
import json
import sys
from pathlib import Path

data_dir = Path(sys.argv[1])
best_root = Path(sys.argv[2])
for name in ["text_blocks_train.jsonl", "text_blocks_eval.jsonl", "image_captions.jsonl", "speech_transcripts.jsonl"]:
    with (data_dir / name).open(encoding="utf-8") as handle:
        print(f"{name}: {sum(1 for _ in handle)}")
audit = json.loads((best_root / "requirement_audit.json").read_text(encoding="utf-8"))
failed = [row for row in audit.get("checks", []) if not row.get("passed")]
print(f"audit_status: {audit.get('status')} failed={len(failed)} checks={len(audit.get('checks', []))}")
if failed:
    for row in failed[:10]:
        print(f"failed: {row.get('name')} {row.get('detail')}")
    raise SystemExit(1)
PYCOUNTS

python -m py_compile scripts/eval_conditional_retrieval.py scripts/eval_encoder_baselines.py scripts/collect_eval_controls.py scripts/summarize_prefix_controls.py

bash -n run.sh scripts/submit_runai.sh scripts/submit_eval_control_sweep.sh \
  scripts/submit_autoresearch_wave_260709.sh scripts/analyze_autoresearch_wave_260709.sh \
  scripts/preflight_autoresearch_wave_260709.sh

echo "planned_jobs:"
for job in \
  sparse-moe-mm-rank49-fullbank-cap7 \
  sparse-moe-mm-prefix75-balanced-cap8 \
  sparse-moe-mm-whisper-space-cap6 \
  sparse-moe-route-z3e5-cap7-img15 \
  sparse-moe-route-drop02-cap7-img15 \
  sparse-moe-route-dynbias-cap7-img15 \
  sparse-moe-mm-hardtext-rank19-cap7 \
  sparse-moe-mm-routerlite-cap7-img15 \
  sparse-moe-mm-frozenrouter-cap7-img15 \
  sparse-moe-top2-distill-hidden-cos \
  sparse-moe-top2-distill-longce \
  sparse-moe-top2-distill-gammatrain; do
  echo "${job}-${STAMP}"
done
eval_control_labels=(
  5way-real-stride
  5way-zero-stride
  5way-random-stride
  5way-shuffled-stride
  5way-noprefix-stride
  5way-encoder-stride
  10way-real-stride
  10way-zero-stride
  10way-random-stride
  10way-shuffled-stride
  10way-noprefix-stride
  10way-encoder-stride
  10way-hardtext
  50way-real-full
  50way-zero-full
  50way-random-full
  50way-shuffled-full
  50way-noprefix-full
  50way-encoder-full
  250way-real-full
  250way-zero-full
  250way-random-full
  250way-shuffled-full
  250way-noprefix-full
  250way-encoder-full
)
for label in "${eval_control_labels[@]}"; do
  echo "sme-eval-${label}-${STAMP}-evalctl"
done

if [ "$CHECK_RUNAI" = "1" ]; then
  runai list jobs -p "${PROJECT:?PROJECT is required when CHECK_RUNAI=1}"
fi
