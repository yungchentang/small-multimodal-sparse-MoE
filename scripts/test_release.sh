#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

python -m compileall -q datasets evaluation model scripts tests training hf_sources.py
find . -path ./.git -prune -o -type f -name '*.sh' -exec bash -n '{}' +
python -m unittest -v \
  tests.test_run_modes \
  tests.test_build_final_promotion_selection \
  tests.test_demo_notebook \
  tests.test_hf_sources \
  tests.test_public_evidence_bundle \
  tests.test_all_hf_model_loads_pinned \
  tests.test_sealed_launcher_contract
