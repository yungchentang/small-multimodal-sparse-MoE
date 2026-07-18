#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
VENV_PATH="${VENV_PATH:-$REPO_ROOT/.venv}"
python -m venv --system-site-packages "$VENV_PATH"
source "$VENV_PATH/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-cluster.txt
python - <<'PY'
import importlib.util
mods = ["torch", "transformers", "datasets", "accelerate", "peft", "PIL", "soundfile", "librosa", "yaml"]
for mod in mods:
    print(mod, importlib.util.find_spec(mod) is not None)
PY
