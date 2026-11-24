#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_ID="tencent/WePocketModel1.0-2B"

"$PYTHON_BIN" - <<'PY'
import sys
if not ((3, 10) <= sys.version_info[:2] <= (3, 13)):
    raise SystemExit("Use Python 3.10, 3.11, 3.12, or 3.13.")
print("Python:", sys.version.split()[0])
PY

if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# Standard macOS PyTorch wheels support Apple Silicon.
python -m pip install torch==2.10.0 torchvision==0.25.0
python -m pip install -r requirements.txt

mkdir -p vendor models inputs outputs reports
if [[ ! -d vendor/WePocketModel/.git ]]; then
  git clone --depth 1 https://github.com/Tencent/WePocketModel.git vendor/WePocketModel
fi

python -m pip install -r vendor/WePocketModel/inference/requirements.txt
python -m pip install 'huggingface_hub>=1.3.0,<2.0'
hf download "$MODEL_ID" --local-dir models/WePocketModel1.0-2B

echo
echo "Setup complete. Run:"
echo "  .venv/bin/python main.py benchmark"
echo "  .venv/bin/python main.py"
