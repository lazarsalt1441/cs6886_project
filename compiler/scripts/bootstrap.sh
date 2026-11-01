#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python scripts/fetch_mlir.py
scripts/build.sh
python scripts/export_gqa.py
iree-opt ir/torch.mlir --torch-function-to-torch-backend-pipeline \
  --torch-backend-to-linalg-on-tensors-backend-pipeline --canonicalize -o ir/linalg.mlir
python tests/test_structural.py
