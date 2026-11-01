#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
compiler_path="$PWD/vendor/iree-build/tools/iree-compile"
optimizer_path="$PWD/vendor/iree-build/tools/iree-opt"
filecheck_path="$PWD/toolchain/usr/lib/llvm-19/bin/FileCheck"
[[ -x "$filecheck_path" ]] || filecheck_path="$PWD/toolchain/usr/bin/FileCheck-19"
"$optimizer_path" --pass-pipeline='builtin.module(func.func(iree-codegen-generic-vectorization{enable-vector-masking=true}))' \
 --split-input-file research2/patches/gather_index_projection.mlir \
 | "$filecheck_path" research2/patches/gather_index_projection.mlir
python research2/minimal_gather_repro.py --compiler "$compiler_path" --tag source-fixed
python - <<'PY'
import json
assert all(v['correct'] for v in json.load(open('research2/results/minimal-gather/source-fixed/result.json')).values())
PY
python research2/gather_attention_audit.py --compiler "$compiler_path" --tag source-fixed
python - <<'PY'
import json
assert all(v['correct'] for r in json.load(open('research2/results/gather-attention-audit-source-fixed.json')) for v in r['variants'].values())
PY

"$optimizer_path" --split-input-file --pass-pipeline='builtin.module(util.func(iree-stream-materialize-copy-on-write))' \
 research2/patches/borrowed_cast_cow.mlir | "$filecheck_path" research2/patches/borrowed_cast_cow.mlir
python research2/minimal_cache_snapshot.py --compiler "$compiler_path" --tag source-fixed --expect-correct
