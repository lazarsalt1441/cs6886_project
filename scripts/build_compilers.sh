#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo_root"
if [[ -z "${MLIR_DIR:-}" ]]; then
  python real_model/compiler/scripts/fetch_mlir.py
  export MLIR_DIR="$repo_root/real_model/compiler/toolchain/usr/lib/llvm-19/lib/cmake/mlir"
  if [[ ! -e "$repo_root/compiler/toolchain" ]]; then
    ln -s ../real_model/compiler/toolchain "$repo_root/compiler/toolchain"
  fi
fi
sdk_lib=$(cd "$MLIR_DIR/../.." && pwd)
export LD_LIBRARY_PATH="$sdk_lib:$repo_root/real_model/compiler/toolchain/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
for project in compiler real_model/compiler; do
  cmake -S "$project" -B "$project/build" -G Ninja -DMLIR_DIR="$MLIR_DIR" -DCMAKE_BUILD_TYPE=Release
  cmake --build "$project/build" -j "${BUILD_JOBS:-2}"
done
