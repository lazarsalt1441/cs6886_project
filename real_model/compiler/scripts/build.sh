#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
cmake -S . -B build -G Ninja -DMLIR_DIR="${MLIR_DIR:-$PWD/toolchain/usr/lib/llvm-19/lib/cmake/mlir}" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
