#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
build/gqa-opt ir/linalg.mlir --gqa-head-reuse --canonicalize -o ir/optimized.mlir
for variant in baseline optimized; do
  input=ir/optimized.mlir
  if [[ "$variant" == baseline ]]; then input=ir/linalg.mlir; fi
  iree-compile "$input" --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-target-cpu=host \
    --iree-hal-dump-executable-files-to="ir/cpu-$variant" -o "results/cpu-$variant.vmfb"
  iree-compile "$input" --iree-hal-target-backends=cuda --iree-cuda-target="${CUDA_ARCH:-sm_80}" \
    --iree-hal-dump-executable-files-to="ir/cuda-$variant" -o "results/cuda-$variant.vmfb"
done
