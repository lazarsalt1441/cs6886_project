#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
source_rev=89ece87b89852100c0a4ac3d515e14842d090967
source_dir="$PWD/vendor/iree"
build_dir="$PWD/vendor/iree-build"
mkdir -p vendor research2/logs
if [[ ! -d "$source_dir/.git" ]]; then
  git clone --no-checkout https://github.com/iree-org/iree.git "$source_dir"
  git -C "$source_dir" checkout "$source_rev"
fi
[[ "$(git -C "$source_dir" rev-parse HEAD)" == "$source_rev" ]] || { echo 'Use the pinned source checkout'; exit 1; }
git -C "$source_dir" submodule update --init --depth 1 third_party/llvm-project third_party/flatcc third_party/benchmark third_party/googletest third_party/musl
for patch_name in iree-gather-index-projection.patch iree-borrowed-cast-cow.patch; do
  patch_path="$PWD/research2/patches/$patch_name"
  if git -C "$source_dir" apply --check "$patch_path"; then
    git -C "$source_dir" apply "$patch_path"
  else
    git -C "$source_dir" apply --reverse --check "$patch_path"
  fi
done
cp research2/patches/gather_index_projection.mlir "$source_dir/compiler/src/iree/compiler/Codegen/Common/test/"
cp research2/patches/borrowed_cast_cow.mlir "$source_dir/compiler/src/iree/compiler/Dialect/Stream/Transforms/test/"
cmake -S "$source_dir" -B "$build_dir" -G Ninja \
 -DCMAKE_BUILD_TYPE=Release '-DCMAKE_CXX_FLAGS_RELEASE=-O1 -DNDEBUG' '-DCMAKE_C_FLAGS_RELEASE=-O1 -DNDEBUG' \
 -DIREE_BUILD_COMPILER=ON -DIREE_BUILD_TESTS=OFF -DIREE_BUILD_SAMPLES=OFF \
 -DIREE_BUILD_PYTHON_BINDINGS=OFF -DIREE_BUILD_BENCHMARKS=OFF \
 -DIREE_TARGET_BACKEND_DEFAULTS=OFF -DIREE_TARGET_BACKEND_LLVM_CPU=ON \
 -DIREE_TARGET_BACKEND_LLVM_CPU_WASM=OFF -DIREE_TARGET_BACKEND_VMVX=OFF \
 -DIREE_HAL_DRIVER_DEFAULTS=OFF -DIREE_HAL_DRIVER_LOCAL_SYNC=ON \
 -DIREE_INPUT_TORCH=OFF -DIREE_INPUT_STABLEHLO=OFF -DIREE_INPUT_TOSA=OFF \
 -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF -DLLVM_PARALLEL_LINK_JOBS=1 \
 -DIREE_ERROR_ON_MISSING_SUBMODULES=OFF -DIREE_DEFAULT_CPU_LLVM_TARGETS=X86 \
 > research2/logs/iree-source-configure-reproduction.log 2>&1
# Run exactly one Ninja process per build directory. Lower BUILD_JOBS for low RAM.
ninja -C "$build_dir" -d keepdepfile -j"${BUILD_JOBS:-3}" iree-compile iree-opt \
 > research2/logs/iree-source-build-reproduction.log 2>&1
