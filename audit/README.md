# IREE/MLIR novelty audit - 1 November 2026

Read `TECHNICAL_REPORT.md` first, then `EXECUTIVE_SUMMARY.md` for the decision.
This is a bounded, source-backed audit, not a proof about every IREE target or flag combination.

## What is preserved

- `reproducers/`: valid MLIR sources and their input/semantic manifest.
- `results/pinned/` and `results/latest/`: exact commands, four compiler stages, VMFBs, numerical checks, allocation counters and raw timing samples.
- `results/actual_pass/`: independent execution of the recovered original C++ transformations, structural tests, and full-model graph applicability.
- `results/native_feature_tiling_excerpt.mlir`: the feature-axis/online-softmax scheduling failure; the complete trace is also retained.
- `results/diagnostics/`: earlier scalar-i1 bridge compilation failures, retained separately from the corrected final reproducer.
- `project/compiler/`: recovered project pass source, license, CMake build and hash-pinned MLIR19 SDK fetch helper. No SDK or compiler wheel is bundled.
- `sources/`: pinned source snapshots, issue/PR responses and original evidence excerpts.
- `manifest.json`: SHA256 of every included file except the manifest itself.

## Repeat the CPU compiler audit

Run from this directory. Python 3.12 on x86-64 Linux was used.

```bash
python3 -m venv .venv-pinned
.venv-pinned/bin/pip install iree-base-compiler==3.13.0 iree-base-runtime==3.13.0 numpy
.venv-pinned/bin/python scripts/run.py --compiler "$PWD/.venv-pinned/bin/iree-compile" --label repeat-pinned
.venv-pinned/bin/python scripts/run.py --compiler "$PWD/.venv-pinned/bin/iree-compile" --label repeat-pinned --only native --profiles early_decompose
.venv-pinned/bin/python scripts/run.py --compiler "$PWD/.venv-pinned/bin/iree-compile" --label repeat-pinned --only projection,partial_rope --profiles max_reassociation
.venv-pinned/bin/python scripts/run.py --compiler "$PWD/.venv-pinned/bin/iree-compile" --label repeat-pinned --only gather_forks,winner,projection,partial_rope --profiles aggressive
.venv-pinned/bin/python scripts/run.py --compiler "$PWD/.venv-pinned/bin/iree-compile" --label repeat-pinned --only concat --profiles aggressive
.venv-pinned/bin/python scripts/paired_benchmark.py --label repeat-pinned
```

The shipped reproducers include original C++ outputs. Rebuilding that tool is optional when repeating compilation/runtime checks:

```bash
python3 -m pip install cmake ninja
python3 project/compiler/scripts/fetch_mlir.py
(cd project/compiler && bash scripts/build.sh)
python3 scripts/generate.py
python3 scripts/actual_pass.py --project project
python3 scripts/additional.py --project project
```

`generate.py` rebuilds the conceptual base fixtures; the next two commands add the actual C++ outputs back to the manifest. Canonicalized full-model before/after IR is retained under `results/actual_pass/A` and `B`; this is pass-output evidence, not packaged pretrained model weights.

For the latest tested binary use a separate environment:

```bash
python3 -m venv .venv-latest
.venv-latest/bin/pip install numpy
.venv-latest/bin/pip install \
  'https://github.com/iree-org/iree/releases/download/iree-3.14.0rc20261031/iree_base_compiler-3.14.0rc20261031-cp312-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl' \
  'https://github.com/iree-org/iree/releases/download/iree-3.14.0rc20261031/iree_base_runtime-3.14.0rc20261031-cp312-abi3-manylinux_2_28_x86_64.whl'
.venv-latest/bin/iree-compile --version
.venv-latest/bin/python scripts/run.py --compiler "$PWD/.venv-latest/bin/iree-compile" --label repeat-latest
```

Repeat the optional profiles above using that interpreter/compiler, then the paired benchmark. Verify version strings against the report rather than assuming a wheel filename identifies its embedded compiler commit.

## Native attention workaround using upstream passes only

```bash
.venv-latest/bin/iree-compile reproducers/native_256_128.mlir \
  --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-target-cpu=host \
  --iree-opt-level=O3 \
  '--iree-preprocessing-pass-pipeline=builtin.module(util.func(iree-linalg-ext-convert-attention-to-online-attention,iree-linalg-ext-decompose-attention))' \
  -o attention_early_decomposed.vmfb
```

This was validated on CPU. CUDA validation is pending; do not assume it fixes the T4 schedule without execution.

## Complete GPU verification

Use an actual NVIDIA machine with the matching IREE runtime driver and available CUDA compiler support. Keep baseline/custom and both compiler versions on the **same GPU**.

```bash
nvidia-smi > gpu_environment.txt
.venv-latest/bin/python scripts/run.py \
  --compiler "$PWD/.venv-latest/bin/iree-compile" --label t4-latest \
  --backend cuda --target sm_75 --profiles default,O3,strong
.venv-latest/bin/python scripts/run.py \
  --compiler "$PWD/.venv-latest/bin/iree-compile" --label t4-latest \
  --backend cuda --target sm_75 --only native --profiles early_decompose
```

`run.py` CUDA timing explicitly downloads the result to synchronize; it measures host invocation plus output transfer, not device-only latency. It preserves numerical failures and skips their timings. Its exit status is not an all-tests-pass certificate: inspect `summary.json` and per-case checks. For publication, add randomized calibration/holdout ordering to the GPU runner (as in the CPU paired script), CUDA-event/device-side measurements, independent sessions, and Nsight traffic counters. The paired benchmark script is CPU-only.

CPU runs include local-sync synchronous invocation and resident inputs; compilation and output downloads are excluded. Different allocation or ownership contracts must not be pooled. `scripts/snapshots.py` checks an explicit returned-old-cache contract and times only programs preserving that contract.

## Test actual current main

The tested nightly commit is earlier than the inspected main commit by one compiler fix. To close that boundary, source-build main at the pinned SHA:

```bash
git clone https://github.com/iree-org/iree.git iree
git -C iree checkout 29114a017c104882cab055a49758e8f7055ea0d5
git -C iree submodule update --init --recursive
cmake -S iree -B iree-build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DIREE_BUILD_COMPILER=ON -DIREE_BUILD_TESTS=OFF
cmake --build iree-build --target iree-compile -j4
.venv-latest/bin/python scripts/run.py \
  --compiler "$PWD/iree-build/tools/iree-compile" --label main-source-build
```

The source build itself was not executed in this audit. It needs the LLVM source/submodules, build dependencies, and substantially more build resources than installing the released wheels. Match or rebuild the runtime if bytecode/runtime compatibility changes.
