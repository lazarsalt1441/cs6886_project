# Attention compiler implementation and operator studies

This directory contains the standalone C++ MLIR driver, original GQA exports,
operator-level attention/cache studies and structural tests. The complete project
setup and final pretrained-model GPU results are in the [root README](../README.md).

The original GQA head-reuse pass removes supported explicit K/V repeat patterns
before IREE dispatch creation. The tested expansion is also eliminated by IREE
matmul generalization together with aggressive reshape movement; the initial
default-lowering speedup is therefore not a new capability over that baseline.
The recovered flag study is in `../audit/sources/previous-GQA-flag-ablation.md`
after artifact extraction. The extra-scalar example is an IR proof of concept.

## Build and structural validation

From the repository root:

```bash
python3 scripts/unpack_artifacts.py
bash scripts/build_compilers.sh
python compiler/tests/test_structural.py
python real_model/tests/test_concat_factor.py
python real_model/tests/test_concat_numerical.py
python real_model/experiments/stable_trig/test_trig.py
```

Install the root README dependencies first. The build uses a hash-pinned
MLIR19 SDK and produces `compiler/build/gqa-opt`. Standard textual tensor IR
connects this driver to IREE3.13; this is a standalone integration, not a complete
in-tree IREE build. The legacy `scripts/bootstrap.sh` is specific to Ubuntu24.04
x86-64; the root build instructions are the common entry point.

## Original PyTorch export and execution

After installing Torch2.11 and the root `requirements.txt`, run from this directory:

```bash
python scripts/export_gqa.py
iree-opt ir/torch.mlir --torch-function-to-torch-backend-pipeline \
  --torch-backend-to-linalg-on-tensors-backend-pipeline --canonicalize -o ir/linalg.mlir
build/gqa-opt ir/linalg.mlir --gqa-head-reuse --canonicalize -o ir/optimized.mlir
iree-compile ir/linalg.mlir --iree-hal-target-backends=llvm-cpu \
  --iree-llvmcpu-target-cpu=host -o results/baseline.vmfb
iree-compile ir/optimized.mlir --iree-hal-target-backends=llvm-cpu \
  --iree-llvmcpu-target-cpu=host -o results/optimized.vmfb
python scripts/check_baseline.py
```

The saved baseline and optimized IR are restored by the unpacker. The numerical
and benchmark scripts in `scripts/` retain the original shape suites, masks,
nonzero accumulation, allocator counters and randomized CPU measurements.
Consult each script's arguments and the [original study record](REPORT.md).

## Studies and scope

| Location | Study |
|---|---|
| `src/` | GQA, cache forwarding, gather/commit/coalescing, snapshot protection, projection and trig passes |
| `tests/` | Original positive, negative, dynamic-shape, multi-use and FileCheck tests |
| `research/` | Projection placement, partial RoPE and native-attention reduction legality |
| `research2/` | Functional cache versions, ownership and CPU allocation/timing experiments |
| `FOLLOWUP.md` | Native-width diagnosis and calibrated GPU follow-up protocol |
| `pbs_gpu/` | PBS compute-node runner and profiler instructions |
| `ir/`, `results/` | Generated evidence restored from checked archives |

Directory reports describe individual experiment stages and their own baselines.
Their original CPU or compile-only results should be interpreted within that
scope. Final T4 measurements and current IREE comparisons are in `../results/dram`.
Unsupported graph patterns remain unchanged. Source-patch proposals and compiled
standalone transformations are identified separately in the study records.
