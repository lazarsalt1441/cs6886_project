Latest implemented work: [Observation-aware KV-cache report](research2/TECHNICAL_REPORT.md), [reproduction guide](research2/README.md), [GPU commands](research2/GPU_RUN.md). Seven compiled C++ passes; two reproduced correctness defects; CPU benchmarks and GPU compile-only artifacts. GPU runtime and global novelty remain unverified.

# Attention compiler optimization research

**New research artifact:** [Phase-safe attention study](research/README.md), with
[paper PDF](research/paper/manuscript.pdf), editable LaTeX, C++ phase/resource
guard, projection placement, partial rotary handling, numerical contracts and
actual CPU experiments. Independent native attention audit: 14/20 original
failures, 20/20 repaired; seed-free 0.6 versus 0.9 counterexample. Projection
placement: 3.93x geometric mean over 16 strong-baseline CPU cases. GPU offline
compilation: 54/54, no GPU runtime result. This is a research draft, not an
accepted or submission-ready NeurIPS paper.

**Correction to the initial study below:** existing IREE O3 + matmul
generalization + aggressive reshape movement removes the original head expansion
and reaches the same 8192-byte workspace. The original default-baseline 1.86x
measurement remains valid for those settings, but is not a new contribution
against that stronger upstream baseline. Start with the new research guide.

## Original GQA head-reuse study

A compiled C++ MLIR pass that removes materialized KV head repetition from
**decomposed** grouped-query attention. Actual PyTorch exports reproduce the
problem; transformed Linalg compiles and executes through IREE on a CPU.

The original test shape improves **1.86x** in a pinned CPU benchmark, with
first-call buffer allocations falling **267,264 -> 9,216 bytes**. The project
includes 15 numerical/benchmark shapes, nine structural tests, masked attention,
nonzero accumulation, optional loop-layout/fusion experiments, CPU assembly,
LLVM IR and offline NVIDIA sm_80 PTX. One irregular shape regresses to 0.60x.
All raw results are included. No GPU runtime or hardware DRAM-traffic claim is made.

**Scope matters:** IREE already reuses KV heads on the native SDPA GQA path.
This pass addresses an observed gap in explicit repeat/matmul/softmax/matmul
lowering. It removes temporary expansion; it does not implement a persistent KV
cache. Read [REPORT.md](REPORT.md) before presenting novelty or performance claims.

## Quick start: Ubuntu 24.04 x86_64

Prerequisites: Python 3.12, a C++17 compiler (tested GCC13), curl, dpkg-deb, and
network access to PyPI, the PyTorch CPU wheel index and Ubuntu package archives.
Approximately 2 GiB disk is needed for the extracted compiler SDK. No root needed.

```bash
bash scripts/bootstrap.sh
source .venv/bin/activate
source scripts/env.sh
```

Bootstrap creates an isolated Python environment, installs pinned wheels,
fetches SHA256-verified MLIR19 packages, builds `build/gqa-opt`, exports the
reference graph and runs structural tests. The SDK fetch script is specific to
Ubuntu24.04 x86_64; on another system install matching MLIR19 development tools
and set `MLIR_DIR` to their CMake directory before `scripts/build.sh`.

Do not mix the old `iree-compiler` / `iree-runtime` distributions with the new
`iree-base-*` packages in one environment. The bootstrap uses only the latter.

## Build, apply and execute

From the repository root, after activating the environment:

```bash
source scripts/env.sh
scripts/build.sh
python scripts/export_gqa.py
iree-opt ir/torch.mlir --torch-function-to-torch-backend-pipeline \
  --torch-backend-to-linalg-on-tensors-backend-pipeline \
  --canonicalize -o ir/linalg.mlir
build/gqa-opt ir/linalg.mlir --gqa-head-reuse --canonicalize -o ir/optimized.mlir
iree-compile ir/linalg.mlir --iree-hal-target-backends=llvm-cpu \
  --iree-llvmcpu-target-cpu=host -o results/baseline.vmfb
iree-compile ir/optimized.mlir --iree-hal-target-backends=llvm-cpu \
  --iree-llvmcpu-target-cpu=host -o results/optimized.vmfb
python scripts/check_baseline.py
```

The registered function pass accepts exact, static f32 RHS broadcast/reshape
patterns. It splits the flattened query-head axis into KV head and within-group
axes, preserving projected-permutation indexing maps. Unsupported patterns stay
unchanged. Run before IREE dispatch creation. Textual standard Linalg bridges
MLIR19 and the LLVM24-based IREE wheel; this is the standalone integration route,
not an untested source patch to IREE. No full LLVM/IREE source build is required.

## Tests and measurements

```bash
python tests/test_structural.py          # MLIR verifier + FileCheck + negatives
python scripts/run_suite.py             # 15 exports, baseline/pass compilation, correctness, timing
python scripts/extra_correctness.py      # masks and nonzero contraction initialization
python scripts/check_broadcast_frontend.py # unsqueeze/expand/reshape frontend
python scripts/evaluate_extensions.py   # compact-loop C++ pass and fusion variants
python scripts/measure_memory.py        # fresh-process allocator counters + peak RSS
python scripts/rebenchmark.py           # pinned CPU randomized-order timing confirmation
python scripts/investigate_native_gqa.py # native SDPA alternative: export, run, inspect
python scripts/compare_native.py         # pinned comparison with native SDPA
scripts/inspect_codegen.sh              # x86 assembly + offline CUDA PTX
python scripts/make_report.py           # report and plots from retained measurements
```

Or run `scripts/run_all.sh`; verbose output goes to logs. Each numerical case
uses deterministic random inputs. Files under `results/cases/` contain real Torch,
baseline Linalg, optimized Linalg, HAL allocation plans, and compilation logs.
VMFB files are regenerated by the scripts and are ignored by Git. Measurements
are graph-specific, include invocation overhead, and exclude compile time,
input upload and output copying. See REPORT.md for methodology and regressions.

`--gqa-compact-head-reuse` is a second registered C++ pass that merges adjacent
query loops. It is an experimental alternative; it does not consistently improve
the default. Aggressive IREE fusion results are also retained without promoting
a universal tuning setting.

## GPU preparation

Offline compilation was tested for sm_80, without a GPU or CUDA toolkit:

```bash
CUDA_ARCH=sm_80 scripts/inspect_codegen.sh
iree-run-module --device=cuda --module=results/cuda-optimized.vmfb \
  --function=main --input=1x8x1x32xf32=0.1 \
  --input=1x2x128x32xf32=0.1 --input=1x2x128x32xf32=0.1
iree-benchmark-module --device=cuda --module=results/cuda-optimized.vmfb \
  --function=main --input=1x8x1x32xf32=0.1 \
  --input=1x2x128x32xf32=0.1 --input=1x2x128x32xf32=0.1
```

The execution commands require a real CUDA device and compatible runtime and
have **not** been executed here. Repeat with the baseline file and validate
random-input numerical equivalence before trusting GPU performance. Nsight
profiling should distinguish cache hit rates, device traffic, occupancy and
allocation changes. Hardware counter access was denied in this CPU environment.

## Repository map

- `src/GQAHeadReuse.cpp`: genuine C++ passes, match validation and registration.
- `tests/`: positive, negative and dynamic-shape fixtures and FileCheck checks.
- `scripts/`: bootstrap, build, export, integration, correctness, timing and reporting.
- `ir/`: actual exported and transformed IR plus CPU/CUDA code generation dumps.
- `results/`: measurements, errors, raw samples, allocation/RSS and plots.
- `REPORT.md`: technical report with limitations, native-path findings and regressions.
- `STATUS.md`, `RESUME.md`: milestones and exact continuation instructions.
- `docs/toolchain.sha256`: immutable SDK checksums; `requirements.txt`: Python pins.

The packaged repository includes local Git history for M0-M5. Large dependency
SDKs, build artifacts and the inspected IREE vendor checkout are intentionally
excluded and reconstructed by bootstrap. Codegen dumps include observed output
from the exact compiler version in `results/environment.txt`.
# Latest GQA flag ablation

See [ablation/REPORT.md](ablation/REPORT.md) for the 160-configuration study, 96 numerical checks, and a remaining scalar-observation optimization opportunity. The latter is a manual proof of concept, not a general C++ pass.



## GPU follow-up checkpoint, 2026-10-31
See FOLLOWUP.md for new code, executed tests, Colab commands and limitations. New diagnostic: research/native_width_probe.py; runner: run_gpu_followup.py. CPU: original+decomposed fallback pass 10 seeds; native wide-Q/K probe fails. All nine candidates compile for sm_75; GPU execution pending. Six selection tests pass. Next command on Colab: python run_gpu_followup.py --mode probe --arch sm_75 --seeds 10 --trials 101. Next three tasks: isolate T4 width failure; validate fallback; repeat calibrated cache/winner benchmarks. Local dependencies temporarily at /workspace/scratch/4088b657d6f5/followup_deps; PYTHONPATH must include it for local IREE.
