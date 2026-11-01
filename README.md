# GQA and Attention Optimization in IREE

CS6886 project by Nishanth Senthil Kumar (EE23B049) and Dhruv Prasad (EE23B130), IIT Madras.

We implemented C++ MLIR passes to reduce repeated attention computation and unnecessary KV-cache copies in IREE.

[Final report (PDF)](report/CS6886_GQA_Attention_Report.pdf) · [LaTeX source](report/CS6886_GQA_Attention_Report.tex) · [Per-study results](report/study_results/per_study_results.md)

## What we implemented

We started with GQA head expansion: several query heads share the same keys and values, but the exported graph repeats K/V before attention. Our pass removed those repeats. We then found that two existing IREE flags already remove them on the tested graphs:

```text
--iree-opt-generalize-matmul
--iree-dispatch-creation-enable-aggressive-reshape-movement
```

The later experiments use these flags as part of the baseline. They cover:

- **Cache-read forwarding:** read directly from the old cache or its small update instead of constructing a temporary updated cache. We implemented conditional and masked-gather forms.
- **Update coalescing:** combine adjacent updates before forwarding reads.
- **Winner-cache factoring:** select between small candidate updates before appending the winner to a shared history.
- **Projection placement:** move key/value projections across matrix products when shapes and numerical assumptions allow it, including graphs with separate rotary coordinates.
- **Correctness repairs:** preserve old cache snapshots, normalize gather indices, and handle problematic attention schedules and large RoPE angles.

The array identities and projection absorption have prior work. The project studies their implementation, upstream coverage and performance in IREE. See the [audit](audit/TECHNICAL_REPORT.md) for details.

## Results

The model experiment runs one step of a custom two-candidate decoder using SmolLM2-135M-Instruct (30 layers, FP32) on a Tesla T4. Winner-cache factoring matches K and V commits in all 30 layers.

| Cached tokens | IREE baseline | With commit factoring | Latency reduction | Total DRAM change |
|---:|---:|---:|---:|---:|
| 2048 | 43.264 ms | 31.815 ms | 26.46% | 3.91% lower |
| 3072 | 63.326 ms | 60.386 ms | 4.64% | 1.24% higher |
| 4096 | 82.409 ms | 77.858 ms | 5.52% | 0.27% lower |

Both variants use the same compiler settings and numerical repairs. Times are medians of nine repetition medians across three prompts. Compilation, prefill, upload and profiling are excluded. All nine natural cases and eighteen forced-winner checks per backend passed.

These results measure a single branching step. They do not measure full generation or serving throughput. The 2048-token improvement appears in every paired repeat, but its kernel-level cause remains unresolved. DRAM traffic does not consistently fall in the model experiment.

An isolated sparse-cache benchmark reduced DRAM traffic by 99.92%, with a 1.44× speedup. That result applies to the constructed cache workload. The dense no-rewrite control showed unstable timing despite equivalent IR.

[Analysis and uncertainty](evaluation/results/ANALYSIS.md) · [Raw GPU results](results/dram) · [Follow-up experiments](evaluation/README.md)

The follow-up GPU runner is prepared but has not been executed.

## Inspect the saved results

```bash
git clone https://github.com/lazarsalt1441/cs6886_project.git
cd cs6886_project
python3 scripts/unpack_artifacts.py
python3 scripts/check_submission.py
python3 scripts/reproduce_results.py
python3 report/summarize_studies.py
```

These commands unpack the saved artifacts, check the submission and regenerate summaries from recorded results. They do not need a GPU or model download, and they do not rerun inference. The analysis checks 112 final GPU profile CSVs and writes regenerated summaries under `reproduced/`.

Archive checksums and campaign names are recorded in `results/archive_provenance.json`. Track A's uploaded archive had B in its filename; its internal plan identifies it as A. Duplicate B uploads count as one run. Raw timing samples, exported counters and failures are retained; weights and profiler binaries are excluded.

## CPU validation and offline GPU compilation

Tested reproduction platform: Python 3.12, x86-64 Linux. SDK bootstrap requires Ubuntu 24.04-compatible glibc, `curl`, `dpkg-deb`, a C/C++ compiler and development headers (for example `build-essential`, `zlib1g-dev`, `libzstd-dev`, `libxml2-dev`). The old IITM glibc 2.17 host cannot run these modern wheels directly; use a suitable container/compute environment or Colab. Do not run benchmarks on a cluster login node.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
python -m unittest discover -s dram/tests -v
python dram/tests/cpu_integration.py --cache local_validation --compiler .venv/bin/iree-compile
python dram/tests/compile_gpu.py --cache local_validation --arch sm_75 --compiler .venv/bin/iree-compile
```

The unit suite has 25 tests. CPU integration executes saved C++-generated baseline/optimized synthetic IR against independent FP64 NumPy references, verifies caller inputs, and retains the quarantined single-fork ownership probe. CUDA compilation is **not GPU execution**. These commands do not require Torch or pretrained weights; Turbine/Transformers are required only for model regeneration/execution.

## Build and test the C++ passes

The standalone MLIR19 driver rewrites tensor IR before IREE dispatch creation; the generated standard MLIR is consumed by IREE 3.13 (embedded compiler commit `89ece87b89852100c0a4ac3d515e14842d090967`, LLVM24). The driver runs before IREE compilation and emits standard MLIR.

```bash
# From repository root, in the environment above:
python -m pip install cmake==4.0.3 ninja==1.13.0
bash scripts/build_compilers.sh
python compiler/tests/test_structural.py
python real_model/tests/test_concat_factor.py
python real_model/tests/test_concat_numerical.py
python real_model/experiments/stable_trig/test_trig.py
```

The build downloads hash-pinned LLVM/MLIR19.1.1 Debian packages without root. Set `MLIR_DIR` to an existing compatible SDK to avoid download. Full legacy test suites also expect `FileCheck` under their `toolchain/` paths. Unsupported types, shapes, conditions, encoding/lowering configurations and unsafe uses are rejected by the individual passes. Negative/idempotence and numerical tests are included. SDK and build directories are ignored.

For the original Torch export, install `torch==2.11.0` from the appropriate CPU/CUDA index separately, then use `compiler/scripts/export_gqa.py` and `compiler/README.md`. Historical recorded environment files remain unchanged.

## Run the final GPU experiments in Colab

Use **Runtime → Change runtime type → GPU**. Match the recorded environment (Torch 2.11 CUDA, T4 sm_75) for direct replication. Other GPUs are new experiments; record them separately. Nsight Compute must be available and permitted to read performance counters. A GPU alone is insufficient to guarantee profiler permission. Use B for the project optimization; a separate session can run A as an ordinary-decode/no-match control, or repeat B in an independent session.

Cell 1:

```python
!git clone https://github.com/lazarsalt1441/cs6886_project.git /content/cs6886_project
!python /content/cs6886_project/scripts/unpack_artifacts.py
from google.colab import drive
drive.mount('/content/drive')
```

Cell 2:

```python
from pathlib import Path
TRACK = 'B'               # Control: 'A'; independent repeat: 'B'
PLAN = 'main'             # Automatically runs smoke first; 'extended' is larger
DRAM_KIT_ROOT = '/content/cs6886_project/dram'
exec(Path(DRAM_KIT_ROOT, 'COLAB_RUN.py').read_text())
```

New runs collect IREE latency and traffic only; PyTorch supplies numerical references and the streaming counter control. Original datasets retain the extra backends collected by the earlier protocol.

The runner installs/checks pinned IREE/Transformers packages, retains CUDA Torch, downloads and SHA256-verifies the pinned checkpoint, generates parameters, builds executables on the GPU runtime, validates before timing, then profiles separate inference ranges. It rejects a mismatched Torch version rather than silently changing the baseline. Match Torch 2.11 deliberately if Colab's default changes. No venv creation is needed in these cells.

It backs up partial/final archives to `MyDrive/Attention_DRAM_Results` and attempts a browser download. Drive authorization is interactive. Browser downloads and Colab session duration are not guaranteed; keep the session connected and laptop awake until a backup appears. Main can take tens of minutes or longer; extended can take hours. Compilation is not included in step latency. Do not run both tracks simultaneously on one GPU. Keep the entire `attention_dram_*.tar.gz`, including failures.

Both variants use O3, matmul generalization, aggressive reshape movement, the same split-argmax workarounds, and the same f64-preservation flags. Timing uses resident outputs, explicit device-context synchronization and randomized backend ordering. Nsight ranges use `app-range` replay; kernel-level diagnostics are kept separate. Logical memcpy payload is not added to DRAM counter totals. Full protocol: [dram/README.md](dram/README.md).

## Regenerate model IR from PyTorch

The primary GPU runner uses saved, checksummed C++-generated IR; no SDK build is needed in Colab. To regenerate it independently, use the C++ build above, install CPU/CUDA `torch==2.11.0` followed by `python -m pip install -r requirements.txt`, and run:

```bash
python real_model/scripts/download_model.py
python real_model/scripts/prepare.py --set A
python real_model/scripts/prepare.py --set B
python real_model/experiments/stable_trig/prepare_repair.py
```

The checkpoint is `HuggingFaceTB/SmolLM2-135M-Instruct` revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`. No weights are committed. Export uses natural HF eager attention and a range-proved legalization of unsupported symbolic guard arithmetic. The fixed prepared assets used for the recorded DRAM results remain under `dram/assets`; compare a regenerated export before changing those assets and their checksum manifest.

## Repository contents

| Directory | Contents |
|---|---|
| `compiler/` | Original GQA driver, cache/projection passes, structural tests, CPU studies and GPU follow-up |
| `real_model/` | Pinned pretrained model wrappers/export, concat tests, original benchmarks and stable-trig extension |
| `dram/` | Final correctness-gated profiling and separately timed GPU study; two Colab notebooks |
| `audit/` | Independent upstream/flag/novelty audit, reproducers and source evidence |
| `results/` | Returned GPU correctness, timing, environment and counter evidence, including failures |
| `artifacts/` | Checksummed compressed generated IR/logs and larger evidence files; unpack with the provided script |
| `scripts/` | Submission integrity, C++ build and raw-result reanalysis |

[STATUS.md](STATUS.md) summarizes the implementation and evaluation; [RESUME.md](RESUME.md) records the reproduction environment. [DOCUMENTATION.md](DOCUMENTATION.md) distinguishes the current reproduction guides from stage-specific experiment records. The final evaluated GPU results are in `results/dram`.

References: [IREE](https://github.com/iree-org/iree), [MLIR](https://mlir.llvm.org/), [GQA paper](https://arxiv.org/abs/2305.13245), [SmolLM2 checkpoint](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct), [repository](https://github.com/lazarsalt1441/cs6886_project). These compiler transformations build on established tensor identities; the repository does not establish global research novelty.
