# CS6886: attention and KV-cache compiler optimizations

This CS6886 project investigates attention and KV-cache optimization using C++ MLIR passes and IREE. The repository contains PyTorch exports, compiler implementations, correctness tests, CPU and Tesla T4 evaluations, an upstream optimization audit, and hardware DRAM measurements.

**Project report:** [PDF](report/CS6886_Report.pdf) · [Editable LaTeX](report/CS6886_Report.tex). Two main pages plus one page of references.

## What the results support

The original repeated-K/V-head rewrite improves the tested default CPU lowering, but existing IREE matmul generalization **together with** aggressive reshape movement eliminates the tested expansion. O3 alone is not the explanation. This is an investigation result, not a claim of a universally new GQA optimization. The extra-scalar-observation graph was an artificial IR counterexample, not a natural-model result. See [the recovered flag study](audit/sources/previous-GQA-flag-ablation.md) after unpacking artifacts, and [the independent audit](audit/TECHNICAL_REPORT.md).

The most useful later transformations concern functional cache versions:

* **Read forwarding:** read the original cache except at indices covered by a small update. Applies to supported sparse/gather consumers of forked caches. It is not a blanket replacement for a full, externally observed updated cache.
* **Winner commit factoring:** `select(c, concat(history,a), concat(history,b))` becomes `concat(history, select(c,a,b))`. The real-model driver chooses between two candidate continuations. Ordinary greedy decoding does not match this pattern.
* Further experiments cover gather normalization, update coalescing, snapshot ownership, projection placement, partial rotary embeddings, and native-attention numerical failures. Their scope and negative results are retained in the historical studies and audit.
* A separate bounded f64 trig argument-reduction pass repairs long-position FP32 RoPE discrepancies under the fixed numerical criterion. Both performance comparators use the same repair and precision flags.

### Final T4 result: custom two-candidate pretrained step

SmolLM2-135M-Instruct, FP32, 30 layers, real prompt-derived caches; three prompts and three timing repetitions per context. Values below are medians of the nine per-repetition medians. Compilation, prefill, upload, and profiling are excluded from latency. DRAM counters cover prompt 0, three independent profiles, three fixed-input calls per range.

| Context | Strong repaired IREE | Commit-factored IREE | Latency reduction | Total DRAM change |
|---:|---:|---:|---:|---:|
| 2048 | 43.264 ms | 31.815 ms | 26.46% | 3.91% lower |
| 3072 | 63.326 ms | 60.386 ms | 4.64% | 1.24% higher |
| 4096 | 82.409 ms | 77.858 ms | 5.52% | 0.27% lower |

These comparisons measure the optimization within IREE: strong repaired IREE versus the same compiler settings with commit factoring. The displayed tables, default plots and regenerated model summary show IREE variants only. Original measurements from all collected backends remain in the raw experiment records for reproducibility.

All nine natural cases and eighteen forced-winner checks per backend passed. This is a custom one-step branching driver, **not** complete beam/speculative generation or a serving throughput benchmark. Lower writes are partly offset by higher reads; these data do not establish consistent full-model DRAM reduction. No-match synthetic timing regressions and changing GPU clocks are retained and limit causal attribution and generalization.

The large synthetic forked-cache case reduces measured IREE DRAM traffic by about 99.92% in both tracks. That demonstrates a mechanism on that fixture; it is not the full-model traffic result. Logical expansion, physical allocation, DRAM bytes and runtime improvement are separate claims.

## Checkout and inspect the saved results

```bash
git clone https://github.com/lazarsalt1441/cs6886_project.git
cd cs6886_project
python3 scripts/unpack_artifacts.py
python3 scripts/check_submission.py
python3 scripts/reproduce_results.py
```

The last command needs no GPU or model download. It independently parses **112 valid final GPU profile CSVs**, verifies their saved byte totals, and regenerates aggregate CSV/JSON files under `reproduced/`. It also regenerates the per-track DRAM summaries. It does not run inference or create new performance measurements. The archive is split into modest Git blobs; the unpacker verifies every part and restores before/after IR, assembly/PTX, raw historical measurements and logs to their original paths.

`results/archive_provenance.json` records original archive SHA256s and campaign names. Track A's original archive happened to contain `B` in its filename; the internal plan identifies it as **A**. The second uploaded B zip was byte-identical to the B tar contents and is not counted as an independent run. Profiler report binaries, weights, SDKs and executable modules are excluded; exported counter CSVs and raw timing samples are included.

## CPU validation and offline GPU compilation

Tested reproduction platform: Python 3.12, x86-64 Linux. SDK bootstrap requires Ubuntu 24.04-compatible glibc and development tools. The old IITM glibc 2.17 host cannot run these modern wheels directly; use a suitable container/compute environment or Colab. Do not run benchmarks on a cluster login node.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
python -m unittest discover -s dram/tests -v
python dram/tests/cpu_integration.py --cache local_validation --compiler .venv/bin/iree-compile
python dram/tests/compile_gpu.py --cache local_validation --arch sm_75 --compiler .venv/bin/iree-compile
```

The unit suite has 24 tests. CPU integration executes saved C++-generated baseline/optimized synthetic IR against independent FP64 NumPy references, verifies caller inputs, and retains the quarantined single-fork ownership probe. CUDA compilation is **not GPU execution**. These commands do not require Torch or pretrained weights; Turbine/Transformers are required only for model regeneration/execution.

## Build and exercise the actual C++ passes

The standalone MLIR19 driver rewrites tensor IR before IREE dispatch creation; the generated standard MLIR is consumed by IREE 3.13 (embedded compiler commit `89ece87b89852100c0a4ac3d515e14842d090967`, LLVM24). This is genuine compiled C++ pattern rewriting, not a source-level PyTorch substitution. It is not an upstream IREE plugin or a new GPU kernel.

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

For the original Torch export, install `torch==2.11.0` from the appropriate CPU/CUDA index separately, then use `compiler/scripts/export_gqa.py` and `compiler/README.md`. The stale unavailable 2.14.1 setup pin has been replaced; historical recorded environment files remain unchanged.

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

The runner installs/checks pinned IREE/Transformers packages, retains CUDA Torch, downloads and SHA256-verifies the pinned checkpoint, generates parameters, builds executables on the GPU runtime, validates before timing, then profiles separate inference ranges. It rejects a mismatched Torch version rather than silently changing the baseline. Match Torch 2.11 deliberately if Colab's default changes. No venv creation is needed in these cells.

It backs up partial/final archives to `MyDrive/Attention_DRAM_Results` and attempts a browser download. Drive authorization is interactive. Browser downloads and Colab session duration are not guaranteed; keep the session connected and laptop awake until a backup appears. Main can take tens of minutes or longer; extended can take hours. Compilation is not included in step latency. Do not run both tracks simultaneously on one GPU. Return the entire `attention_dram_*.tar.gz`, including failures.

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
