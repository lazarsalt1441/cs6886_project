# Phase-safe attention compiler research artifact

> Experiment-stage record. See the repository root README for the final GPU evaluation and current reproduction entry point.

The study is **Phase-Safe Attention Compilation: Reduction Legality and Latent Projection Placement**. Read `paper/manuscript.pdf` (editable LaTeX beside it). This document records the CPU projection and reduction-legality study. Weight absorption is established prior work; no GPU execution is reported.

## Results and boundaries

- An independent native-IREE audit fails **14 of 20** shape/alias cases on the measured CPU compiler. The seed-free two-token example returns **0.599999964 instead of 0.9**. No custom projection transformation is needed to reproduce it.
- A real C++ configured-IR phase/resource guard repairs **20/20** cases; worst absolute error against float64 is **9.27e-7**. It preserves complete feature reductions before online-softmax updates.
- Compiled projection placement: **3.93x** geometric mean across 16 decomposed CPU operators against stronger upstream flags. Four partial-rotary cases: **11.40x**. These compare with unabsorbed latent-history graphs, not manually optimized MLA.
- A four-candidate calibration/holdout portfolio: **15.84x** against correct native attention, **4.82x** against the correct unprojected materialized alternative. The separate baseline matters: much of the first gain is avoiding a poor CPU fused schedule.
- **36 structural tests**, additional numerical suites, range-overflow counterexample, negative contracts, positive/negative/zero explicit-scale executions. Synthetic f32 operator inputs; no complete pretrained model.
- Offline CUDA sm_80 compilation: **54/54** binaries, 27 pairs. LLVM IR/PTX retained. No GPU correctness, timing, memory bandwidth, or traffic claim.
- The original GQA head-copy optimization is reproduced by existing strong IREE options. Its initial speedup is not evidence of a new research contribution.

## Installation and short verification

Ubuntu 24.04 x86_64, Python3.12, GCC13, ~2GiB SDK storage, network access required. From `compiler/` in the repository checkout:

```bash
bash scripts/bootstrap.sh
source .venv/bin/activate
source scripts/env.sh
python tests/test_structural.py
python research/test_projection_pass.py
python research/test_online_guard.py
python research/analytic_counterexample.py
python research/guard_validation.py
```

The prebuilt paper PDF is included. Rebuilding it with `scripts/build_paper.sh` additionally requires pdflatex (TeX Live); Poppler is used for visual QA.

`bootstrap.sh` installs pinned Python wheels, verifies downloaded MLIR19 SDK SHA256s, compiles `build/gqa-opt`, and exports the original baseline. Research structural fixtures are included. Do not install legacy `iree-compiler`/`iree-runtime` distributions alongside the pinned `iree-base-*` distributions.

The compiler wheel reports revision `89ece87b89852100c0a4ac3d515e14842d090967`, LLVM24git. C++ pass SDK is MLIR19.1.1. The inspected source checkout `f1eb9ce728e5cf9553307865532807673142e775` differs from that binary. Results are grounded in executable reproductions, not an assumption that these commits match.

## Reproduce the experiments

Run scripts sequentially from the repository root; redirect output to logs. Compiling/timing concurrently invalidates controlled comparisons. Scripts overwrite their named result files: copy measured results or use a new Git branch before rerunning.

```bash
python research/native_attention_audit.py > research/logs/repro-audit.log 2>&1
python research/analytic_counterexample.py > research/logs/repro-analytic.log 2>&1
python research/guard_validation.py > research/logs/repro-guard.log 2>&1
python research/projection_study.py > research/logs/repro-placement.log 2>&1
python research/native_projection_study.py > research/logs/repro-native.log 2>&1
python research/rotary_projection_study.py > research/logs/repro-rotary.log 2>&1
python research/range_safety_study.py > research/logs/repro-range.log 2>&1
python research/explicit_scale_validation.py > research/logs/repro-scale.log 2>&1
python research/placement_ablation.py > research/logs/repro-ablation.log 2>&1
python research/fallback_comparison.py > research/logs/repro-fallback.log 2>&1
python research/portfolio_study.py > research/logs/repro-portfolio.log 2>&1
python research/final_timings.py > research/logs/repro-timings.log 2>&1
python research/final_ablation_timings.py > research/logs/repro-ablation-timings.log 2>&1
python research/make_paper.py
(cd research/paper && pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex > build.log && pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex >> build.log)
```

Primary timings: `results/final-timings.json`, 27 groups, 21 randomized-order observations, five warmups, CPU0, local-sync, uploaded inputs resident, output downloads excluded, Python invocation included. `final-ablation-timings.json` confirms six ablations and three correct fallback comparisons. Early exploratory timing is retained but is not the paper's headline data. Compilation time is recorded separately in study JSON. Timing-bootstrap intervals condition on the fixed workload set and cannot capture shared-host interference or new-machine uncertainty.

Allocation counters measure invocation buffer bytes, with inputs already uploaded. They are distinct from peak process memory and hardware memory transactions. Some legacy peak counters include allocator history and are not used to claim isolated-program peaks.

## Real C++ passes and integration

- `src/GQAHeadReuse.cpp`: original exact head expansion matcher; strong upstream configuration already covers the demonstrated pattern.
- `src/AttentionProjection.cpp`: `--attention-project-reassociate`, opt-in f32 reassociation with MAC profitability, partial rotary sharing, and experimental native-attention support. `src/AttentionBounds.h` provides conditional range screening. Native scale properties are preserved.
- Same source: `--attention-cpu-safe-lowering`, supported unmasked native attention to standard Linalg, with score materialization.
- `src/OnlineAttentionGuard.cpp`: `--attention-guard-online-reductions`, configured CPU schedule repair. Feature axes come from affine maps; complete vector contraction volume is bounded by shrinking legal query/token tiles.

`attention.allow_reassociation` is required for projection changes. It permits rounding changes, not bitwise equality. Native projection additionally needs `attention.enable_native_projection`. Optional `attention.require_range_safety` plus `attention.input_abs_bounds` refuses rewrites with unknown or overflow-prone bounds. Caller bounds must be valid; this is not a certified model-wide error guarantee.

IREE24 and MLIR19 do not share a private C++ ABI. `guard_config.py` uses IREE's structured parser to isolate compatible tensor operations, executes the C++ pass, and copies only the changed lowering configuration back. It does not make the schedule decision in Python. `cpu_compile.py` stops and resumes IREE at `executable-configurations`.

## GPU execution tomorrow

Use a Linux NVIDIA GPU host with a working driver. No local CUDA toolkit is needed for IREE's offline LLVM/PTX compilation. Start with `nvidia-smi`; choose the actual device architecture, for example `sm_80` for A100. Do not use sm_80 blindly for every GPU. On your own machine, consult the installed IREE compiler's supported target list if it rejects an architecture.

```bash
source .venv/bin/activate
source scripts/env.sh
nvidia-smi
iree-run-module --list_drivers
python research/gpu_research_suite.py --full --arch sm_80 \
  > research/logs/gpu-runtime.log 2>&1
```

For a quick three-case run, omit `--full`. To test compilation without a GPU, add `--compile-only` (already executed for all 27 pairs). Input baseline/optimized MLIR is included; the runner rebuilds all binaries for your target. Results are `research/results/gpu-sm_80-runtime.json`. Preserve the compile-only record as compilation evidence only.

The runner checks against CPU PyTorch float64 before timing, records failures, and computes speedups only when both IREE outputs pass. Timings include host invocation, GPU execution and output download, with `.to_host()` synchronization; inputs remain uploaded. They are **not kernel-only latency**. For an eager PyTorch GPU comparison install a compatible CUDA-enabled PyTorch wheel in a separate environment and add `--torch-cuda`; CPU-only PyTorch cannot supply that comparison. The optional GPU-only branch has not been executed here.

PTX/LLVM IR and per-kernel benchmark fixtures are emitted under `research/results/gpu/<arch>/<case>/<variant>-codegen/`. Use those fixtures with a matching IREE build or profile the runner with Nsight Systems/Compute. Profile baseline and optimized separately and report GPU name, clocks/power, warmup, synchronization, transfer inclusion, precision and reference errors. Do not infer DRAM bytes from static load/store sites. `inspect_gpu_codegen.py` records static code inventory only.

## Failures and artifact integrity

Naive feature untile caused vector-size failures; independently bounding operand vectors was also insufficient. Retained logs show these failed approaches. Complete contraction-volume guarding passes the measured suite. The repaired fused kernel can still be much slower than the standard-Linalg fallback.

Some outputs were observed empty during long runs; cause unknown. Compiler output publishing now uses unique temporary paths and atomic replacement; VMFB loading copies the buffer. Empty IR is never accepted as an input. Six empty optional HAL dumps from the legacy study are omitted and listed in `results/legacy-empty-artifact-omissions.json`; regenerate them from adjacent source MLIR after bootstrap. Final source-IR integrity and numerical hashes are recorded; generated binaries can be rebuilt. SDK/vendor/build trees are excluded from the archive.

See `UPSTREAM_ISSUE_DRAFT.md`, `NOVELTY_AND_GAPS.md`, `STATUS.md`, and root `RESUME.md` for the reproducible bug, research limits, checkpoints, and next work. No issue was posted externally.
