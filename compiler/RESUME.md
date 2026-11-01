# Continuation checkpoint — observation-aware attention caches

Working directory: `/workspace/scratch/4088b657d6f5/gqa-compiler`. Branch: `research-cache-dataflow`. Latest completed research stage:R13; inspect `git log -1 --oneline` for the final checkpoint. Previous R12 commit:`2d77192e78f1dcd9512e3f2c9bc088ad8969f7a3`.

Read `research2/TECHNICAL_REPORT.md`, `research2/README.md`, `research2/RESULTS.md`, `research2/EXTENSIONS.md`, then `research2/GPU_RUN.md`. Earlier `research/` and original M0–M5 artifacts are preserved but are separate studies. Strong upstream flags already solve the original head-repeat proposal; do not present that default-baseline speedup as novel.

## Exact environment

Python3.12.14, PyTorch2.14.1+cpu, IREE compiler/runtime packages3.13.0, compiler `89ece87b89852100c0a4ac3d515e14842d090967` (3.13.0rc20261016, LLVM24), Turbine3.9.0, NumPy2.3.5, ml_dtypes0.6.0, CMake4.4.4, Ninja1.13.2, GCC13. Standalone SDK:MLIR/LLVM19.1.1 under `toolchain/`; source `scripts/env.sh`. CPU AMD EPYC9V74,8GiB cgroup,8CPU quota. No GPU.

`vendor/iree` checks out the measured compiler commit. LLVM source pin for a future source build is `32d702efa29d0fdd02cff7b79dea8fdbbcf059d4`. The separately fetched upstream HEAD `9ea0d950d7354009c247159a607e85f92d912d1d` has both implicated source files byte-identical to the measured release; this is not an executed HEAD build. The optional IREE3.12 compiler is isolated under `toolchain/iree-3.12`; select it only with a scoped PYTHONPATH. It also reproduces snapshot corruption.

## Working implementation and results

Seven genuine C++ passes, six new source files. Cache reads, normalized gather indices, masked gathered updates, winner-commit factoring, adjacent update coalescing, early snapshot protection, and phase-local borrowed-storage COW repair. The main driver composes coalescing→factoring→snapshot protection→forwarding. Auto chooses first compilable representation, not fastest.

100 structural unittest cases and two COW FileCheck cases pass. 556 bitwise dataflow/commit/snapshot executions pass.40 expected stock snapshot failures are recorded separately. Nine attention shapes, two synthetic full blocks, two escaping-cache controls, runtime-winner attention, dynamic positions and update-chain extensions have executed correctness checks. Main14-group CPU study and separate5-group extension study retain raw holdout timings. Full blocks2.39×/7.90×, winner3.60×; update chains4–16tokens2.17–2.34×; corrected snapshot attention2.58× vs fastest correct baselines tested. Regressions/no-op controls retained.19groups are two separate isolated runs; do not invent an aggregate cross-device result.

Stock IREE misvectorizes explicit unit-component gather indices and corrupts directly returned old-cache snapshots. Compiled C++ fixes execute through the actual IREE compiler/runtime. The snapshot Stream repair is applied at the pre-COW verifier checkpoint, with local-scope generic printing to preserve opaque attribute syntax. It repairs actual GQA and passes80 exact checks. A snapshot-barrier preprocessing alternative also works. Wrong kernels are never timing baselines.

GPU offline compilation:43/44 main forms,4/4 winner forms,21/21 extension forms. The irregular masked-gather case fails bufferization; conditional fallback compiles. Prepared sm_80 COW IR is included. No GPU numerical/timing/VRAM/DRAM-counter result exists.

## Successful commands

From repository root:

```bash
source scripts/env.sh
scripts/build.sh
python research2/run_structural_tests.py
python research2/minimal_gather_repro.py --tag recovery
python research2/minimal_cache_snapshot.py
python research2/minimal_cache_snapshot.py --repair early
python research2/minimal_cache_snapshot.py --repair cow
python research2/validate_coalescing_bits.py
python research2/validate_snapshot_bits.py
python research2/snapshot_study.py
python research2/snapshot_read_study.py
python research2/probe_update_chain.py
python research2/coalesce_study.py
# No compiler builds or other experiments during timing:
python research2/extension_timings.py
python research2/analyze_extensions.py
python research2/extension_gpu_suite.py --compile-only --arch sm_80
python research2/extension_gpu_suite.py --compile-only --family snapshot --prepared-ir
python scripts/package_project.py
```

Regenerate main14-group executables before rerunning `research2/final_timings.py`; the ordered commands are in README. The measured binaries'76 main/winner hashes were verified. Research2 contains no empty MLIR/VMFB artifacts.

## Source-build limitation and recovery

There is no active full IREE source build. A6-job build exceeded8GiB on NVVMDialect; the failed object then compiled successfully alone and a3-job build continued. A subsequent workspace rollback lost that unfinished build. Code/results were recovered from the saved project archive, packages restored, and the standalone tool rebuilt. The old workspace is preserved in `../gqa-compiler-before-r10-recovery`.

The direct gather source object compiled before rollback, with logs retained. The typed COW source patch is proposed, not compiled into a complete compiler. Both working standalone repairs are compiled and end-to-end verified. Do not confuse those validation levels. Future source build: `BUILD_JOBS=3 research2/build_patched_iree.sh`, then `research2/test_source_patch.sh`. Do not run simultaneous Ninja processes in one build directory. Old historical session IDs are invalid; do not wait on them. Pause builds before timing.

## Next command and next three tasks

Next command on an A100 GPU host, using the supplied IR:

```bash
python research2/gpu_suite.py --arch sm_80
python research2/extension_gpu_suite.py --arch sm_80 --prepared-ir
```

1. Execute GPU correctness/latency, save GPU/driver details, and profile actual memory traffic where available.
2. Build/test both typed source patches in a persistent environment; validate the full source compiler against the supplied native regressions.
3. Review novelty independently and integrate an actual serving model; compare additional compilers/backends. No upstream issue/PR has been posted, and no publication acceptance is claimed.

Paper drafting was deferred by user steering. The unfinished research2/paper source is a draft, not a final manuscript. The completed technical report is Markdown. Package atomically with `scripts/package_project.py`; preserve local Git and replace the existing saved archive rather than creating ambiguous copies.
# 2026-10-27 continuation

See `ablation/REPORT.md` and `ablation/*.py` for the new flag ablation and scalar-observation proof of concept. Compiler 3.13.0 at commit 89ece87b89852100c0a4ac3d515e14842d090967 was installed under `/workspace/scratch/4088b657d6f5/deps` using pip. Run with `PYTHONPATH=/workspace/scratch/4088b657d6f5/deps IREE_COMPILE=/workspace/scratch/4088b657d6f5/deps/bin/iree-compile python ablation/run.py`; correctness and POC commands appear in the report. The existing archive excluded the MLIR19 SDK, and no new C++ rewrite was built for this POC. Next tasks: implement a guarded C++ read-forwarding rewrite; add negative and dynamic-shape tests; test realistic models and GPU against current upstream/native GQA.



## GPU follow-up checkpoint, 2026-10-31
See FOLLOWUP.md for new code, executed tests, Colab commands and limitations. New diagnostic: research/native_width_probe.py; runner: run_gpu_followup.py. CPU: original+decomposed fallback pass 10 seeds; native wide-Q/K probe fails. All nine candidates compile for sm_75; GPU execution pending. Six selection tests pass. Next command on Colab: python run_gpu_followup.py --mode probe --arch sm_75 --seeds 10 --trials 101. Next three tasks: isolate T4 width failure; validate fallback; repeat calibrated cache/winner benchmarks. Local dependencies temporarily at /workspace/scratch/4088b657d6f5/followup_deps; PYTHONPATH must include it for local IREE.
