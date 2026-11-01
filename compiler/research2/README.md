# Forked attention-cache dataflow study

This study contains seven compiled C++ MLIR passes and actual CPU executions. It follows the earlier `research/` investigation. It does not establish global novelty, GPU speedup, or publication acceptance.

The measured IREE compiler is `89ece87b89852100c0a4ac3d515e14842d090967` (3.13.0rc20261016; package3.13.0). Source checkout matches the wheel. Standalone passes use MLIR19.1.1; IREE uses LLVM24 at `32d702efa29d0fdd02cff7b79dea8fdbbcf059d4`. The bridge serializes registered IREE IR to generic syntax, runs the C++ tool, then reparses with IREE. No private C++ ABI mixing occurs.

Read [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) for the complete investigation, [RESULTS.md](RESULTS.md) for the14-group study, [EXTENSIONS.md](EXTENSIONS.md) for the5-group update/snapshot study, and [GPU_RUN.md](GPU_RUN.md) to run the supplied IR on a GPU without rebuilding the SDK.

## Implementations

- `../src/CacheReadForwarding.cpp`: `--attention-forward-cache-reads`. Replace scalar reads of functional cache updates with lazy conditional reads from the original cache or update. Supports static positive rank1–5 regions, runtime offsets, and positive static strides. Rejects escaping versions and unsupported consumers.
- `../src/CacheGatherForwarding.cpp`: `--attention-forward-gather-updates`. A fusible representation: gather original cache, translate IDs, then masked-gather the update into that result. Safe inactive IDs, integer-only address computation, no floating-point reassociation. Contiguous complete-row writes and normalized depth1 leading-axis gathers only. Default minimum pool/read element ratio4; the original benchmark ablation explicitly uses ratio2.
- `../src/GatherIndexProjection.cpp`: `--attention-canonicalize-gather-indices`. Remove an optional trailing unit component axis for depth1 gathers. Repairs an independently reproduced IREE vectorization bug. Preserves masks; supports dynamic batch extents.

`patches/iree-gather-index-projection.patch` is the direct IREE source repair: index transfer reads must project leading batch axes. `patches/gather_index_projection.mlir` checks one and two batch axes. The modified source object compiles successfully. Full source build/regression execution status is recorded in `STATUS.md`.

- `../src/CacheCommitFactoring.cpp`: `--attention-factor-cache-commit`. Factor a scalar-condition selection of matching cache writes into selection of their small updates followed by one committed write. Equal bases, offsets, sizes and positive strides are required; different regions and elementwise conditions are rejected. This enables observation forwarding while retaining the actual runtime winner's cache. No floating-point arithmetic is changed. Standard selection/read-through-write algebra is not claimed as a new identity.

- `../src/CacheUpdateCoalescing.cpp`: `--attention-coalesce-cache-updates`. Combine adjacent unobserved per-token writes into one small flattened update, before forwarding. Static leading-axis offsets, unit strides and small updates required; gaps, overlaps, observed intermediate versions and single-use pools are rejected.
- `../src/CacheSnapshotProtection.cpp`: `--attention-protect-cache-snapshots`. Preserve direct returned input caches that also feed functional updates, preventing the demonstrated old-snapshot corruption. `--attention-materialize-borrowed-cache-cow` repairs the phase-local Stream case: cast-hidden non-consuming imports used as async.update targets. The latter is applied at the actual pre-COW checkpoint by `repair_snapshot_cow.py`.

Stock IREE returns updated values for an original-cache output in the native and single-GQA counterexamples. Both compiled C++ repairs preserve the snapshot. The phase-local repair passes the native FileCheck regression and actual GQA execution; the proposed typed source patch is not a built full compiler. Scope is deliberately limited to the supported forms, not arbitrary aliases/control flow.

## Build and quick tests

From the repository root:

```bash
scripts/bootstrap.sh
source scripts/env.sh
python research2/run_structural_tests.py
python research2/test_cache_forwarding.py
python research2/test_gather_projection.py
python research2/test_gather_forwarding.py
python research2/test_commit_factoring.py
python research2/test_update_coalescing.py
python research2/test_snapshot_protection.py
python research2/minimal_gather_repro.py
python research2/minimal_cache_snapshot.py
python research2/minimal_cache_snapshot.py --repair cow
```

Bootstrap downloads CPU PyTorch, pinned IREE/Turbine and the MLIR19 SDK. Existing environments can use `scripts/build.sh`. GCC13 was used here. The minimal native-IREE program requires no Torch export: expected head values254/255, original wheel observed14/15, normalized result254/255. Original code has poison-derived invalid addresses, so its failure is not a defined alternative function.

## Regenerate experiments

Commands below compile and execute; run them before final timing. They overwrite their corresponding result files, so preserve a checkout for historical results.

```bash
source scripts/env.sh
python research2/probe_cache.py
python research2/study_cache.py
python research2/add_patch_variant.py
python research2/block_study.py
python research2/commit_study.py
python research2/complete_fusion_baselines.py
python research2/validate_cache_semantics.py
python research2/dynamic_attention.py
python research2/validate_ranks.py
python research2/validate_commit_bits.py
python research2/probe_update_chain.py
python research2/coalesce_study.py
python research2/validate_coalescing_bits.py
python research2/validate_snapshot_bits.py
python research2/snapshot_study.py
python research2/snapshot_read_study.py
python research2/winner_study.py
python research2/gather_attention_audit.py
python research2/inspect_codegen.py
python research2/inspect_gpu_schedule.py
# Finish or stop all source builds before timing:
python research2/final_timings.py
python research2/extension_timings.py
python research2/analyze_results.py
python research2/analyze_extensions.py
```

`study_cache.py` creates all frontend IR and baseline executables. `add_patch_variant.py` adds masked forwarding. Block and commit scripts create all four main variants; the fusion-control script adds stronger baselines. Results reside under `results/`; logs under `logs/`. Raw explicit-gather code fails6/9 audited configurations, including one crashing child process; it is excluded from performance baselines. Original generic and normalized gather each pass9/9. Numerical audit processes are isolated because bad compiled indices may crash.

The nine attention cases span query/KV ratios1,2,4,8; D24,32,64; query length1,3,16; read length65–512; one to eight branches; capacities256–65536. Two synthetic full transformer blocks include RMSNorm, projections, RoPE, GQA, residuals and MLP with runtime weights. They are not pretrained generation benchmarks. Commit controls return one or every divergent cache version; externally observed versions remain materialized.

556 passing data-movement/commit/snapshot executions test bitwise f32/f16/i32 preservation, NaN payloads, signed zeros, infinities, runtime offsets, strided partial updates and overlapping last-write-wins.40 stock snapshot failures are recorded separately, not counted as passing tests. 100 structural unittest cases plus two COW FileCheck cases pass. 12 dynamic-position attention executions test four forms at three positions. Full floating-point attention uses float64 PyTorch references and records maximum absolute error. The runtime-winner extension checks seven compiler forms with both possible winners and two seeds:28 executions. It returns the attention-derived decision and selected full K/V caches; cache and decision bits match PyTorch. Its invocation allocation is25,168,064 bytes for the fused baseline and8,391,872 bytes for factoring plus masked forwarding. This is a measured allocation reduction; latency is reported only after isolated timing.

## Current measured results

[RESULTS.md](RESULTS.md) contains the14-group holdout table, allocations, raw-data links and plots. Synthetic full-block speedups are2.39× and7.90× against the best correct baseline tested; the attention-derived winner case is3.60×. These are CPU results on one host, with regressions and no-op cases retained. They are not GPU or pretrained-model claims. The full source build exceeded8GiB with6jobs; the failed NVVM source file then compiled successfully alone, followed by a3-job continuation.

## Timing and memory interpretation

`results/final-timings.json` retains every observation. CPU affinity is fixed, local-sync executes synchronously, PyTorch uses one thread, inputs remain resident, outputs are downloaded only for validation. Compilation is separate. Compile-time diagnostic fields were often gathered under concurrent source-build load; no compilation-speed improvement is claimed. Five warmups precede randomized interleaved measurements. Seven calibration observations select an implementation before21 holdout observations on different inputs. Selection is a compile-time calibration experiment, not a portable static heuristic. The best correct baseline is chosen conservatively from available strong, aggressive-fusion and normalized-gather forms.

Allocation bytes are IREE allocator deltas for one invocation after uploading inputs; they include workspaces and outputs. HAL copy lengths are scheduled copy bytes, not measured DRAM traffic. LLVM IR and disassembly in `results/cpu-codegen/` show compiled kernels. Hardware memory-traffic counters were unavailable. Regressions and no-op controls remain in the dataset. A stricter default profitability gate was added after observing the ratio1/2 regression; the ratio2 measurements remain explicitly reproducible and must not be described as independent validation of that revised gate.

## Compile a supplied tensor-IR program

```bash
python research2/compile_attention.py --input path/to/linalg.mlir --output model.vmfb
python research2/compile_attention.py --input path/to/linalg.mlir --output model.vmfb --target cuda --arch sm_80
```

Inputs should be tensor-level frontend or native IREE IR. The driver normalizes index projection, coalesces supported update chains, factors winner commits, protects returned old-cache snapshots, then runs the C++ masked or conditional rewrite. Auto tries masked then conditional if compilation fails. Its JSON records failed attempts and the selected strategy; this is compatibility selection, not a speed prediction. `--no-coalesce`, `--no-factor-commit`, and `--no-snapshot-protection` support explicit ablations (the last may reproduce wrong stock snapshots); `--strategy` forces a representation, and `--pool-demand-ratio` controls the masked gate. The tested irregular CUDA case falls back successfully; the tested winner compiles directly. Callers must validate numerical outputs for their own programs.

## NVIDIA GPU

Use a CUDA-enabled PyTorch environment with a compatible NVIDIA driver; the IREE runtime wheel exposes the CUDA driver. Match the architecture to the actual GPU (e.g. A100:sm_80, H100:sm_90). Rebuild the standalone CPU pass tool and regenerate source MLIR using commands above if not using the supplied IR.

```bash
source scripts/env.sh
python research2/gpu_suite.py --arch sm_80 --compile-only
python research2/gpu_suite.py --arch sm_80 --torch-cuda
```

The GPU runner validates every successful IREE executable against float64 CPU PyTorch before measuring it. It records a compatibility fallback (masked, conditional, fused, baseline); this order is not performance tuning. Timing excludes input upload and includes invocation plus every output download for both IREE and optional eager CUDA PyTorch. It is synchronized host latency, not kernel-only CUDA-event time. The runtime command fails if a case lacks a correct baseline or compatible implementation. Use Nsight Compute separately for actual device traffic; compilation does not supply those measurements.

The recorded main sm_80 offline run compiled43/44 forms; all4 runtime-winner and21 extension forms also compile. See GPU_RUN.md for the prepared-IR route. Masked forwarding of the irregular D24,N65 case fails GPU bufferization; conditional forwarding compiles for that case. This machine has no GPU, so no CUDA numerical or timing result is claimed.

## Prior work boundary

Read-through-write is standard compiler algebra; copy-on-write KV storage and gather fusion are established ideas. IREE#24374 already tracks generic gather fusion; IREE#21222 tracks paged attention. PagedAttention and ForkKV are related systems work, not contributions of this repository. The candidate contribution is the demonstrated functional-cache materialization gap, its tested compiler representations and profitability limits, plus an independently isolated vectorization correctness defect. A literature search without an exact match is not proof of novelty.

- https://github.com/iree-org/iree/issues/24374
- https://github.com/iree-org/iree/issues/21222
- https://arxiv.org/abs/2309.06180
- https://arxiv.org/abs/2604.06370

## Direct IREE source repair

`BUILD_JOBS=3 research2/build_patched_iree.sh` checks out the exact measured revision, applies both source patches and builds CPU-only compiler tools. The initial build is substantial (LLVM/MLIR included). Never run concurrent builders in one build directory. After completion, run `research2/test_source_patch.sh`; it checks leading-batch transfer maps, executes the native counterexample, and repeats the nine-case attention audit with the patched compiler. The unmodified wheel fails the new FileCheck regression as expected. See `STATUS.md` for the distinction between successful source-object compilation and completed source-compiler execution.

The current upstream HEAD queried for this investigation is9ea0d950d7354009c247159a607e85f92d912d1d. Both implicated source files are byte-identical to the tested release; this is a source audit, not an executed HEAD build. IREE3.12.0 also reproduces the native snapshot failure. No first-affected-version claim is made.
