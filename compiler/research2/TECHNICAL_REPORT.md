# Observation-aware KV-cache lowering in IREE

Seven C++ MLIR passes now execute in a complete PyTorch→IREE pipeline. The investigation identified two independently reproduced correctness defects and a family of cache-materialization gaps. The fixes are implemented and tested; being the first public report of these issues is not established.

## What changed from the original proposal

The original repeated-head GQA pattern is already optimized by stronger upstream IREE flags. Its default-baseline speedup is not a new research contribution. The retained original project documents that negative result. This investigation instead studies which functional cache versions must exist in memory when attention observes only a subset of their rows, and what happens when old cache snapshots escape through function outputs.

The measured compiler is IREE3.13.0rc20261016, commit `89ece87b89852100c0a4ac3d515e14842d090967`. The installed package is3.13.0. CPU experiments use an AMD EPYC9V74, one pinned CPU, PyTorch2.14.1+cpu and local-sync. No GPU execution was available. Exact environment details are in `environment.json`; later recovery commands and the separate3.12 comparison are recorded in logs.

## Optimization: demand, update chains and committed versions

For a cache update U at position p, a read of the resulting cache at i reads U[i−p] inside the update interval and C[i] elsewhere. This is established read-through-write algebra. The compiler opportunity is deciding when to represent those reads directly instead of materializing entire cache versions, and choosing a representation that downstream IREE can fuse.

The scalar-read pass emits a lazy conditional, preserving safe addresses. The alternative pass gathers the base and applies a masked gather from the small update. It translates IDs with integer arithmetic and uses safe inactive indices. Both retain the original attention calculation and head-sharing semantics. No floating-point reassociation is introduced by these passes. Attention outputs are checked against float64 PyTorch; exact data movement is checked separately at the bit level.

Forked cache versions exposed the initial opportunity: strong IREE optimization, aggressive fusion, and a corrected explicit-gather baseline still copy full pools. For the large-capacity case, CPU and CUDA HAL schedules contain67,110,912 bytes of copies before forwarding and zero afterwards. CPU invocation allocations fall from33,622,016 bytes to3,200 bytes in the strong-baseline comparison. These are allocation and scheduled-copy measurements, not hardware DRAM counters.

Two extensions were required for less simplified frontend expressions:

- **Per-token update chains.** With4–16 adjacent slice_scatter updates, the earlier forwarding passes stop matching. Coalescing adjacent unobserved writes into a flattened small concatenation restores the required representation. The pass rejects gaps, overlaps, strided writes, observed intermediate versions, large updates, and already single-use pools. Coalescing alone adds small temporary costs; its benefit comes from composition with forwarding.
- **Runtime winner commits.** A scalar attention-derived decision selects matching cache versions. Factoring that selection into the small updates allows the ephemeral versions to disappear while one actual winning cache is committed. Different bases/regions and elementwise conditions are rejected. Returned cache values and decisions match PyTorch bit for bit.

Escaping full cache versions remain materialized. The all-versions-returned control has no claimed copy elimination. A stricter default pool/read cost ratio skips a measured dense-read regression; this revised gate was developed after observing that result and is not independently validated by the same holdout data. Hkv1 also favors the conditional representation in the tested case. The driver uses compatibility fallback, not a universal performance predictor.

## Correctness defect1: gather index projection

IREE accepts an optional unit index-component axis for depth1 gathers, such as indices shaped N×1. Its vectorization model constructs an N-lane index vector but uses a default minor-identity transfer map. That reads the unit component axis instead of the leading batch axis. Poison padding can become a live gather address.

A native IREE counterexample with zero Q/K and arithmetic-progression V must produce head values254 and255. The pinned compiler produces14 and15; compiled index normalization produces254 and255 exactly. In an independent9-case attention audit, original generic gathers pass9/9, raw explicit-component gathers pass3/9 and include one crashing child process, and normalized gathers pass9/9. Incorrect kernels are excluded from every speedup baseline.

`GatherIndexProjection.cpp` is the compiled repair used end to end. `patches/iree-gather-index-projection.patch` fixes the leading-batch map directly in IREE. That modified source object compiled against the pinned LLVM24 source before the workspace rollback; a complete source-compiler build did not finish. The proposed source regression checks one and two batch axes.

## Correctness defect2: original-cache snapshot corruption

The self-contained native program returns an updated tensor and its original input tensor. With base=[0…63], updates101/102 at7/8 and201/202 at9/10, the second output must retain7/8/9/10. Stock IREE returns101/102/201/202 for both outputs. This reproduces with default flags, O0, strong flags, and the separately installed3.12.0 compiler. The first affected revision is unknown.

A real single-branch GQA export returning old K/V and updated K/V reproduces the same corruption on three seeds. A two-branch control happens to preserve the snapshots. This distinction matters: the failure is not asserted for every cache graph.

The pass trace exposes the mechanism. ABI export(import) folding removes the original tensor observation from resource SSA. A stream.async.cast shares storage with a non-consuming external import, but the local copy-on-write check treats the single-use cast result as independently owned. The following stream.async.update can therefore overwrite storage still exposed by the original-cache output. Inserting the missing clone at that exact checkpoint repairs the result when the remaining IREE pipeline is replayed.

Two C++ repairs are compiled and exercised:

1. An early guard preserves direct returned cache arguments through an optimization barrier until copy-on-write analysis. It applies only when the same argument feeds a functional insert_slice update.
2. A phase-local Stream pass inserts the required clone for async.update targets reached through casts from non-consuming tensor imports. The pass preserves affinity properties and skips consuming imports. `repair_snapshot_cow.py` invokes this compiled pass and resumes the real IREE pipeline; Python handles IR transport, not the rewrite.

Across40 rank/dtype/order configurations, stock snapshots fail40/40. Both repairs pass40/40 each, preserving updated outputs and original snapshots bit for bit, including NaN payloads and signed zeros. The Stream repair also fixes the actual GQA export on three seeds. Its FileCheck regression passes with the compiled C++ repair and fails with the stock pass. The typed in-tree COW patch is a proposal and has not been built into a full source compiler.

Preserving snapshots can require a full copy when updated caches are returned. When only attention and old caches are returned, forwarding can eliminate the ephemeral updated cache entirely. That corrected snapshot-attention case improves2.58× against the fastest correct baseline tested, with invocation allocation2,098,176→2,176 bytes. The incorrect stock executable is a diagnostic, never a performance baseline.

## Evidence and measured limits

`RESULTS.md` contains the original14-group study. Synthetic full transformer blocks improve2.39× and7.90×; runtime-winner attention improves3.60×. They include projections, normalization, RoPE and MLP where described, but are not pretrained generation benchmarks. `EXTENSIONS.md` contains a separate isolated5-group run: combined coalescing/forwarding improves2.17–2.34× for4–16-token update chains, and corrected snapshot attention improves2.58×. Regressions and no-op controls remain visible.

Runtime timing is separate from compilation, uses resident inputs, excludes output downloads on CPU, randomizes implementation order, and freezes a calibration choice before holdout inputs. Raw observations and bootstrap intervals are retained. Intervals describe invocation variability on this host/run, not uncertainty across devices or workloads. Compile-time diagnostic fields are not an isolated compilation-speed study.

Validation includes100 structural unittest cases, two actual COW FileCheck cases,556 passing bitwise executions across the dataflow studies,40 explicitly recorded stock snapshot failures, and the documented attention/model checks. Floating-point attention is not claimed bitwise identical. The largest retained full-block absolute error is1.006×10⁻⁴ under the stated atol/rtol checks.

CUDA offline compilation passes43/44 main forms,4/4 winner forms, and21/21 extension forms. The irregular masked form fails bufferization; conditional fallback compiles. Prepared GPU IR avoids rebuilding the standalone SDK on a GPU host. These results establish compilation only: no CUDA numerical result, latency, VRAM peak or memory-traffic counter was measured.

## Integration, reproducibility and novelty boundary

The working compiler implementation uses a standalone MLIR19 C++ tool and a verified generic-IR bridge to IREE's LLVM24 build. C++ private ABIs are not mixed. Phase-local IR uses local-scope printing to avoid losing aliases nested in opaque IREE attributes. The pinned IREE compiler reparses and verifies transformed IR before code generation.

Both implicated upstream source files remain byte-identical at the separately fetched HEAD `9ea0d950d7354009c247159a607e85f92d912d1d`; see `results/upstream-source-audit.json`. That is a source audit, not an executed HEAD build or proof that no compensating change exists elsewhere. Search did not establish an exact existing report; absence of a search hit is not proof of global novelty. Known generic-gather fusion, paged attention, copy-on-write systems and the read-through-write identity are prior art.

The concrete candidate research contribution is observation-sensitive lowering of functional KV state: recover demand through update chains, preserve externally observed old snapshots, and materialize only required committed versions. The demonstrated current-release correctness defects and measured compiler representations support that investigation. Stronger publication claims still need independent novelty review, real serving-model integration, GPU execution, hardware counters and broader compiler comparisons.

Reproduction starts in `README.md` and `GPU_RUN.md`. Source, failed diagnostics, correct baselines, IR, LLVM/disassembly, raw timing data, plots and local Git checkpoints are retained. The source-build interruption and rollback are documented in `RESUME.md`; working standalone fixes are preserved independently.
