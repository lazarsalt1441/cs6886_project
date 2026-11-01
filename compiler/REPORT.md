> Research continuation correction: stronger existing IREE flags eliminate the original broadcast example. Its default-baseline speedup below does not establish novelty. The current study, independent backend counterexample, correct baselines and limitations are in [research/README.md](research/README.md) and [the manuscript](research/paper/manuscript.pdf).

# GQA KV head reuse: implementation and experimental report

Date: 2026-10-22 (Asia/Kolkata). This is a completed CPU research prototype,
with offline CUDA compilation. GPU runtime validation remains outstanding.

## Result and scope

A real C++ MLIR pass was implemented, compiled, executed and integrated into
PyTorch -> Turbine -> standard Linalg -> C++ pass -> IREE -> CPU executable.
It removes redundant KV head materialization in **decomposed attention** written
as repeat_interleave -> matmul -> softmax -> matmul. The original shape improves
from 18.47 to 9.94 microseconds
(1.86x), with runtime peak allocated buffers falling from
333824 to 75776 bytes.

The proposal's broad novelty claim needs qualification: current IREE already
folds head broadcasts for native scaled_dot_product_attention(enable_gqa=True).
The native exported path was compiled, executed and inspected: a single attention
dispatch reads compact 2x128x32 K/V inputs. The opportunity here is the decomposed
contraction path, whose separate broadcast dispatch survives ordinary lowering.
This project is not a new attention algorithm, hardware-managed cache, or a
claim to eliminate a persistent autoregressive cache.

## Environment and implementation route

- CPU: AMD EPYC 9V74, 9 visible vCPUs, roughly 9.7 GiB RAM. No GPU exposed.
- Python 3.12.14, PyTorch 2.14.1+cpu, Turbine 3.9.0, NumPy 2.3.5.
- IREE wheel 3.13.0; compiler reports 3.13.0rc20261016,
  commit 89ece87b89852100c0a4ac3d515e14842d090967; LLVM 24.0.0git.
- C++ pass built using GCC13 and Ubuntu MLIR/LLVM 19.1.1 extracted into a local
  SDK. SHA256 hashes and a rootless downloader are provided.
- An IREE source checkout was also inspected at
  f1eb9ce728e5cf9553307865532807673142e775. It is not the compiler used for timings.
- Full IREE/LLVM source build was avoided in favor of the authorized standalone
  fallback: wheels do not provide a matching C++ development SDK. The pass is a
  genuine registered MLIR pass, not a Python rewrite or model source rewrite.
  Standard textual Linalg is the bridge between MLIR19 and IREE's LLVM24 build.
- Initial old/new IREE wheel collisions caused a symbol import error. Removing
  old compiler bindings and reinstalling only base packages resolved it. Use
  the clean venv bootstrap; do not mix iree-compiler and iree-base-compiler.
- System apt encountered sandbox UID failures and package-manager stalls.
  SDK extraction avoids system changes and is the reproducible supported route.

## Transformation and correctness argument

`src/GQAHeadReuse.cpp` registers `--gqa-head-reuse` and the experimental
`--gqa-compact-head-reuse`. The main matcher accepts a tensor batch_matmul whose
RHS comes from a validated pure-copy broadcast followed by exact supported
collapse/transpose chains. It verifies static positive shapes, rank, head order,
reassociation, indexing maps, element types, producer body, and scalar
multiply/accumulate semantics. Dynamic shapes, fp16, noncopy producers and
unsupported axis arrangements are rejected. Integer shape products are checked
for overflow. Shared producer users are preserved.

Let h be a query head, Hkv the number of KV heads, and R=Hq/Hkv. Splitting the
flattened batch/query-head axis as (batch*Hkv, R) implements
kv_head=floor(h/R) with within_group=h mod R. The original repeated tensor has
identical values for all R entries. Thus QK and PV can read the compact K/V head
for the same contraction indices. The pass preserves destination initialization
and reduction operand order. It uses projected-permutation indexing maps rather
than introducing floordiv into Linalg maps, preserving structured tiling support.
The softmax, scaling, additive masks and user outputs remain in the graph.

This removes temporary expanded K/V copies. It does not append to, resize,
paginate or manage the lifetime of a persistent KV cache. It does not guarantee
that hardware loads are shared across heads; that depends on tiling and cache.

## Validation

- 9 MLIR structural tests pass, including actual FileCheck, verifier rejection,
  wrong-axis broadcasts, noncopy producers, shared users, dynamic/fp16 rejection,
  and idempotence.
- 15 end-to-end shapes: ratios 1,2,3,4,8; batch 1,2; M=1..64; N=32..8192;
  D=16..128 including non-power-of-two N and D. Ratio 1 is an unchanged control.
- Baseline and optimized outputs are compared with PyTorch explicit attention,
  and an independent grouped einsum reference checks the PyTorch semantics.
- Maximum absolute error across those runs: 4.17232513e-07. Relative L2 errors
  are retained per shape in results/benchmark.json. Tolerance: atol=2e-5, rtol=2e-4.
- Four further masked-attention cases (causal and padding bias) passed against
  PyTorch before and after rewriting. An additional contraction test checks a
  nonzero destination seed. See results/extra-correctness.json. These errors are
  recorded separately because the unnormalized contraction has a larger scale.
- The compact-loop extension passes numerical checks in six representative
  shapes, as do aggressive-fusion and combined variants (18 executed comparisons).
- The unsqueeze/expand/reshape frontend spelling was separately exported; both
  contractions rewrite and the executed result agrees with PyTorch within 1.5e-7.
- CUDA sm_80 baseline and optimized VMFB/PTX compilation both succeed. There is
  no numerical GPU execution result or GPU performance result.

## Benchmark methodology

Primary timings are results/pinned_benchmark.json: one pinned CPU (CPU 0),
local-sync IREE execution, single-thread PyTorch, 10 warmups, 21 trials, randomized
variant order. Each trial batches calls, with batch size adapted from original
latency (1..100). Inputs are preuploaded and reused; timing includes Python IREE
invocation overhead and execution, excludes input upload, output copying and
compilation. PyTorch uses inference_mode and resident inputs. Small shapes can
be dominated by wrapper overhead. This is not a kernel-only benchmark.

The initial unpinned suite, all raw samples, per-model compiler wall times,
pass-process wall times, and errors are retained. CPU frequency, host contention
and cache behavior are not controlled beyond CPU affinity. The large decode
case measured 52.4x unpinned but 16.4x in the pinned confirmation;
only the latter is used as the primary result. Do not generalize it to other
hardware, a complete LLM or an optimized fused baseline.

Fresh-process memory experiments run each variant three times. IREE classifies
CPU HAL buffers under device_bytes_*; host_bytes_* are zero and must not be
interpreted as zero memory use. Separate processes prevent cumulative peaks
from cached shared devices contaminating comparisons. Peak process RSS is also
recorded, but includes Python, NumPy, libraries, input arrays and runtime code;
RSS is not an allocator-only measure. HAL alloca sizes are compiler plans and
are recorded separately from measured runtime buffers.

| Shape | Torch eager us | IREE baseline us | IREE reuse us | Speedup | First-call allocated bytes baseline -> reuse |
|---|---:|---:|---:|---:|---:|
| B1 H4/4 M1 N32 D16 | 21.20 | 7.11 | 6.80 | 1.046x | 1280 -> 1280 |
| B1 H4/2 M1 N64 D16 | 21.08 | 8.11 | 7.21 | 1.125x | 34048 -> 2304 |
| B1 H8/2 M1 N128 D32 | 32.71 | 18.47 | 9.94 | 1.857x | 267264 -> 9216 |
| B2 H8/2 M1 N128 D32 | 49.08 | 38.63 | 12.62 | 3.060x | 534528 -> 18432 |
| B1 H8/1 M1 N256 D64 | 80.88 | 35.10 | 15.74 | 2.230x | 534528 -> 18432 |
| B1 H16/2 M1 N512 D64 | 248.98 | 216.11 | 44.59 | 4.846x | 4231168 -> 69632 |
| B1 H16/4 M1 N1024 D64 | 486.98 | 525.50 | 105.69 | 4.972x | 8458240 -> 135168 |
| B1 H8/2 M16 N128 D32 | 80.49 | 57.03 | 49.85 | 1.144x | 344064 -> 147456 |
| B2 H8/2 M16 N256 D64 | 435.77 | 381.23 | 311.69 | 1.223x | 2424832 -> 589824 |
| B1 H16/4 M32 N512 D64 | 1490.26 | 1447.38 | 1278.68 | 1.132x | 5373952 -> 2228224 |
| B1 H8/2 M1 N4096 D64 | 1277.96 | 1328.61 | 301.26 | 4.410x | 16910336 -> 264192 |
| B1 H16/2 M1 N8192 D128 | 55816.50 | 52748.18 | 3206.85 | 16.449x | 134750208 -> 1056768 |
| B1 H6/2 M3 N37 D24 | 26.83 | 16.62 | 27.55 | 0.603x | 47040 -> 7104 |
| B2 H12/4 M7 N65 D40 | 99.79 | 158.08 | 119.13 | 1.327x | 569792 -> 114304 |
| B1 H8/2 M64 N256 D64 | 630.52 | 606.21 | 584.56 | 1.037x | 1703936 -> 1179648 |

![Pinned latency and speedup](results/performance.png)

![Measured runtime buffer peaks](results/memory.png)

The ratio-one control's 1.046x difference is measurement noise: its
contractions are unchanged. The small irregular B1 H6/2 M3 N37 D24 case regresses
(0.603x) despite lower memory. Fewer bytes do not guarantee
faster vectorized contractions: this grouped layout changes loop structure and
backend scheduling. The measured regression is established; a specific
microarchitectural cause is not proven without counters and further tuning.
No shape-based profitability heuristic is silently fitted to this small suite.

## Allocation, traffic and assembly: distinct evidence

| Claim | Evidence | Limit |
|---|---|---|
| Logical expansion removed | Both broadcast/collapse chains disappear from optimized Linalg for private producers | Shared external users retain their producer |
| Expanded physical storage removed | Separate writable dispatch binding disappears; HAL workspace 266240 -> 8192 bytes; measured first-call bytes 267264 -> 9216 | HAL queue alloca site count remains 2: large arena shrinks, allocation count need not drop |
| Copy traffic removed at generated-code level | Original assembly has vector load/store copy loops with four replica iterations; optimized code has no broadcast kernel | This is issued copy work, not a measured DRAM-byte or cache-miss count |
| Runtime improves for a shape | Timed execution with raw observations | One demonstrated regression; platform and graph specific |

For the original shape, a broadcast entry is called twice (K and V). It writes
2*B*Hq*N*D*4 = 262144 bytes of expanded data per invocation of the model, before
considering other layout work and liveness reuse. These explicit copy stores
are absent after rewriting. This formula does not claim equivalent DRAM savings.
The compiled command buffer has 5 -> 3 dispatch invocations;
CPU/CUDA executable entry counts are 4 -> 3.

Inspect ir/cpu-baseline/*.s and ir/cpu-optimized/*.s, their LLVM .ll files,
and ir/cuda-baseline/*.ptx versus ir/cuda-optimized/*.ptx.
results/codegen-summary.json records exact entry names. The baseline copy loop
contains vmovaps loads/stores and cmpq $4 loop control; it is absent from the
optimized binary. PTX likewise loses the elementwise_broadcast entry.
The ordinary Linux PERF_COUNT_HW_CACHE_MISSES probe returns EACCES. Hardware
traffic counters and GPU profiling therefore remain unavailable.

## Extensions and native-path comparison

The compact pass merges adjacent within-group/query-position loops. It is a
working optional C++ transformation, not a promoted universal improvement.
IREE aggressive fusion and the combination were also compiled and validated.
The following comparison uses the same pinned timing run (microseconds).

| Shape | Reuse | Compact loops | Aggressive fusion | Compact + fusion |
|---|---:|---:|---:|---:|
| b1_q8_kv2_m1_n128_d32 | 9.94 | 9.86 | 10.32 | 9.51 |
| b1_q16_kv2_m1_n512_d64 | 44.59 | 42.92 | 44.12 | 43.84 |
| b1_q16_kv4_m32_n512_d64 | 1278.68 | 1248.88 | 1283.27 | 1280.63 |
| b1_q16_kv2_m1_n8192_d128 | 3206.85 | 3158.99 | 3212.64 | 3110.73 |
| b1_q6_kv2_m3_n37_d24 | 27.55 | 27.69 | 27.68 | 27.00 |
| b1_q8_kv2_m64_n256_d64 | 584.56 | 583.92 | 569.05 | 586.14 |

The gains vary and the irregular-shape regression persists. The default stays
the original grouped pass. Initial extension timings are retained separately
and should not be mixed with the pinned confirmation.

A separate pinned, interleaved original-shape comparison also measured:

| Path | Median us |
|---|---:|
| PyTorch explicit repeat attention | 36.96 |
| PyTorch native SDPA GQA | 16.13 |
| IREE explicit baseline | 20.65 |
| IREE with C++ reuse pass | 10.98 |
| IREE native SDPA GQA | 31.18 |

Native IREE avoids expansion and fuses attention, but its CPU kernel is slower
for this particular small shape. That does not establish superiority over
native attention generally. The separate rerun's absolute values differ slightly
from the main suite; use within-run comparisons. Upstream source contains
broadcast-attention fusion tests in compiler/src/iree/compiler/DispatchCreation/test/
elementwise_op_fusion.mlir. The native-path observation rules out claiming
that all IREE GQA necessarily materializes expanded buffers.

## Reproduction, limits and next research

Follow README.md for bootstrap, build, tests, full suite, memory, native path,
assembly and offline GPU compilation. STATUS.md and RESUME.md provide exact
commands and milestone history. The repository includes all exported and
transformed fixtures, raw measurements, plots, compiler logs, local Git history,
and a source-only reproduction path. Large SDK/build/vendor directories are
excluded from the deliverable; the downloader reconstructs the pinned SDK.

Supported now: static positive f32 tensor shapes, exact repeat-lowered RHS
batch_matmul patterns, transposed keys or untransposed values, multiple head
ratios, and prefill/decode examples. Unsupported layouts are left unchanged.
Missing: dynamic-shape specialization, fp16/bf16 support, broad pattern coverage,
a built-in upstream IREE plugin, multi-thread/multi-GPU benchmarks, persistent
cache mutation, paged attention, dropout semantics and GPU execution/profiling.
The standalone pass deliberately runs before IREE dispatch creation rather
than modifying already-bufferized IR.

Highest-value next steps are to run the provided GPU commands on real hardware,
profile the irregular CPU regression, add wider precision and shape support
with careful scalar-cast semantics, and port the proven pattern to IREE's own
LLVM revision for an upstreamable pass or preprocessing plugin. Compare with
native SDPA and fused attention baselines before making research novelty claims.

## Primary documentation

- IREE build and extension architecture: https://iree.dev/building-from-source/getting-started/
  and https://iree.dev/reference/extensions/
- MLIR Linalg structured operations and indexing maps: https://mlir.llvm.org/docs/Dialects/Linalg/
- IREE source revision inspected: https://github.com/iree-org/iree/tree/f1eb9ce728e5cf9553307865532807673142e775
- User proposal: docs/proposal.pdf (also extracted in docs/proposal.txt).
