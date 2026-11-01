# Exhaustive-request audit: verified coverage, missed cases and limits

Independent research review, 1 November 2026. “Missing” below always means a **specified graph family in the tested binaries/settings**. This audit does not prove absence across every frontend, target, dtype, tuning specification or possible pass pipeline. It does not call a lack of documentation proof of absence.

## Executive judgment

The most promising central contribution is a compiler analysis of **functional KV versions with small patches and multiple competing consumers**, preserving observation/ownership semantics while deciding what to materialize, forward and commit. This is stronger than presenting seven unrelated rewrites as novel.

The project has credible new-to-IREE missed cases. It has not established a globally new optimization principle or end-to-end LLM GPU acceleration. GQA elimination is already handled in the demonstrated family. Projection placement has substantial existing mathematical, systems and compiler precedent. Native fallback is primarily a correctness/pipeline repair.

| Area | Classification | Confidence | Upstream support / boundary | Source links | Independent evidence |
|---|---|---|---|---|---|
| 1. GQA expansion | A (tested family) | High | Generic contraction generalization + reshape motion + fusion; native/broadcast pathways also exist. | [S1], [S2], [S3], [S7] | Grouped fixture: 2 dispatches default/O3 -> 1 with both flags; compact K consumed directly. |
| 2. KV read forwarding | B overall; C for conditional fork-gather subcase | High for static FP32 fixtures | MLIR exact-slice forwarding and Stream update elision already exist. Conditional reads through competing cache versions are missed in tested configurations. | [M1], [S4], [S5] | Two pool sizes; actual C++ pass emits two conditional reads; all checks pass on both binaries. |
| 3. Winner-cache factoring | C for equal-region/concat fixtures | High for restricted patterns | No equivalent selection-before-update transformation triggered. Generic fusion can reduce dispatches without factoring the commit. | [S4], [S6], [X2] | Actual C++ insert and concat patterns tested. Explicit old-cache contract passes. Full-model A=0, B=60 matches. |
| 4. Latent projection placement | C in tested IREE; established prior art | High for explicit FP32 graphs | Default/O3, strong, aggressive and reassociation profiles retain per-history projections. General projection chains differ from quantized reassociation. | [S6], [X1], [P1], [P3] | Two complete softmax shapes plus actual rank-3 C++ score rewrite; numerical checks pass. |
| 5. Partial-RoPE projection | C at explicit split/concat boundary; D for general RoPE recognition | Medium | Content projection is not absorbed in tested split graphs; arbitrary rotary math/layout recognition was not established. | [S6], [P2] | Actual concat-based C++ score pattern plus complete attention with already-rotated operands pass. |
| 6. Update-chain forwarding/coalescing | B; C for eligible adjacent-chain subcase | High for narrow cases | Exact overwrite folds, slice composition and dead update elimination exist. Adjacent small-patch coalescing and conditional overlapping-chain reads remain separate. | [M2], [S4], [S5] | Same-slice control folds; disjoint read retains copies; actual eligible coalescing fires but gives no reliable speedup. |
| 7. Native attention fallback | B: manual upstream route exists; automatic safeguard fails | High on tested CPU; D for current T4 | Existing early online-attention conversion/decomposition avoids failure; ordinary scheduling still fails wide Q/K. | [S8], [S9], [S10], [S11], [S12], [S13] | 64/128 QK pass; 256/512 fail 20/20 per affected ordinary profile. Early decomposition passes all six shapes. |


## 1. Versions, provenance and what actually ran

| Component | Exact evidence |
|---|---|
| Supplied experiment | IREE package 3.13.0; embedded compiler 3.13.0rc20261016, `89ece87b89852100c0a4ac3d515e14842d090967`; LLVM 24.0.0git |
| Independent pinned installation | Same embedded version and full commit; `results/pinned/version.txt` |
| Newer binary tested | 3.14.0rc20261031, `94cb5993a90523669770a2a7a998d0a9a09147b5`; GitHub release wheels; `results/latest/version.txt` |
| Upstream main source inspected | `29114a017c104882cab055a49758e8f7055ea0d5`, committed 30 October 2026 at 11:48 UTC |
| LLVM/MLIR in experiment source | `32d702efa29d0fdd02cff7b79dea8fdbbcf059d4` |
| LLVM/MLIR in newer source | `f1ee8b38f40fd1f0e9596197b8e690eb9b57ec0c` |
| Independent CPU | Intel Xeon Platinum 8573C, x86-64; host targeting; local-sync; one CPU affinity selected during measurements |
| Original project C++ tool | Rebuilt with the package's hash-pinned MLIR/LLVM 19.1.1 SDK and GCC; textual IR bridge, not IREE-private C++ ABI mixing |
| GPU in this execution environment | No NVIDIA device/`nvidia-smi`; no new GPU execution performed |

The newer wheel is **not the exact inspected main HEAD**. Comparing the two source trees shows one compiler change: main fixes retention of nested Flow executable canonicalization/CSE ([S16]); other differences are test configuration files. That fix may affect generated IR, so direct main-binary execution remains an explicit boundary. The report never labels the nightly as a source-built main compiler. Commands to close this boundary are in README.

The attached consolidated PDF identified the experimental commit. I also retrieved the previously saved consolidated result archive and the current real-model test package. The latter supplies the actual C++ sources and prepared full-model IR. Historical T4 measurements in those files are **reported prior evidence**, not rerun numbers. The recovered real-model package is newer supporting material; its existing pretrained numerical results were not independently rerun here because its checkpoint/parameter archive was not packaged for this audit.

The original pass was actually rebuilt and executed here: two gather patterns forward; one insert-based commit factors; eligible adjacent updates coalesce; key projection and concat-split rotary patterns rewrite. The real-model ordinary decode graph remains byte-identical after the requested pass pipeline, with zero markers. The two-candidate custom lookahead graph produces 60 concat-commit markers. These are pass/applicability results, not newly executed pretrained inference.

## 2. Experimental matrix and evidence boundaries

The final main suite contains 42 MLIR programs and 322 distinct program/configuration records across two compiler versions. Each was separately compiled to global optimization, Flow, Stream and final VMFB: 1288 successful phase invocations. The main suite performed 3700 numerical executions: 3340 passed and 360 failed. Each ordinary failed native variant fails 20/20 seeds; the failures are not intermittent or hidden. Original concat structural tests pass 8/8; 8 bit-pattern scenarios with two compiled variants per compiler pass separately, including NaNs, infinities and signed zero. Explicit snapshot tests add 10 seeds for each of three variants on each compiler.

All main numerical checks re-upload independent fresh inputs. Ordinary arithmetic compares with NumPy float64 reference computations using atol=5e-5, rtol=2e-3. Data-movement checks compare exact finite values; the dedicated concat tests compare uint32 bit patterns. Scalar/dynamic indices stay within MLIR's original insertion/extraction domain. Unsupported out-of-bounds behavior is not assigned new semantics.

### Profiles actually tested

| Profile | Additional flags |
|---|---|
| default | none |
| O3 | `--iree-opt-level=O3` |
| strong | O3 + `--iree-opt-generalize-matmul` + `--iree-dispatch-creation-enable-aggressive-reshape-movement` |
| generalize / reshape | each individual shape flag, for GQA ablation |
| O0strong | both shape flags with `--iree-opt-level=O0`, for GQA |
| aggressive | strong + aggressive fusion, multi-use fusion, horizontal contraction fusion, matmul-producer blocking disabled, multi-reduction elementwise fusion, experimental transfer-gather vectorization |
| max_reassociation | strong + quantized-matmul reassociation + CPU FP reduction reassociation + `--acc-reassoc` |
| early_decompose | O3 + upstream online-attention conversion and decomposition in preprocessing |

All exact spellings and commands are preserved in each `compile.json`, with the actual registered options in the two saved `*-help.txt` files. Aggressive and reassociation profiles were used on relevant subsets, not mechanically on every graph. The experimental quantized-matmul option was explicitly tried on projection fixtures; it does not implement the desired FP32 chain transformation. Target-specific ROCm/MFMA schedules are not claimed tested by this CPU audit. No finite flag sample is called a universal strongest possible IREE configuration.

The paired CPU study first uses 31 randomized calibration invocations to independently choose a stock profile and custom profile; those choices are frozen before 201 randomized holdout invocations. It reports medians, p10/p90 and paired bootstrap intervals. Both compiler versions run on this same host. Inputs remain resident; compilation and output downloads are excluded. The runtime is local-sync, so CPU calls complete synchronously. Bootstrap intervals describe samples from one session, not cross-session reproducibility. Tiny few-percent wins are not publication-grade evidence.

## 3. Source-level findings for each optimization

### 3.1 GQA K/V expansion elimination — A for the tested family

IREE need not contain a pass named “GQA elimination” to avoid repeat materialization. `GeneralizeLinalgNamedOpsPass` turns contractions into generics when enabled ([S1]); `BubbleUpExpandShapesPass` permits reshape motion across reductions behind the aggressive option ([S2]); dispatch preprocessing runs fusion, reshape motion and fusion again ([S3]). This permits consumers to read the original compact operand using grouped indexing. `ConvertBroadcastBatchMatmulToMatmul` also folds compatible broadcast weights ([S7]). The Torch flex-attention import support is related but is not itself proof that every K/V expansion is physically absent.

Independent score-side fixture: Hkv=2, head ratio=4, sequence=32, width=16. Default and O3 produce two dispatch calls. Either shape flag alone still gives two; both give one, including at O0. Inspection of the final dispatch shows compact K directly feeding the contraction with grouped dimensions. The remaining allocation is the returned score tensor, not a repeated K tensor. The MQA and ratio-one controls each use one dispatch already in these fixtures. These new tests exercise K/score; the supplied older ablation separately reported K+V/full-attention coverage. Do not silently merge the two test counts.

Both knobs default false in their option declarations ([S17], [S18]); O3 did not turn the decisive combination on in either binary. PR [#22707](https://github.com/iree-org/iree/pull/22707), merged at `34830974a1aa494a8073377da5c03f7714694274`, introduced the aggressive option. Issue [#22631](https://github.com/iree-org/iree/issues/22631) remains open with documented regressions preventing unconditional reshape motion. PR [#19962](https://github.com/iree-org/iree/pull/19962) adds attention/broadcast transpose fusion; PR [#24313](https://github.com/iree-org/iree/pull/24313) adds Torch flex-attention GQA importing. Import support, logical fusion and absence of physical repeat storage must still be checked separately.

No central novelty claim should be made for repeat elimination in this family. Multi-use/escaping expanded values, frontend differences, mask layouts and dynamic heads remain a coverage question, not a blanket no-materialization claim.

### 3.2 KV-cache read forwarding — B overall, C for conditional forked gathers

The basic law is established array/compiler semantics:

`read(update(C,P,s),i) = P[i-s]` when `s <= i < s+len(P)`, otherwise `C[i]`.

MLIR already folds an exact `extract_slice(insert_slice(...))` to the patch ([M1]). IREE Flow folders fold constant/full updates and related shape cases ([S5]). Stream `isSafeToElideUpdateOp` removes updates whose modified range cannot be observed; it explicitly analyzes tied results and full overwrite chains ([S4]). This is already implemented and enabled in Stream's ordinary pipeline ([S15]). PR [#22808](https://github.com/iree-org/iree/pull/22808) added update elision in January 2026; PR [#23208](https://github.com/iree-org/iree/pull/23208) tightened tied-operation correctness. Those precedents rule out claiming general read-through-write elimination is new.

The independently missed subcase is two competing updates of a shared static 2D cache, each used only by a small gather with runtime row indices. Both versions retain full-version copy/update schedules under default, O3, strong and aggressive profiles. The rebuilt `ForwardRead` pattern replaces each gather's captured tensor read with safe `scf.if` loads from the patch or original cache. Both 512x16 and 8192x16 pools pass all checks. The 8192 case requests 1,049,600 bytes/invocation stock versus 1,024 bytes actual C++ output in the strong profile. Both stock and transformed gather cases preserve the input cache in the independent post-call observation test.

The large gather result is an actual C++ transformation benchmark; it is still a synthetic cache/gather workload, not full attention or LLM throughput. The scalar single-use forwarding controls are deliberately different: they are slower and mutate/consume a cache in the stock route. The original pass's profitability guard rejects such lone single-use versions. They are not evidence that the implemented guarded pass chooses those regressions.

Legality requires conditional loads: computing both extracts and selecting a value can introduce an out-of-bounds patch load on the inactive path. Positive strides additionally require congruence and bounds checks. Full-version outputs or other observations cannot be discarded merely because one small read is forwarded.

### 3.3 Winner-cache factoring — C for specified equal-region and concat patterns

For a uniform predicate and identical base/region,

`select(c, update(C,A,r), update(C,B,r)) -> update(C, select(c,A,B), r)`.

The original `FactorCommit` validates matching base, offsets, sizes, strides, ranks and simple all-parallel selection regions. Its concat extension validates a common history and selects only the appended patch. I independently rebuilt those patterns; they fire on the minimal valid cases. Stock IREE retains full-cache selection in those inputs, including aggressive fusion. One dispatch alone does not imply the full updates were factored: Stream still contains the base/cache copies.

The initial single-output winner comparison exposes a contract issue: the transformed update route mutates the provided input buffer while stock preserves it. Those timings must not be used as evidence for an immutable-input workload without explicitly permitting donation or checking/preserving the old value. I added a returned-old-cache contract and ran both guarded variants. All updated outputs, old snapshots and input observations pass for ten seeds each, on both versions. The snapshot timings below are the appropriate immutable-contract evidence for this case. The unguarded two-output fixture also passes here; this does not reproduce or dismiss the older cast-hidden COW failure reported in your PDF.

The concat case is more cautionary. Both candidates still need their attention users in a real branch graph. Factoring the final commit does not erase all candidate caches. In the standalone large concat test, O3/strong transformed lowering can be slower, while a separately calibrated default transformed lowering can be faster. Compiler scheduling and ownership effects dominate a simple graph-size argument. Full-model lookahead has 60 logical matches, but no new GPU or full-model throughput result was produced here. Ordinary HF decode has zero matches.

This law is not globally new. XLA's `HandleSelect` already moves selection inside a dynamic-update-slice for a related base-vs-updated form ([X2]). The exact two-updated-branch case is narrower than that cited rule; no XLA executable coverage claim is made. Branch selection, update motion and persistent version management have substantial compiler/runtime precedent.

### 3.4 Latent projection placement — C in the tested IREE patterns, old optimization principle

For `L: NxR`, `Wk: RxD`, `Q: MxD`, the score identity is

`Q (L Wk)^T = (Q Wk^T) L^T`.

With probabilities `P`, the value identity is `P (L Wv) = (P L) Wv`. The softmax remains between score and value contractions. The original score scale must be preserved: replacing D by R in `1/sqrt(D)` changes the model.

Both complete-attention fixtures (N=32,R=8,D=16 and N=512,R=32,D=128) retain history-wide projection results in stock global/Flow/Stream IR under all tested relevant profiles. Aggressive fusion reduces some dispatch counts but does not change the contraction order. Independently compiled transformed forms remove those large key/value expansions and pass all tolerance checks. I also ran the actual original C++ rank-3 key-side pattern and validated its generated IR on both compilers. The complete rank-2 attention fixtures are reconstructed mathematical equivalents, not claimed to be automatically matched by the rank-3 original pattern.

The pass is not a strict IEEE-equivalence canonicalization. Matrix reassociation changes floating-point rounding, cancellation, underflow and possibly overflow. The project requires explicit `attention.allow_reassociation`; its optional range checks are caller contracts/analysis, not measured accuracy guarantees. A legal production pass needs an agreed numerical contract and mixed-precision validation. Counting FLOPs is also insufficient: latent rank may exceed per-head key width, and head sharing, prefill length, Tensor Core shapes, layouts and attention kernel support matter.

Projection absorption is explicit in DeepSeek-V2 ([P1]) and TensorRT-LLM ([P3]). XLA's `AssociativeReorderNestedDot` already implements cost-based nested-dot reassociation, called when its associative-reordering option is enabled ([X1]). This is concrete compiler prior art, not just the observation that multiplication is associative. The new-to-IREE opportunity is safe recognition, placement and backend profitability on relevant exported attention graphs.

### 3.5 Partial-RoPE projection — C at an explicit split boundary; D beyond it

The lawful content/position split is

`score = (Qc Wk^T) L^T + Qrope Krope^T`.

Compatible content projection can move; position-dependent rotary factors remain in the positional branch. Dense arbitrary projection and RoPE do not generally commute. DeepSeek-V2 explicitly describes decoupled RoPE to enable absorption ([P2]); this is not a newly discovered partial-RoPE algorithm.

The full-attention reproducer takes already-rotated Qrope/Krope as inputs and preserves their contribution and original scale. Separately, the actual C++ pattern recognizes concat(content-projection, rotary-key), splits the query at that boundary and emits the absorbed representation. Both validations pass. These tests establish the explicit split/concat case. They do **not** establish recognition of arbitrary exported sin/cos, interleaved rotary layouts, quantization, dynamic position handling, or a proof that an unknown rotary implementation is compatible. Those broader claims remain D. The independent tiny-shape performance is approximately neutral; historical larger T4 rotary measurements remain prior evidence only.

### 3.6 Update-chain forwarding/coalescing — B, with a narrow coalescing gap

MLIR's `foldInsertAfterInsertSlice` already drops a prior update to the exact same slice ([M2]). Flow composes slice-of-slice offsets ([S5]). Stream update elision handles fully overwritten regions and rejects unproven tied chains; the source explicitly notes missing stronger Presburger overlap reasoning ([S4]). Thus chain simplification is already partially implemented.

An exact-slice positive control disappears. A disjoint intervening-update read still has three Stream copy commands in this fixture; that is retained materialization, not an expensive compute dispatch count. The independently rebuilt `CoalesceUpdates` pass merges two adjacent small updates in a forked-cache fixture. It requires a single-use intermediate and rejects unrelated/observed intermediates; it does not justify arbitrary reordering of overlapping writes. The separate overlapping dynamic read control uses last-write-wins conditional forwarding and is a conceptual stress test, not a claim that the original adjacent-coalescing pattern supports overlaps.

The eligible adjacent case has essentially no independent CPU latency benefit and can slightly increase allocation requests/copy count by materializing the joined small patch. Your older T4 chain results were also near 1x. Do not promote “coalesces updates” into a performance or research headline. A more substantial opportunity is an integrated range/version analysis that avoids materializing the joined patch and cooperates with Stream COW and scheduling.

### 3.7 Native attention fallback — existing manual route, automatic correctness gap

The independent shape sweep is batch-head=1, M=1, N=32, FP32. Each configuration has ten random cases and ten last-feature-only cases:

| QK | V | Pinned default | Pinned O3 | Pinned strong | Pinned early | Latest default | Latest O3 | Latest strong | Latest early |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 64 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 |
| 128 | 128 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 |
| 256 | 128 | 0/20 | 0/20 | 0/20 | 20/20 | 0/20 | 0/20 | 0/20 | 20/20 |
| 512 | 128 | 0/20 | 0/20 | 0/20 | 20/20 | 0/20 | 0/20 | 0/20 | 20/20 |
| 128 | 512 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 |
| 512 | 512 | 0/20 | 0/20 | 0/20 | 20/20 | 0/20 | 0/20 | 0/20 | 20/20 |


The last-feature cases distinguish loss of Q/K feature contributions from value width. Failed native programs are excluded from timings. No general threshold claim is made for every width above 128 or for every target.

The CPU trace supplies more than a numerical symptom. After vector-reduction tiling, the compiler emits an outer loop stepping through the 256 feature coordinates in blocks of 16 and an inner loop through KV positions. An `online_attention` operation consumes each partial feature dot product while carrying max/sum/output state. Online softmax state can combine partitions of KV tokens; it cannot combine incomplete feature sums as independent attention logits. That partial-feature online schedule is the CPU correctness failure shown by the trace. The source threshold at K1=128 explains why the tested wider cases enter this route ([S8], [S12], [S13]). The corresponding T4 root cause was **not** independently established; similar symptoms do not prove an identical GPU defect.

IREE already has conversion to online attention and decomposition passes ([S10], [S11]). Scheduling them in preprocessing, before the problematic tiling, passes all six shapes on both CPU binaries:

```text
--iree-preprocessing-pass-pipeline=builtin.module(util.func(iree-linalg-ext-convert-attention-to-online-attention,iree-linalg-ext-decompose-attention))
```

Therefore “IREE has no native-attention fallback setting” is false. What is missing on the tested automatic route is a correct schedule or automatic choice of a safe alternative. The project's fallback/CPU guard is an engineering workaround, not a new attention-performance algorithm. Issue [#23323](https://github.com/iree-org/iree/issues/23323) documents broader attention-test coverage gaps, but is not proof that this exact failure was already reported. The saved [#25150](https://github.com/iree-org/iree/issues/25150) concerns Stream read/write reordering; it is related to snapshot correctness, not the same attention defect.

## 4. Performance results and allocation evidence

### 4.1 All independently measured paired CPU comparisons

Ratios are stock/custom, so less than 1 is a regression. “Selected” uses independently frozen calibration choices for each program. “Matched strong” uses the same strong flags for both. This separates transformation benefit from compiler-profile choice. The low-level single-output winner/scalar rows have the ownership limitation described above; the actual explicit-snapshot results are separate. `*_cpp` means actual C++ output. Full rank-2 `projection_*` rows use reconstructed optimized IR. Gather rows use actual C++ output as their custom candidate.

| Compiler | Fixture | Stock profile | Custom profile | Stock us | Custom us | Selected ratio | Within-run bootstrap 95% | Matched strong ratio |
|---|---|---|---|---|---|---|---|---|
| pinned | forward_64 | O3 | strong | 7.463 | 13.905 | 0.537x | 0.531-0.541 | 0.535x |
| pinned | forward_4096 | O3 | default | 7.335 | 13.865 | 0.529x | 0.525-0.534 | 0.523x |
| pinned | winner | strong | O3 | 13.349 | 8.528 | 1.565x | 1.553-1.580 | 1.573x |
| pinned | chain | O3 | strong | 7.693 | 14.223 | 0.541x | 0.537-0.545 | 0.540x |
| pinned | projection_32 | max_reassociation | strong | 9.446 | 9.247 | 1.022x | 1.013-1.030 | 1.027x |
| pinned | projection_512 | default | O3 | 97.084 | 17.158 | 5.658x | 5.476-5.794 | 6.075x |
| pinned | partial_rope | strong | max_reassociation | 10.405 | 10.144 | 1.026x | 1.020-1.034 | 1.028x |
| pinned | gather_forks_512 | aggressive | default | 12.507 | 8.256 | 1.515x | 1.493-1.538 | 1.571x |
| pinned | gather_forks_8192 | aggressive | O3 | 74.914 | 10.292 | 7.279x | 7.067-7.459 | 7.365x |
| pinned | winner_cpp | strong | O3 | 10.601 | 6.728 | 1.576x | 1.561-1.593 | 1.568x |
| pinned | projection_cpp | O3 | O3 | 8.513 | 8.367 | 1.017x | 1.008-1.026 | 1.019x |
| pinned | partial_rope_cpp | strong | strong | 9.276 | 9.146 | 1.014x | 1.006-1.022 | 1.014x |
| pinned | coalesce_cpp | strong | O3 | 12.621 | 12.672 | 0.996x | 0.990-1.001 | 1.000x |
| pinned | concat_cpp | default | default | 64.038 | 29.501 | 2.171x | 2.135-2.238 | 0.782x |
| latest | forward_64 | O3 | default | 7.070 | 13.525 | 0.523x | 0.517-0.529 | 0.521x |
| latest | forward_4096 | O3 | default | 7.198 | 13.657 | 0.527x | 0.521-0.533 | 0.525x |
| latest | winner | O3 | strong | 13.485 | 8.586 | 1.571x | 1.562-1.582 | 1.569x |
| latest | chain | default | O3 | 7.514 | 14.129 | 0.532x | 0.526-0.538 | 0.532x |
| latest | projection_32 | O3 | max_reassociation | 9.315 | 9.057 | 1.028x | 1.022-1.035 | 1.022x |
| latest | projection_512 | default | O3 | 100.004 | 17.732 | 5.640x | 5.417-5.850 | 5.857x |
| latest | partial_rope | O3 | aggressive | 10.269 | 10.048 | 1.022x | 1.014-1.031 | 1.019x |
| latest | gather_forks_512 | aggressive | default | 12.303 | 7.970 | 1.544x | 1.532-1.558 | 1.583x |
| latest | gather_forks_8192 | strong | strong | 72.586 | 10.269 | 7.068x | 6.938-7.238 | 7.068x |
| latest | winner_cpp | default | O3 | 10.836 | 6.778 | 1.599x | 1.584-1.615 | 1.593x |
| latest | projection_cpp | O3 | O3 | 8.310 | 8.224 | 1.010x | 1.001-1.017 | 1.020x |
| latest | partial_rope_cpp | default | strong | 9.234 | 9.001 | 1.026x | 1.019-1.035 | 1.008x |
| latest | coalesce_cpp | strong | default | 12.465 | 12.484 | 0.998x | 0.991-1.007 | 0.996x |
| latest | concat_cpp | default | default | 65.116 | 31.316 | 2.079x | 2.045-2.141 | 0.790x |


These measurements are CPU synthetic kernels/graphs, not T4 measurements or model tokens/s. Within-run confidence intervals do not account for thermal/VM noise or independent sessions. Values within a few percent of 1 should be treated as neutral. The large projection result is one favorable complete-attention shape, not a geometric-mean LLM result.

### 4.2 Default vs O3 vs the stronger relevant settings, latest binary

All cells are holdout median microseconds from the same paired study:

| Fixture | Default | O3 | Strong | Aggressive | Reassociation | Custom default | Custom strong |
|---|---|---|---|---|---|---|---|
| forward_64 | 7.048 | 7.070 | 7.054 | - | - | 13.525 | 13.532 |
| forward_4096 | 7.161 | 7.198 | 7.170 | - | - | 13.657 | 13.650 |
| winner | 13.459 | 13.485 | 13.474 | 13.479 | - | 8.567 | 8.586 |
| chain | 7.514 | 7.533 | 7.528 | - | - | 14.200 | 14.144 |
| projection_32 | 9.548 | 9.315 | 9.330 | 9.331 | 9.271 | 9.384 | 9.125 |
| projection_512 | 100.004 | 101.170 | 103.315 | 262.576 | 102.170 | 18.067 | 17.639 |
| partial_rope | 10.604 | 10.269 | 10.206 | 10.278 | 10.380 | 10.278 | 10.016 |
| gather_forks_512 | 12.568 | 12.535 | 12.597 | 12.303 | - | 7.970 | 7.956 |
| gather_forks_8192 | 73.723 | 74.002 | 72.586 | 72.146 | - | 10.033 | 10.269 |
| winner_cpp | 10.836 | 10.903 | 10.878 | - | - | 6.820 | 6.829 |
| projection_cpp | 8.638 | 8.310 | 8.398 | - | - | 8.351 | 8.232 |
| partial_rope_cpp | 9.234 | 9.443 | 9.070 | - | - | 9.286 | 9.001 |
| coalesce_cpp | 12.445 | 12.434 | 12.465 | - | - | 12.484 | 12.513 |
| concat_cpp | 65.116 | 95.355 | 94.048 | 94.182 | - | 31.316 | 118.983 |


O3 is not uniformly fastest. In particular, default transformed concat scheduling can substantially outperform the same transformed program under strong settings. Selecting profiles after looking at holdout results would be biased; the saved calibration data and chosen profiles make that choice reviewable.

### 4.3 Explicit snapshot and immutable-input contract

Each check validates the committed output, returned old cache, and original input buffer. Timings are randomized 201-observation synchronous invocations with identical output/ownership contracts:

| Compiler | Explicit-snapshot fixture | All three checks pass | Median us |
|---|---|---|---|
| pinned | stock_unguarded | 10/10 | 12.032 |
| pinned | stock_guarded | 10/10 | 12.061 |
| pinned | custom_guarded | 10/10 | 9.293 |
| latest | stock_unguarded | 10/10 | 11.784 |
| latest | stock_guarded | 10/10 | 11.981 |
| latest | custom_guarded | 10/10 | 9.153 |


The custom guarded comparison is the appropriate insertion-commit evidence when old values must remain available. The older project-reported cast-hidden snapshot failures and proposed COW repairs are separate correctness work. This simple independent unguarded two-output fixture passes; the older bug is neither reproduced nor disproven by it.

### 4.4 Allocation/copy distinctions, latest strong profile

Allocator values are cumulative runtime allocation requests divided by 101 timed invocations after warmup. They are not peak RSS, reserved VRAM or hardware DRAM bytes. Copy counts are static Stream command occurrences and exclude implicit memory accesses inside kernels.

| Latest strong fixture | Allocated bytes / invocation | Flow dispatch calls | Stream copy commands |
|---|---|---|---|
| gather_forks_512_stock | 66,560 | 2 | 4 |
| gather_forks_512_actual | 1,024 | 2 | 0 |
| gather_forks_8192_stock | 1,049,600 | 2 | 4 |
| gather_forks_8192_actual | 1,024 | 2 | 0 |
| winner_stock | 49,152 | 1 | 5 |
| winner_custom | 0 | 1 | 1 |
| projection_512_stock | 526,848 | 5 | 0 |
| projection_512_custom | 4,736 | 5 | 0 |
| coalesce_stock | 66,560 | 2 | 5 |
| coalesce_actual | 67,072 | 2 | 6 |
| concat_stock | 786,592 | 2 | 3 |
| concat_actual | 786,528 | 3 | 1 |


The gather result is a strong allocation reduction on a correct, input-preserving example. Concat illustrates the converse: fewer static copy commands and almost unchanged requested bytes can coexist with a strong-setting latency regression. Eliminating a high-level op, fusing a dispatch, eliminating a temporary, reducing allocation requests and reducing actual DRAM traffic are distinct observations. No DRAM-traffic reduction is claimed here.

## 5. Prior work: new to IREE is not new to compiler optimization

| System / work | Verified precedent | Consequence for this project |
|---|---|---|
| Upstream MLIR | Exact read-after-insert slice folds and repeated identical-slice overwrite elimination ([M1], [M2]) | Claim conditional/versioned remainder, not general forwarding |
| LLVM | MemorySSA supplies versioned memory dependence information used by optimization passes ([L1]); load forwarding and dead-store elimination are established | General read-over-write and unobservable-update arguments are prior art |
| XLA | Nested-dot reassociation with a FLOP heuristic ([X1]); related select-inside-update rule ([X2]); slice-of-update replacement ([X3]) | Projection placement and selection/update motion are already compiler techniques |
| TensorRT-LLM | Explicit MLA weight absorption and distinct prefill/decode tradeoffs ([P3]) | Must compare against already-absorbed kernels, not only an unabsorbed frontend |
| Apache TVM | PagedKVCache, fused-QKV attention and explicit MLA-cache append ([T1], [T2]) | Mutable/paged cache APIs can avoid constructing the functional versions the pass removes; compare those execution strategies |
| DeepSeek-V2 | Joint KV compression, absorption and decoupled RoPE ([P1], [P2]) | Neither absorption nor decoupled positional handling is new here |
| TASO; TENSAT | Verified graph substitutions/cost search and equality saturation ([P5], [P4]) | Automatic graph recognition and cost-based choice alone are broad established concepts; specify the new analysis/coverage |

TVM and TensorRT-LLM evidence here establishes runtime/operator/algorithm precedent. It does not prove that either automatically recognizes every arbitrary functional MLIR graph. XLA and LLVM were inspected/researched but not built and benchmarked. The review is a targeted prior-art audit, not a comprehensive literature survey establishing global novelty.

## 6. Ranked research opportunities

| Rank | Opportunity | Why promising | Main threat / evidence still needed |
|---|---|---|---|
| 1 | Demand-driven functional KV version analysis for branching inference | Correct input-preserving fork-gather gaps and allocation/latency gains; connects read forwarding, live snapshots and commit placement | Read-over-write is old; natural HF greedy decode has no matches. Show real beam/speculative/tree workloads and superiority to mutable/paged cache lowering |
| 2 | Automatic MLA content/rotary recognition with numerically controlled projection placement and backend selection | Real MLA relevance, explicit IREE missed patterns, large favorable shape and actual C++ recognition | DeepSeek/XLA/TensorRT prior art is strong. Need recognizer breadth, mixed precision, prefill/decode profitability and tuned absorbed-kernel baselines |
| 3 | Range- and version-aware Stream optimization across patches | Existing conservative overlap analysis and COW/scheduling interactions give a clear integration point | Simple adjacent merging is neutral. Need a materially stronger analysis and workloads where it reduces copies without introducing patch temporaries |

Native attention feature-reduction correctness should be fixed upstream regardless of paper direction. It is valuable compiler engineering and useful regression-test work; the current evidence supports a bug-fix contribution, not a central optimization paper. GQA expansion elimination should become a coverage/baseline test rather than the project's main contribution.

## 7. Next implementation plan and pipeline placement

1. **Fix the correctness floor first.** Extend CPU attention configuration/tiling legality around `KernelDispatch.cpp` and the reduction-stage tiling in `addCPULinalgExtTileAndVectorizePipeline` ([S8], [S12]). Complete QK feature reductions before online softmax updates; choose an existing early decomposition route when a legal fused schedule is unavailable. Keep resource bounds; do not simply untile arbitrarily large feature dimensions. Add tail-feature analytical E2E tests, wide QK / wide V controls, masks and dtype/layout sweeps. Investigate the GPU route separately.
2. **Add a tensor-level functional-version analysis before irreversible lowering.** Use tensor/linalg form after frontend normalization, while insert-slice/concat, source identity and consumer demand are visible, before `ConvertTensorToFlow` and dispatch formation. Represent a version as base plus ordered patches, with use/escape and snapshot constraints. Extend or add a GlobalOptimization pass and register it in `GlobalOptimization/Passes.cpp` before dispatch shape/fusion processing. Preserve unknown users conservatively.
3. **Integrate selective commits and forwarding.** Factor only uniform matching-region/append commits. Rewrite demanded reads with safe conditional patch membership and last-write-wins ordering. Do not remove versions still needed by candidate attention, exported results or snapshots. Model donation/borrowed imports explicitly; apply snapshot correctness tests before collecting performance data. Extend concat support because natural HF DynamicCache emits concat rather than insert-slice.
4. **Cooperate with Stream instead of duplicating it.** Improve `AsyncAccessRange` overlap reasoning and `ElideAsyncCopies` ([S4]) for proved disjoint/overwritten ranges; coordinate with COW and concurrency scheduling ([S14], [S15]). Keep tensor-level value rewrites separate from post-bufferization memory-dependence legality. Avoid treating a tied-result alias as independent memory. Add invariants/test cases connected to issue #25150.
5. **Add a separate opt-in MLA pass.** Recognize content/rotary boundaries and batched/generic contractions before data tiling/dispatch formation. Preserve scale, mask and softmax semantics and original output projections. Require a numerical-reassociation contract. Estimate both FLOPs and memory/kernel costs using heads, M/N/R/D, layout/dtype, target, and known native-attention legality. Use calibrated fallbacks or tuning when static estimates are uncertain. Do not hide a correctness workaround in the profitability selector.
6. **Validate the actual system contribution.** Export naturally occurring graphs from two distinct branching decoders, run fresh pretrained logits/tokens tests, measure resident-cache end-to-end decode latency and tokens/s against strong IREE, Torch eager/Inductor/SDPA and a paged/mutable-cache path. Use identical precision and input/output ownership. Profile dispatch/launch/copy costs and hardware memory counters. Repeat across independent sessions and at least two GPU architectures. Report coverage, rejection reasons, compilation overhead and regressions, not only selected wins.

### Paper gate: the evidence required for a defensible central claim

A defensible claim would be: *a compiler analysis and lowering strategy for demand-driven functional cache versions in branching LLM inference, with explicit observational legality and target-aware materialization decisions*. To support it, show a nontrivial analysis beyond composing familiar local identities; compare against the existing MLIR/Stream/XLA mechanisms and optimized cache runtimes; demonstrate that natural graphs need it; and establish reproducible model-level GPU benefits without broken baselines or changed ownership contracts.

Current status is promising prototype evidence, not that gate. Projection placement can be a secondary contribution or an alternative compiler-systems direction only if its automatic recognition/legality/profitability substantially exceeds the established absorption implementations. None of the seven transformations, by itself as currently evidenced, justifies a claim of a globally new attention algorithm or a publication-ready result.

## 8. Reproducers, failures and reproduction instructions

See README for exact installation, original C++ build and execution commands. `reproducers/cases.json` gives every shape and input contract. The following map identifies minimal evidence for every claimed gap:

| Gap / control | Minimal files |
|---|---|
| Conditional forked gather | `gather_forks_512_stock.mlir`, `gather_forks_512_actual.mlir`; larger 8192 variants |
| Matching insert commit | `winner_generic_stock.mlir`, `winner_actual.mlir`; explicit-snapshot variants under `results/*/snapshots/` |
| Concat commit | `concat_stock.mlir`, `concat_actual.mlir` |
| Projection placement | complete `projection_32_*`, `projection_512_*`; actual C++ `original_projection_*` |
| Explicit partial-RoPE split | complete `partial_rope_*`; actual concat-score `original_rope_*` |
| Adjacent coalescing | `coalesce_stock.mlir`, `coalesce_actual.mlir` |
| Existing folds / chain controls | `same_slice.mlir`, `overwrite.mlir`, `disjoint_chain.mlir`, conceptual overlapping `chain_*` |
| Native failure / existing fallback | `native_256_128.mlir`, `native_512_128.mlir`; `early_decompose` command records and trace |

All final registered inputs compile. Earlier scalar-i1 bridging construction triggered an IREE `tensor.from_elements` type-verification failure during CPU configuration; the diagnostic logs are retained. The final winner reproducer uses a direct scalar tensor<i1> input, as in the original project tests. Those bridge failures were not mislabeled as missing optimization support or used for timings.

What remains unexecuted: exact source-built main; new GPU tests; device-only timing; hardware DRAM counters; independent-session replication; fresh pretrained model runtime checks and strong cross-framework performance comparisons. The supplied GPU results are not fabricated or silently recounted as this audit's work. The README gives executable CPU/GPU/source-build commands and specifies what the CUDA runner measures and what additional profiling is needed.

## 9. Exact source-code index

Source snapshots corresponding to these links are included. Function start links are entry points to inspected surrounding implementations, not claims that one line proves whole-pipeline behavior.

| Ref | Function / mechanism | Pinned IREE path and starting line |
|---|---|---|
| [S1] | `GeneralizeLinalgNamedOpsPass::runOnOperation` | `GlobalOptimization/GeneralizeLinalgNamedOps.cpp:43` |
| [S2] | `BubbleUpExpandShapesPass reshape control` | `DispatchCreation/BubbleUpExpandShapes.cpp:674` |
| [S3] | `dispatch preprocessing registration` | `DispatchCreation/Passes.cpp:178` |
| [S4] | `isSafeToElideUpdateOp` | `Dialect/Stream/Transforms/ElideAsyncCopies.cpp:1401` |
| [S5] | `TensorUpdateOp::fold and adjacent canonicalizations` | `Dialect/Flow/IR/FlowOpFolders.cpp:1103` |
| [S6] | `global pass order; generalization, broadcast folding and transpose propagation` | `GlobalOptimization/Passes.cpp:193` |
| [S7] | `ConvertBroadcastBatchMatmulToMatmul` | `GlobalOptimization/ConvertBatchMatmulToMatmul.cpp:144` |
| [S8] | `CPU attention K1 scheduling threshold` | `Codegen/LLVMCPU/KernelDispatch.cpp:2236` |
| [S9] | `GPU attention vector-distribution fallback schedule` | `Codegen/LLVMGPU/KernelConfig.cpp:1369` |
| [S10] | `convertToOnlineAttention` | `Dialect/LinalgExt/Transforms/TileAttention.cpp:35` |
| [S11] | `DecomposeAttentionPass::runOnOperation` | `Dialect/LinalgExt/Transforms/DecomposeAttention.cpp:32` |
| [S12] | `addCPULinalgExtTileAndVectorizePipeline` | `Codegen/LLVMCPU/Passes.cpp:437` |
| [S13] | `OnlineAttentionOp::getTiledImplementation` | `Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp:3312` |
| [S14] | `isSafeToElideCOW` | `Dialect/Stream/Transforms/MaterializeCopyOnWrite.cpp:41` |
| [S15] | `Stream pipeline registration` | `Dialect/Stream/Transforms/Passes.cpp:255` |
| [S16] | `current main Flow executable CSE fix` | `Dialect/Flow/Transforms/Passes.cpp:244` |
| [S17] | `aggressive reshape default` | `DispatchCreation/Passes.h:95` |
| [S18] | `generalize matmul default` | `GlobalOptimization/Passes.h:118` |


The upstream MLIR fold links [M1]/[M2] are pinned to the newer IREE LLVM dependency. The older dependency's equivalent source is also preserved. The XLA source is pinned to `f824cdd4dce2f2c0ce092fb7cfdc7ad1ebc92163` and TVM to `70ec7d5bf20326ef301ee3f06eb6aed51760331d`. The NVIDIA explanatory document and paper links are primary-source prior art, not benchmark results reproduced here.

[S1]: https://github.com/iree-org/iree/blob/29114a017c104882cab055a49758e8f7055ea0d5/compiler/src/iree/compiler/GlobalOptimization/GeneralizeLinalgNamedOps.cpp#L43-L61
[S2]: https://github.com/iree-org/iree/blob/29114a017c104882cab055a49758e8f7055ea0d5/compiler/src/iree/compiler/DispatchCreation/BubbleUpExpandShapes.cpp#L674-L692
[S3]: https://github.com/iree-org/iree/blob/29114a017c104882cab055a49758e8f7055ea0d5/compiler/src/iree/compiler/DispatchCreation/Passes.cpp#L178-L196
[S4]: https://github.com/iree-org/iree/blob/29114a017c104882cab055a49758e8f7055ea0d5/compiler/src/iree/compiler/Dialect/Stream/Transforms/ElideAsyncCopies.cpp#L1401-L1419
[S5]: https://github.com/iree-org/iree/blob/29114a017c104882cab055a49758e8f7055ea0d5/compiler/src/iree/compiler/Dialect/Flow/IR/FlowOpFolders.cpp#L1103-L1121
[S6]: https://github.com/iree-org/iree/blob/29114a017c104882cab055a49758e8f7055ea0d5/compiler/src/iree/compiler/GlobalOptimization/Passes.cpp#L193-L211
[S7]: https://github.com/iree-org/iree/blob/29114a017c104882cab055a49758e8f7055ea0d5/compiler/src/iree/compiler/GlobalOptimization/ConvertBatchMatmulToMatmul.cpp#L144-L162
[S8]: https://github.com/iree-org/iree/blob/29114a017c104882cab055a49758e8f7055ea0d5/compiler/src/iree/compiler/Codegen/LLVMCPU/KernelDispatch.cpp#L2236-L2254
[S9]: https://github.com/iree-org/iree/blob/29114a017c104882cab055a49758e8f7055ea0d5/compiler/src/iree/compiler/Codegen/LLVMGPU/KernelConfig.cpp#L1369-L1387
[S10]: https://github.com/iree-org/iree/blob/29114a017c104882cab055a49758e8f7055ea0d5/compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/TileAttention.cpp#L35-L53
[S11]: https://github.com/iree-org/iree/blob/29114a017c104882cab055a49758e8f7055ea0d5/compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/DecomposeAttention.cpp#L32-L50
[S12]: https://github.com/iree-org/iree/blob/29114a017c104882cab055a49758e8f7055ea0d5/compiler/src/iree/compiler/Codegen/LLVMCPU/Passes.cpp#L437-L455
[S13]: https://github.com/iree-org/iree/blob/29114a017c104882cab055a49758e8f7055ea0d5/compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp#L3312-L3330
[S14]: https://github.com/iree-org/iree/blob/29114a017c104882cab055a49758e8f7055ea0d5/compiler/src/iree/compiler/Dialect/Stream/Transforms/MaterializeCopyOnWrite.cpp#L41-L59
[S15]: https://github.com/iree-org/iree/blob/29114a017c104882cab055a49758e8f7055ea0d5/compiler/src/iree/compiler/Dialect/Stream/Transforms/Passes.cpp#L255-L273
[S16]: https://github.com/iree-org/iree/blob/29114a017c104882cab055a49758e8f7055ea0d5/compiler/src/iree/compiler/Dialect/Flow/Transforms/Passes.cpp#L244-L262
[S17]: https://github.com/iree-org/iree/blob/29114a017c104882cab055a49758e8f7055ea0d5/compiler/src/iree/compiler/DispatchCreation/Passes.h#L95-L113
[S18]: https://github.com/iree-org/iree/blob/29114a017c104882cab055a49758e8f7055ea0d5/compiler/src/iree/compiler/GlobalOptimization/Passes.h#L118-L136
[M1]: https://github.com/llvm/llvm-project/blob/f1ee8b38f40fd1f0e9596197b8e690eb9b57ec0c/mlir/lib/Dialect/Tensor/IR/TensorOps.cpp#L2805-L2832
[M2]: https://github.com/llvm/llvm-project/blob/f1ee8b38f40fd1f0e9596197b8e690eb9b57ec0c/mlir/lib/Dialect/Tensor/IR/TensorOps.cpp#L2943-L2988
[X1]: https://github.com/openxla/xla/blob/f824cdd4dce2f2c0ce092fb7cfdc7ad1ebc92163/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L3830-L4095
[X2]: https://github.com/openxla/xla/blob/f824cdd4dce2f2c0ce092fb7cfdc7ad1ebc92163/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L10045-L10089
[X3]: https://github.com/openxla/xla/blob/f824cdd4dce2f2c0ce092fb7cfdc7ad1ebc92163/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L7619-L7678
[T1]: https://github.com/apache/tvm/blob/70ec7d5bf20326ef301ee3f06eb6aed51760331d/python/tvm/relax/frontend/nn/llm/kv_cache.py#L122-L159
[T2]: https://github.com/apache/tvm/blob/70ec7d5bf20326ef301ee3f06eb6aed51760331d/python/tvm/relax/frontend/nn/llm/kv_cache.py#L270-L285
[P1]: https://arxiv.org/html/2405.04434v5#S2.SS1.SSS2
[P2]: https://arxiv.org/html/2405.04434v5#S2.SS1.SSS3
[P3]: https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog03_Optimizing_DeepSeek_R1_Throughput_on_NVIDIA_Blackwell_GPUs.md
[P4]: https://arxiv.org/abs/2101.01332
[P5]: https://theory.stanford.edu/~aiken/publications/papers/sosp19.pdf
[L1]: https://llvm.org/docs/MemorySSA.html
