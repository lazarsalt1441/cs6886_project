from pathlib import Path
import json,hashlib,subprocess,shutil
P=Path(__file__).resolve().parents[1]
MAIN='29114a017c104882cab055a49758e8f7055ea0d5';LATEST='94cb5993a90523669770a2a7a998d0a9a09147b5';PIN='89ece87b89852100c0a4ac3d515e14842d090967';LLVM='f1ee8b38f40fd1f0e9596197b8e690eb9b57ec0c';XLA='f824cdd4dce2f2c0ce092fb7cfdc7ad1ebc92163';TVM='70ec7d5bf20326ef301ee3f06eb6aed51760331d'
def load(n):return json.loads((P/n).read_text())
def table(headers,rows):return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows)+'\n'
def line(repo,file,needle):
 p=P/'sources'/repo/file
 return next((i for i,s in enumerate(p.read_text().splitlines(),1) if needle in s),1)
specs=[
 ('S1','GlobalOptimization/GeneralizeLinalgNamedOps.cpp','if (enableGeneralizeMatmul','GeneralizeLinalgNamedOpsPass::runOnOperation'),
 ('S2','DispatchCreation/BubbleUpExpandShapes.cpp','return (enableReshapeMovementAcrossReductions','BubbleUpExpandShapesPass reshape control'),
 ('S3','DispatchCreation/Passes.cpp','createBubbleUpExpandShapesPass','dispatch preprocessing registration'),
 ('S4','Dialect/Stream/Transforms/ElideAsyncCopies.cpp','static bool isSafeToElideUpdateOp','isSafeToElideUpdateOp'),
 ('S5','Dialect/Flow/IR/FlowOpFolders.cpp','OpFoldResult TensorUpdateOp::fold','TensorUpdateOp::fold and adjacent canonicalizations'),
 ('S6','GlobalOptimization/Passes.cpp','createConvertBatchMatmulToMatmulPass','global pass order; generalization, broadcast folding and transpose propagation'),
 ('S7','GlobalOptimization/ConvertBatchMatmulToMatmul.cpp','class ConvertBroadcastBatchMatmulToMatmul','ConvertBroadcastBatchMatmulToMatmul'),
 ('S8','Codegen/LLVMCPU/KernelDispatch.cpp','constexpr int64_t kTypicalK1Threshold','CPU attention K1 scheduling threshold'),
 ('S9','Codegen/LLVMGPU/KernelConfig.cpp','setAttentionVectorDistributionConfig(','GPU attention vector-distribution fallback schedule'),
 ('S10','Dialect/LinalgExt/Transforms/TileAttention.cpp','void convertToOnlineAttention','convertToOnlineAttention'),
 ('S11','Dialect/LinalgExt/Transforms/DecomposeAttention.cpp','void DecomposeAttentionPass::runOnOperation','DecomposeAttentionPass::runOnOperation'),
 ('S12','Codegen/LLVMCPU/Passes.cpp','void addCPULinalgExtTileAndVectorizePipeline','addCPULinalgExtTileAndVectorizePipeline'),
 ('S13','Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp','OnlineAttentionOp::getTiledImplementation','OnlineAttentionOp::getTiledImplementation'),
 ('S14','Dialect/Stream/Transforms/MaterializeCopyOnWrite.cpp','static bool isSafeToElideCOW','isSafeToElideCOW'),
 ('S15','Dialect/Stream/Transforms/Passes.cpp','createMaterializeCopyOnWritePass','Stream pipeline registration'),
 ('S16','Dialect/Flow/Transforms/Passes.cpp','auto &executablePassManager','current main Flow executable CSE fix'),
 ('S17','DispatchCreation/Passes.h','Option<bool> enableAggressiveReshapeMovement','aggressive reshape default'),
 ('S18','GlobalOptimization/Passes.h','Option<bool> generalizeMatmul','generalize matmul default'),
]
refs={};source_rows=[]
for key,file,needle,desc in specs:
 n=line('upstream-iree',file,needle);url=f'https://github.com/iree-org/iree/blob/{MAIN}/compiler/src/iree/compiler/{file}#L{n}-L{n+18}';refs[key]=url;source_rows.append([f'[{key}]',f'`{desc}`',f'`{file}:{n}`'])
refs.update({'M1':f'https://github.com/llvm/llvm-project/blob/{LLVM}/mlir/lib/Dialect/Tensor/IR/TensorOps.cpp#L2805-L2832','M2':f'https://github.com/llvm/llvm-project/blob/{LLVM}/mlir/lib/Dialect/Tensor/IR/TensorOps.cpp#L2943-L2988','X1':f'https://github.com/openxla/xla/blob/{XLA}/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L3830-L4095','X2':f'https://github.com/openxla/xla/blob/{XLA}/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L10045-L10089','X3':f'https://github.com/openxla/xla/blob/{XLA}/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L7619-L7678','T1':f'https://github.com/apache/tvm/blob/{TVM}/python/tvm/relax/frontend/nn/llm/kv_cache.py#L122-L159','T2':f'https://github.com/apache/tvm/blob/{TVM}/python/tvm/relax/frontend/nn/llm/kv_cache.py#L270-L285','P1':'https://arxiv.org/html/2405.04434v5#S2.SS1.SSS2','P2':'https://arxiv.org/html/2405.04434v5#S2.SS1.SSS3','P3':'https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog03_Optimizing_DeepSeek_R1_Throughput_on_NVIDIA_Blackwell_GPUs.md','P4':'https://arxiv.org/abs/2101.01332','P5':'https://theory.stanford.edu/~aiken/publications/papers/sosp19.pdf','L1':'https://llvm.org/docs/MemorySSA.html'})
definitions='\n'.join(f'[{k}]: {v}' for k,v in refs.items())
classification=[
 ['1. GQA expansion','A (tested family)','High','Generic contraction generalization + reshape motion + fusion; native/broadcast pathways also exist.','[S1], [S2], [S3], [S7]','Grouped fixture: 2 dispatches default/O3 -> 1 with both flags; compact K consumed directly.'],
 ['2. KV read forwarding','B overall; C for conditional fork-gather subcase','High for static FP32 fixtures','MLIR exact-slice forwarding and Stream update elision already exist. Conditional reads through competing cache versions are missed in tested configurations.','[M1], [S4], [S5]','Two pool sizes; actual C++ pass emits two conditional reads; all checks pass on both binaries.'],
 ['3. Winner-cache factoring','C for equal-region/concat fixtures','High for restricted patterns','No equivalent selection-before-update transformation triggered. Generic fusion can reduce dispatches without factoring the commit.','[S4], [S6], [X2]','Actual C++ insert and concat patterns tested. Explicit old-cache contract passes. Full-model A=0, B=60 matches.'],
 ['4. Latent projection placement','C in tested IREE; established prior art','High for explicit FP32 graphs','Default/O3, strong, aggressive and reassociation profiles retain per-history projections. General projection chains differ from quantized reassociation.','[S6], [X1], [P1], [P3]','Two complete softmax shapes plus actual rank-3 C++ score rewrite; numerical checks pass.'],
 ['5. Partial-RoPE projection','C at explicit split/concat boundary; D for general RoPE recognition','Medium','Content projection is not absorbed in tested split graphs; arbitrary rotary math/layout recognition was not established.','[S6], [P2]','Actual concat-based C++ score pattern plus complete attention with already-rotated operands pass.'],
 ['6. Update-chain forwarding/coalescing','B; C for eligible adjacent-chain subcase','High for narrow cases','Exact overwrite folds, slice composition and dead update elimination exist. Adjacent small-patch coalescing and conditional overlapping-chain reads remain separate.','[M2], [S4], [S5]','Same-slice control folds; disjoint read retains copies; actual eligible coalescing fires but gives no reliable speedup.'],
 ['7. Native attention fallback','B: manual upstream route exists; automatic safeguard fails','High on tested CPU; D for current T4','Existing early online-attention conversion/decomposition avoids failure; ordinary scheduling still fails wide Q/K.','[S8], [S9], [S10], [S11], [S12], [S13]','64/128 QK pass; 256/512 fail 20/20 per affected ordinary profile. Early decomposition passes all six shapes.'],
]
comparison=table(['Area','Classification','Confidence','Upstream support / boundary','Source links','Independent evidence'],classification)
summaries={v:load(f'results/{v}/summary.json') for v in ['pinned','latest']};pairs={v:load(f'results/{v}/paired_timings.json') for v in ['pinned','latest']}
checks=sum(len(r.get('validation',[])) for rs in summaries.values() for r in rs);passed=sum(sum(z['passed'] for z in r.get('validation',[])) for rs in summaries.values() for r in rs);configs=sum(map(len,summaries.values()))
bench_rows=[]
for label in ['pinned','latest']:
 for r in pairs[label]:
  s=r['variants'][r['selected_stock']];c=r['variants'][r['selected_custom']];ci=r['paired_bootstrap_95pct'];bench_rows.append([label,r['case'],r['selected_stock'].replace('stock_',''),r['selected_custom'].replace('custom_',''),f"{s['median_us']:.3f}",f"{c['median_us']:.3f}",f"{r['speedup_vs_selected']:.3f}x",f"{ci[0]:.3f}-{ci[1]:.3f}",f"{r['matched_strong_speedup']:.3f}x"])
bench=table(['Compiler','Fixture','Stock profile','Custom profile','Stock us','Custom us','Selected ratio','Within-run bootstrap 95%','Matched strong ratio'],bench_rows)
profile_rows=[]
for r in pairs['latest']:
 def fmt(k):return f"{r['variants'][k]['median_us']:.3f}" if k in r['variants'] else '-'
 profile_rows.append([r['case'],fmt('stock_default'),fmt('stock_O3'),fmt('stock_strong'),fmt('stock_aggressive'),fmt('stock_max_reassociation'),fmt('custom_default'),fmt('custom_strong')])
profiles=table(['Fixture','Default','O3','Strong','Aggressive','Reassociation','Custom default','Custom strong'],profile_rows)
alloc=[]
for case in ['gather_forks_512_stock','gather_forks_512_actual','gather_forks_8192_stock','gather_forks_8192_actual','winner_stock','winner_custom','projection_512_stock','projection_512_custom','coalesce_stock','coalesce_actual','concat_stock','concat_actual']:
 r=next(x for x in summaries['latest'] if x['case']==case and x['config']=='strong');alloc.append([case,f"{r['allocator']['delta']['device_bytes_allocated']/101:,.0f}",r['ir_counts']['flow']['flow_dispatch'],r['ir_counts']['stream']['stream_copy']])
alloc_table=table(['Latest strong fixture','Allocated bytes / invocation','Flow dispatch calls','Stream copy commands'],alloc)
native=[]
for D,V in [(64,64),(128,128),(256,128),(512,128),(128,512),(512,512)]:
 row=[D,V]
 for label in ['pinned','latest']:
  for cfg in ['default','O3','strong','early_decompose']:
   r=next(x for x in summaries[label] if x['case']==f'native_{D}_{V}' and x['config']==cfg);vs=r['validation'];row.append(f"{sum(z['passed'] for z in vs)}/{len(vs)}")
 native.append(row)
native_table=table(['QK','V','Pinned default','Pinned O3','Pinned strong','Pinned early','Latest default','Latest O3','Latest strong','Latest early'],native)
snap=[]
for label in ['pinned','latest']:
 for r in load(f'results/{label}/snapshots/summary.json'):
  snap.append([label,r['variant'],f"{sum(c['output_correct'] and c['snapshot_correct'] and c['input_preserved'] for c in r['checks'])}/{len(r['checks'])}",f"{r.get('timing',{}).get('median_us',0):.3f}"])
snap_table=table(['Compiler','Explicit-snapshot fixture','All three checks pass','Median us'],snap)
summary=f'''# Independent IREE/MLIR optimization novelty audit

1 November 2026. The strongest candidate is **demand-driven optimization of competing functional KV-cache versions**, combining read forwarding with selective commit placement. It is a candidate research direction, not an established paper contribution. The array laws are old; the contribution would have to be a general, safe and profitable compiler treatment of real branching inference graphs.

I independently installed the exact experimental compiler and a newer released nightly, inspected pinned IREE/MLIR source, rebuilt the original C++ optimizer, and ran {configs} program/configuration variants through four compilation stages. All final variants compiled. Of {checks} main-suite numerical checks, {passed} passed; {checks-passed} failed, all in the deliberately diagnosed wide-Q/K native-attention configurations. Failures are retained and excluded from performance comparisons. Extra snapshot/bit-pattern tests are recorded separately.

{comparison}

**The main corrections to a novelty narrative:**

1. Explicit-repeat GQA elimination is already available through existing flags; O3 alone is not the deciding setting.
2. IREE/MLIR already implement some read-after-write and update elimination. Only the demonstrated conditional/forked remainder is a gap.
3. Projection absorption and decoupled RoPE are prior art. XLA also implements cost-based nested-dot reassociation.
4. Native-attention decomposition already exists upstream and works when scheduled early on the tested CPU shapes. An automatic correctness safeguard/repair remains missing in the tested route.
5. Fewer dispatches or copies are not a profitability proof. Actual adjacent coalescing is neutral/slower here, and concat factoring changes dramatically with compiler settings.

**Paper recommendation:** pursue a unified functional-cache analysis for branching inference first; pursue automatic MLA recognition/placement second; keep range-aware Stream analysis as a supporting third direction. Do not center a paper on GQA repeat removal, the matrix identity, simple adjacent merging, or a fallback around wrong code.

Before a central contribution is credible, demonstrate natural exporter coverage, correct immutable/snapshot behavior, real beam/speculative workloads, end-to-end GPU benefit against tuned stock IREE and strong PyTorch/paged-cache baselines, mixed precision and a second GPU. A new-to-IREE pass alone does not establish global novelty or publication readiness.

Full commands, scope, timings, regressions, source permalinks and the implementation plan are in `TECHNICAL_REPORT.md`.

{definitions}
'''
(P/'EXECUTIVE_SUMMARY.md').write_text(summary)
report=f'''# Exhaustive-request audit: verified coverage, missed cases and limits

Independent research review, 1 November 2026. “Missing” below always means a **specified graph family in the tested binaries/settings**. This audit does not prove absence across every frontend, target, dtype, tuning specification or possible pass pipeline. It does not call a lack of documentation proof of absence.

## Executive judgment

The most promising central contribution is a compiler analysis of **functional KV versions with small patches and multiple competing consumers**, preserving observation/ownership semantics while deciding what to materialize, forward and commit. This is stronger than presenting seven unrelated rewrites as novel.

The project has credible new-to-IREE missed cases. It has not established a globally new optimization principle or end-to-end LLM GPU acceleration. GQA elimination is already handled in the demonstrated family. Projection placement has substantial existing mathematical, systems and compiler precedent. Native fallback is primarily a correctness/pipeline repair.

{comparison}

## 1. Versions, provenance and what actually ran

| Component | Exact evidence |
|---|---|
| Supplied experiment | IREE package 3.13.0; embedded compiler 3.13.0rc20261016, `{PIN}`; LLVM 24.0.0git |
| Independent pinned installation | Same embedded version and full commit; `results/pinned/version.txt` |
| Newer binary tested | 3.14.0rc20261031, `{LATEST}`; GitHub release wheels; `results/latest/version.txt` |
| Upstream main source inspected | `{MAIN}`, committed 30 October 2026 at 11:48 UTC |
| LLVM/MLIR in experiment source | `32d702efa29d0fdd02cff7b79dea8fdbbcf059d4` |
| LLVM/MLIR in newer source | `{LLVM}` |
| Independent CPU | Intel Xeon Platinum 8573C, x86-64; host targeting; local-sync; one CPU affinity selected during measurements |
| Original project C++ tool | Rebuilt with the package's hash-pinned MLIR/LLVM 19.1.1 SDK and GCC; textual IR bridge, not IREE-private C++ ABI mixing |
| GPU in this execution environment | No NVIDIA device/`nvidia-smi`; no new GPU execution performed |

The newer wheel is **not the exact inspected main HEAD**. Comparing the two source trees shows one compiler change: main fixes retention of nested Flow executable canonicalization/CSE ([S16]); other differences are test configuration files. That fix may affect generated IR, so direct main-binary execution remains an explicit boundary. The report never labels the nightly as a source-built main compiler. Commands to close this boundary are in README.

The attached consolidated PDF identified the experimental commit. I also retrieved the previously saved consolidated result archive and the current real-model test package. The latter supplies the actual C++ sources and prepared full-model IR. Historical T4 measurements in those files are **reported prior evidence**, not rerun numbers. The recovered real-model package is newer supporting material; its existing pretrained numerical results were not independently rerun here because its checkpoint/parameter archive was not packaged for this audit.

The original pass was actually rebuilt and executed here: two gather patterns forward; one insert-based commit factors; eligible adjacent updates coalesce; key projection and concat-split rotary patterns rewrite. The real-model ordinary decode graph remains byte-identical after the requested pass pipeline, with zero markers. The two-candidate custom lookahead graph produces 60 concat-commit markers. These are pass/applicability results, not newly executed pretrained inference.

## 2. Experimental matrix and evidence boundaries

The final main suite contains 42 MLIR programs and {configs} distinct program/configuration records across two compiler versions. Each was separately compiled to global optimization, Flow, Stream and final VMFB: {configs*4} successful phase invocations. The main suite performed {checks} numerical executions: {passed} passed and {checks-passed} failed. Each ordinary failed native variant fails 20/20 seeds; the failures are not intermittent or hidden. Original concat structural tests pass 8/8; 8 bit-pattern scenarios with two compiled variants per compiler pass separately, including NaNs, infinities and signed zero. Explicit snapshot tests add 10 seeds for each of three variants on each compiler.

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

{native_table}

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

{bench}

These measurements are CPU synthetic kernels/graphs, not T4 measurements or model tokens/s. Within-run confidence intervals do not account for thermal/VM noise or independent sessions. Values within a few percent of 1 should be treated as neutral. The large projection result is one favorable complete-attention shape, not a geometric-mean LLM result.

### 4.2 Default vs O3 vs the stronger relevant settings, latest binary

All cells are holdout median microseconds from the same paired study:

{profiles}

O3 is not uniformly fastest. In particular, default transformed concat scheduling can substantially outperform the same transformed program under strong settings. Selecting profiles after looking at holdout results would be biased; the saved calibration data and chosen profiles make that choice reviewable.

### 4.3 Explicit snapshot and immutable-input contract

Each check validates the committed output, returned old cache, and original input buffer. Timings are randomized 201-observation synchronous invocations with identical output/ownership contracts:

{snap_table}

The custom guarded comparison is the appropriate insertion-commit evidence when old values must remain available. The older project-reported cast-hidden snapshot failures and proposed COW repairs are separate correctness work. This simple independent unguarded two-output fixture passes; the older bug is neither reproduced nor disproven by it.

### 4.4 Allocation/copy distinctions, latest strong profile

Allocator values are cumulative runtime allocation requests divided by 101 timed invocations after warmup. They are not peak RSS, reserved VRAM or hardware DRAM bytes. Copy counts are static Stream command occurrences and exclude implicit memory accesses inside kernels.

{alloc_table}

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

{table(['Ref','Function / mechanism','Pinned IREE path and starting line'],source_rows)}

The upstream MLIR fold links [M1]/[M2] are pinned to the newer IREE LLVM dependency. The older dependency's equivalent source is also preserved. The XLA source is pinned to `{XLA}` and TVM to `{TVM}`. The NVIDIA explanatory document and paper links are primary-source prior art, not benchmark results reproduced here.

{definitions}
'''
(P/'TECHNICAL_REPORT.md').write_text(report)
(P/'sources/source-links.json').write_text(json.dumps(refs,indent=2))
print('Report words',len(report.split()),'configs',configs,'checks',checks,'passed',passed)
