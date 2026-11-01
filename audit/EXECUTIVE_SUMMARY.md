# Independent IREE/MLIR optimization novelty audit

1 November 2026. The strongest candidate is **demand-driven optimization of competing functional KV-cache versions**, combining read forwarding with selective commit placement. It is a candidate research direction, not an established paper contribution. The array laws are old; the contribution would have to be a general, safe and profitable compiler treatment of real branching inference graphs.

I independently installed the exact experimental compiler and a newer released nightly, inspected pinned IREE/MLIR source, rebuilt the original C++ optimizer, and ran 322 program/configuration variants through four compilation stages. All final variants compiled. Of 3700 main-suite numerical checks, 3340 passed; 360 failed, all in the deliberately diagnosed wide-Q/K native-attention configurations. Failures are retained and excluded from performance comparisons. Extra snapshot/bit-pattern tests are recorded separately.

| Area | Classification | Confidence | Upstream support / boundary | Source links | Independent evidence |
|---|---|---|---|---|---|
| 1. GQA expansion | A (tested family) | High | Generic contraction generalization + reshape motion + fusion; native/broadcast pathways also exist. | [S1], [S2], [S3], [S7] | Grouped fixture: 2 dispatches default/O3 -> 1 with both flags; compact K consumed directly. |
| 2. KV read forwarding | B overall; C for conditional fork-gather subcase | High for static FP32 fixtures | MLIR exact-slice forwarding and Stream update elision already exist. Conditional reads through competing cache versions are missed in tested configurations. | [M1], [S4], [S5] | Two pool sizes; actual C++ pass emits two conditional reads; all checks pass on both binaries. |
| 3. Winner-cache factoring | C for equal-region/concat fixtures | High for restricted patterns | No equivalent selection-before-update transformation triggered. Generic fusion can reduce dispatches without factoring the commit. | [S4], [S6], [X2] | Actual C++ insert and concat patterns tested. Explicit old-cache contract passes. Full-model A=0, B=60 matches. |
| 4. Latent projection placement | C in tested IREE; established prior art | High for explicit FP32 graphs | Default/O3, strong, aggressive and reassociation profiles retain per-history projections. General projection chains differ from quantized reassociation. | [S6], [X1], [P1], [P3] | Two complete softmax shapes plus actual rank-3 C++ score rewrite; numerical checks pass. |
| 5. Partial-RoPE projection | C at explicit split/concat boundary; D for general RoPE recognition | Medium | Content projection is not absorbed in tested split graphs; arbitrary rotary math/layout recognition was not established. | [S6], [P2] | Actual concat-based C++ score pattern plus complete attention with already-rotated operands pass. |
| 6. Update-chain forwarding/coalescing | B; C for eligible adjacent-chain subcase | High for narrow cases | Exact overwrite folds, slice composition and dead update elimination exist. Adjacent small-patch coalescing and conditional overlapping-chain reads remain separate. | [M2], [S4], [S5] | Same-slice control folds; disjoint read retains copies; actual eligible coalescing fires but gives no reliable speedup. |
| 7. Native attention fallback | B: manual upstream route exists; automatic safeguard fails | High on tested CPU; D for current T4 | Existing early online-attention conversion/decomposition avoids failure; ordinary scheduling still fails wide Q/K. | [S8], [S9], [S10], [S11], [S12], [S13] | 64/128 QK pass; 256/512 fail 20/20 per affected ordinary profile. Early decomposition passes all six shapes. |


**The main corrections to a novelty narrative:**

1. Explicit-repeat GQA elimination is already available through existing flags; O3 alone is not the deciding setting.
2. IREE/MLIR already implement some read-after-write and update elimination. Only the demonstrated conditional/forked remainder is a gap.
3. Projection absorption and decoupled RoPE are prior art. XLA also implements cost-based nested-dot reassociation.
4. Native-attention decomposition already exists upstream and works when scheduled early on the tested CPU shapes. An automatic correctness safeguard/repair remains missing in the tested route.
5. Fewer dispatches or copies are not a profitability proof. Actual adjacent coalescing is neutral/slower here, and concat factoring changes dramatically with compiler settings.

**Paper recommendation:** pursue a unified functional-cache analysis for branching inference first; pursue automatic MLA recognition/placement second; keep range-aware Stream analysis as a supporting third direction. Do not center a paper on GQA repeat removal, the matrix identity, simple adjacent merging, or a fallback around wrong code.

Before a central contribution is credible, demonstrate natural exporter coverage, correct immutable/snapshot behavior, real beam/speculative workloads, end-to-end GPU benefit against tuned stock IREE and strong PyTorch/paged-cache baselines, mixed precision and a second GPU. A new-to-IREE pass alone does not establish global novelty or publication readiness.

Full commands, scope, timings, regressions, source permalinks and the implementation plan are in `TECHNICAL_REPORT.md`.

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
