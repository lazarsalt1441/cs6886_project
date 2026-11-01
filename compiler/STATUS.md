# Compiler studies: implementation and evaluation

* Original GQA: C++ head-reuse implementation, structural tests, PyTorch exports,
  CPU timing/allocation analysis and before/after IR. Existing IREE shape passes
  eliminate the tested head expansion under the stronger baseline.
* Projection study: compiled reassociation, partial-RoPE handling, numerical
  contracts, native-attention legality diagnosis and corrected baselines.
* Functional-cache study: read/gather forwarding, runtime-winner factoring,
  adjacent update coalescing and snapshot/COW protection. The operator study
  records 100 structural tests, two COW FileCheck cases, 556 correct bitwise
  checks and 40 expected stock snapshot failures.
* Original cache CPU results include 2.39x/7.90x synthetic-block gains, 3.60x
  winner attention, 2.17–2.34x update chains and 2.58x protected snapshot reads.
  These belong to their specific recorded CPU baselines, not the final GPU model.
* Offline cache CUDA compilation: 43/44 main, 4/4 winner and 21/21 extension
  forms. Conditional fallback compiles for the irregular masked-gather failure.
* Later GPU studies and final pretrained-model measurements are preserved under
  `../results/`; the root README gives the current IREE-only presentation.

The integration is a standalone MLIR tool with a verified IREE IR bridge.
A complete patched in-tree IREE build was not completed. Typed source-patch
proposals, partial source-object builds and executed standalone fixes retain
separate validation levels. Technical reports, unsupported cases and regressions
are preserved in `research/` and `research2/`.
