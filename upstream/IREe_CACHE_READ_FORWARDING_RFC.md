# RFC: Demand-driven forwarding of tensor.insert_slice reads in IREE

**Status:** proposed investigation, **not** an upstream-ready patch. This is a focused handoff for a possible IREE contributor issue / draft PR.

**Provenance:** Adaptation candidate from [CS6886 standalone MLIR pass](../compiler/src/CacheReadForwarding.cpp). The standalone driver uses MLIR 19 and is not integrated with current IREE.

## Problem and candidate canonicalization

When a functional tensor update creates a new cache version that is used only for a few scalar reads, downstream bufferization may allocate or copy a much larger temporary tensor than the requested reads justify:

```mlir
%updated = tensor.insert_slice %patch into %base[%offset] [%size] [1]
%value = tensor.extract %updated[%i] : tensor<?xf32>
```

For an in-bounds read and a valid insert_slice, the intended equivalence is:

```text
delta = i - offset
if 0 <= delta < size:
    value = patch[delta]
else:
    value = base[i]
```

**Key requirement:** this is *conditional execution*, not an eager arith.select of two tensor.extract operations. The nonselected access can be out-of-bounds. For rank > 1, transform every dimension. A stride > 1 requires a divisibility condition and positive index bounds.

This is an array identity, **not** a claimed novel algorithm. The potential IREE contribution would be conservative pattern recognition, profitability, and correct interaction with bufferization.

## Existing implementation and limitations

- Original pass: [CacheReadForwarding.cpp](../compiler/src/CacheReadForwarding.cpp).
- Optional gather-specific form: [CacheGatherForwarding.cpp](../compiler/src/CacheGatherForwarding.cpp).
- The pass is a standalone `func.func` rewrite over tensor.extract inside parallel linalg.generic; it is not presently an in-tree IREE pass.
- The original includes a static capacity/demand heuristic and multi-use conditions. These are *hypotheses*, not a robust production cost model.
- If the updated cache has other unsupported uses (including return/escape), forwarding only the scalar reads does not eliminate materialization. A fully optimized rewrite must check all users and memory ownership semantics.
- A real GPU implementation must establish the pattern occurs at the target compiler stage and compare against IREE's current canonicalization and dispatch/bufferization behavior.
- Dynamic dimensions/offsets, overflow, strided regions, index types, aliasing, and special encodings need explicit handling or conservative rejection.
- Avoid modifying behavior for lowering_config-marked ops or opaque dispatch implementations without proving legality.
- Do not attach the standalone 1.44x isolated speedup to an upstream patch without matching IREE integration and target GPU measurements.

## Required upstream readiness checklist

- [ ] Confirm the same transformation is not already produced by mainline IREE on a minimal IR reproducer, using relevant optimization flags.
- [ ] Pick the correct MLIR dialect and pass stage with maintainer input; likely a tensor-level, pre-bufferization rewrite.
- [ ] Port implementation against current IREE's LLVM/MLIR revision and build successfully.
- [ ] Add FileCheck tests: inside/outside patch; boundary indices; multi-axis; dynamic offset; strided; unsupported/multiuse escape; zero-size where legal; negative/idempotence.
- [ ] Prove numerical equivalence and no invalid memory access for the accepted patterns with independent expected outputs.
- [ ] Show compiler IR with and without the rewrite, including actual elimination of large temporary storage.
- [ ] Run target-specific benchmarks with baseline flags enabled, at least three pool/demand ratios and an explicitly losing dense-read control.
- [ ] Compare on one realistic attention/branching decode path; report unchanged/regressed cases and profiler uncertainty.
- [ ] Add a focused integration test and documentation; disclose AI-assisted contributions per IREE policy.

## Draft maintainer question

> Would IREE maintainers be interested in a conservative pre-bufferization rewrite that forwards sparse reads through `tensor.insert_slice`, provided it eliminates an otherwise-materialized temporary cache version? We have a standalone MLIR implementation and tests, but do **not** claim the optimization is novel or absent from current IREE. Before porting, we'd appreciate guidance on the correct pass stage, existing canonicalizations, and a suitable cost model. We intend to submit only once an in-tree test demonstrates a real missed optimization.

## Reference evidence (not sufficient for upstream acceptance)

The CS6886 README reports an isolated sparse-cache case with 1.44× latency speedup and 99.92% reduced DRAM traffic, plus separate model-level *winner-commit factoring* results. Those are distinct transformations; **do not attribute the model-level 26.46% improvement to read forwarding alone**.

This document is a contribution-preparation artifact, not the upstream PR.
