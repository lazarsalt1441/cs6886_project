# IREE maintainer issue draft — sparse reads through tensor.insert_slice

> Not posted upstream: GitHub integration currently returned HTTP 403 for creating an issue in iree-org/iree. This is a contributor-reviewed proposal, not a claim that a rewrite is missing in IREE.

## Proposal

Investigate whether sparse `tensor.extract` reads from the result of a `tensor.insert_slice` can be forwarded to the source patch or base tensor before bufferization, when a larger temporary cache version would otherwise be constructed.

Example:

```mlir
%updated = tensor.insert_slice %patch into %base[%offset] [%size] [1]
%value = tensor.extract %updated[%i] : tensor<?xf32>
```

The candidate rewrite selects **which extraction to execute**:
```
if offset <= i < offset+size:
    value = patch[i-offset]
else:
    value = base[i]
```

An eager select of two extracts is **not** generally legal: the nonselected extraction can be out of bounds. For multidimensional slices or strides, each dimension must be validated; overflow and dynamic-index cases require care.

## Motivation and evidence

This can matter for sparse reads of competing functional KV-cache versions in branching or speculative decoding. The existing [standalone MLIR implementation](https://github.com/lazarsalt1441/cs6886_project/blob/main/compiler/src/CacheReadForwarding.cpp) targets a restricted pattern; it has **not** been integrated or compiled against current IREE. The project reports an isolated sparse-cache benchmark with a 1.44x speedup and 99.92% lower DRAM traffic; this does **not** establish a speedup in current upstream IREE or full-model serving.

We do not claim novelty of the array identity or that IREE currently misses the pattern. The simple static profitability heuristic in the standalone implementation is not production-grade.

## Questions

1. Do existing IREE/MLIR passes already produce this forwarding in the presence of multiple functional cache versions?
2. If not, what is the best dialect and pass stage before bufferization?
3. Would a conservative rewrite be welcome once demonstrated on an upstream reproducer with a confirmed eliminated allocation?

We would first submit a minimal regression test showing a missed optimization, then a focused patch with legality and negative tests, before/after bufferization IR, and benchmarks comparing current IREE flags.

_AI assistance disclosure: This issue draft and portions of the standalone source were prepared with AI assistance. A human contributor should review all claims and code before posting or submitting an implementation._
