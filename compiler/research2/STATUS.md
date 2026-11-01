# Functional-cache study record

> Operator-level CPU and compile-only study. The repository root README describes the final pretrained-model GPU evaluation.

The R13 implementation includes read/gather forwarding, winner commit factoring,
adjacent update coalescing and snapshot/COW protection. Seven standalone C++
passes were compiled and exercised through IREE. Validation records include100
structural tests, two COW FileCheck cases,556 correct bitwise checks and40
expected stock snapshot failures. Incorrect stock outputs are excluded from
performance baselines.

The CPU evaluation retains14 main and five extension timing groups. Against the
best correct baselines tested, recorded gains include2.39x/7.90x synthetic blocks,
3.60x winner attention,2.17–2.34x update chains and2.58x protected snapshot reads.
Attention numerical errors, raw observations, dense/no-match controls and
regressions are retained in results/. These operator-specific comparisons are
separate from the final model latency and DRAM results.

Offline CUDA compilation completed43/44 main forms,4/4 winner forms and21/21
extension forms. The irregular masked-gather failure is preserved; conditional
forwarding is the compatible fallback for that case. Compilation alone is not a
GPU correctness or performance measurement.

Source investigation identified gather index-projection and borrowed-cache
ownership defects on the pinned compiler. The native snapshot counterexample
also reproduces on IREE3.12. Source snapshots at the inspected upstream revision
retain the implicated code; this does not establish behavior of every newer
build. A source object compiled, but the full patched in-tree compiler build was
not completed. Typed source patches remain proposals distinct from the executed
standalone repairs.

Read-through-write, selection factoring and copy-on-write storage are established
ideas. The contribution is the supported pattern implementation, investigation
and evaluation; global novelty, universal graph coverage and general serving
throughput are not established.
