# Project status — R13 completed

The working artifact is under `research2/`. Start with its [technical report](research2/TECHNICAL_REPORT.md), [reproduction guide](research2/README.md), [results](research2/RESULTS.md), [extensions](research2/EXTENSIONS.md), and [GPU commands](research2/GPU_RUN.md).

## Completed

- Original M0–M5 environment, C++ head-reuse implementation, tests, CPU benchmarks and reproducible repository checkpoints. Stronger upstream IREE flags already solve that original repeat pattern; the default-baseline result is not a novelty claim.
- R0–R4 projection/online-attention investigation retained in `research/`, including its limitations and corrected baseline comparison.
- R5–R10 forked cache-demand forwarding, masked gathered updates, gather-index correctness repair, runtime-winner commit factoring, model/escaping-cache controls and14 isolated CPU timing groups.
- R11–R13 adjacent update-chain coalescing, original-cache snapshot corruption diagnosis, two compiled C++ snapshot/COW repairs, five additional isolated timing groups, GPU reproduction scripts and final report.

Seven attention-cache C++ passes are compiled and executed.100 structural unittest cases plus two COW FileCheck cases pass.556 exact dataflow/commit/snapshot checks pass;40 expected stock snapshot failures are recorded separately. Numerical attention/model tests and negative controls are retained.

Measured CPU gains against the best correct baselines tested: synthetic blocks2.39×/7.90×; winner-cache attention3.60×;4–16-token update chains2.17–2.34×; corrected snapshot attention2.58×. Regressions and no-op cases remain visible. Incorrect stock outputs are not speedup baselines.

Both correctness defects are reproduced on the pinned3.13 compiler; the native snapshot defect also reproduces on3.12. The implicated source files are unchanged at the separately fetched upstream HEAD9ea0d950. Source inspection does not substitute for executing that HEAD.

CUDA offline results:43/44 main forms,4/4 winner forms,21/21 extension forms compile. Conditional fallback handles the irregular masked-gather bufferization failure. GPU runtime measurements are absent.

## Limitations and remaining validation

The working implementation is a standalone MLIR C++ tool with a verified IREE IR bridge. A complete in-tree source compiler was not built: an8GiB OOM and later workspace rollback interrupted the attempt. The gather source object compiled; the typed COW patch remains a proposal. Both standalone fixes execute end to end and have native regression coverage.

No GPU numerical/performance, hardware DRAM-counter, pretrained serving-model throughput, or global-first-publication claim is made. Independent novelty review and broader model/backend evaluation remain research tasks. Paper drafting was deferred; the completed report is `research2/TECHNICAL_REPORT.md`.

Local Git checkpoints preserve every major milestone. `RESUME.md` records the exact current environment, working commands and next steps. The project archive includes code, Git, IR, binaries, raw results and plots; rebuildable SDK/vendor trees are excluded.
# 2026-10-27 follow-up: GQA flag ablation

Completed 160 Flow compilations over 20 saved fixtures; on the original example both matmul generalization and aggressive reshape movement were necessary, while O3 was not. 96 CPU numerical checks passed. A scalar-observation graph exposes a remaining multi-use broadcast; a manual head-index rewrite removes it and passes 48 dynamic-head checks. See `ablation/REPORT.md`. This is a proof of concept, not a general implemented pass; GPU and broad real-model validation remain open.



## GPU follow-up checkpoint, 2026-10-31
See FOLLOWUP.md for new code, executed tests, Colab commands and limitations. New diagnostic: research/native_width_probe.py; runner: run_gpu_followup.py. CPU: original+decomposed fallback pass 10 seeds; native wide-Q/K probe fails. All nine candidates compile for sm_75; GPU execution pending. Six selection tests pass. Next command on Colab: python run_gpu_followup.py --mode probe --arch sm_75 --seeds 10 --trials 101. Next three tasks: isolate T4 width failure; validate fallback; repeat calibrated cache/winner benchmarks. Local dependencies temporarily at /workspace/scratch/4088b657d6f5/followup_deps; PYTHONPATH must include it for local IREE.
