# Project status — 1 November 2026

* M0: PyTorch GQA reference/export, exact environment and CPU baseline preserved.
* M1: Actual IR/allocation investigation. Existing strong IREE optimizes the tested repeated-head expansion; initial weak-baseline speedup is not a novelty claim. Source-backed independent audit retained.
* M2: Genuine C++ MLIR drivers and structural/negative tests. Cache read/gather forwarding, commit factoring, coalescing, ownership protection, projection experiments and bounded stable-trig repair preserved.
* M3: CPU numerical integration and real pretrained SmolLM2 export. Natural decode has zero commit matches; custom two-candidate driver has 60 concat-commit matches. Final repaired GPU comparisons pass the fixed numerical gates, including both forced winners and input immutability.
* M4: CPU timing/allocations, T4 timing, PTX/assembly and final Nsight counters. Track A: 51 valid profiles; B: 61. Final B latency reductions within IREE are 26.46%, 4.64%, 5.52% at contexts2048/3072/4096. Full-model total DRAM is nearly unchanged at longer contexts; synthetic forwarding has strong traffic reduction. Presentation tables and default plots focus on IREE baseline/optimized comparisons; raw measurements remain intact.
* M5: Extensions, failure records and bounded upstream audit preserved. Local consolidation/reproducibility verification is recorded under validation/. No new final course report is written in this update.

Limitations: one FP32 small model and GPU family; custom branching step, not complete serving/generation; no new research novelty proved; no-match timing regressions and clock changes require independent confirmation. Snapshot/input-ownership failures remain quarantined. Kernel/replay timing cannot replace unprofiled latency. Published historical documents may reflect an earlier checkpoint; final claims are in root README and results/dram.

Remaining: independently repeat B under steadier GPU conditions, investigate no-match timing behavior and full-model kernel differences, then prepare the requested <=2-page report and course presentation. The report was explicitly deferred by the user.
