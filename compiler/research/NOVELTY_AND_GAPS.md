# Research claim audit

> Experiment-stage record. See the repository root README for the final GPU evaluation and current reproduction entry point.

## Established by this artifact

1. A minimal, seed-free numerical counterexample on a pinned CPU compiler. It is independent of the custom projection rewrite.
2. An explanatory algebraic counterexample: online token-state merging does not justify treating partial feature sums as additional tokens. This distinction is straightforward algebra, not a newly invented softmax algorithm.
3. Executed C++ phase/resource guard, structured out-of-tree compiler integration,20/20 audited correctness checks, unsupported-pattern rejection, actual logs and IR.
4. Executed projection placement, shared partial-rotary factoring, explicit-scale handling, opt-in numerical contracts, conditional overflow screening, negative tests and regressions.
5. Stronger-baseline audit, correct alternative comparison, actual CPU timing, separate compilation costs, raw samples and holdout selection.

## Not new, or not established

- GQA/MQA head sharing, latent KV caching, weight absorption, online softmax, FlashAttention, and general calibration/autotuning already exist.
- Existing IREE flags already eliminate the initial head-expansion example. The original pass is useful implementation evidence, not a sufficient novelty claim.
- We have not shown this backend bug was previously unknown, or that no existing issue/fix addresses it. No exhaustive prior-art/issue search or matched upstream build has been completed.
- No GPU runtime evaluation, end-to-end model, bf16/fp16, backward pass, paged mutable cache, arbitrary mask, dynamic-shape performance, certified approximation error, or measured hardware traffic.
- Compiler acceptance of54 GPU binaries is not correctness or speed evidence.
- Synthetic operators with runtime weights are not pretrained serving workloads. Extreme decode ratios compare with unabsorbed history projections, not a hand-optimized MLA baseline.
- Conditional timing-bootstrap intervals omit workload/host uncertainty. A shared CPU environment cannot substantiate broad hardware generalization.

## Evaluation boundaries

Broader validation requires independently reproduced upstream builds, additional
GPU architectures and precisions, real model modules, and matched fused or
manually absorbed baselines. Approximation/error budgets, dynamic shapes,
compile/calibration amortization and serving workloads are outside this study.
A pinned backend repair and known projection identities do not establish a
general scheduling contribution or global research novelty.

## Primary literature checked

- Milakov & Gimelshein, Online normalizer calculation for softmax (2018): https://arxiv.org/abs/1805.02867
- Dao et al., FlashAttention (2022): https://arxiv.org/abs/2205.14135
- DeepSeek-AI, DeepSeek-V2 (2024): https://arxiv.org/abs/2405.04434
- Geens & Verhelst, Hardware-Centric Analysis of DeepSeek's Multi-Head Latent Attention (2025): https://arxiv.org/abs/2506.02523
- FlexAttention, official MLSys2025 proceedings: https://proceedings.mlsys.org/paper_files/paper/2025
- Flashlight, official MLSys2026 proceedings: https://proceedings.mlsys.org/paper_files/paper/2026
- MLIR Linalg: https://mlir.llvm.org/docs/Dialects/Linalg/
- IREE source: https://github.com/iree-org/iree

The supplied paper's identities and counterexample are explained explicitly. Related-work summaries are scoped; no claim that projection absorption is novel.
