# T4 repair results, 2026-11-01

Sources: uploaded repair_results_A.tar.gz and repair_results_B.tar.gz. Torch2.11.0+cu130, TeslaT4/sm_75. All9 cases per set completed; final latency rows have valid=true and inputs are unchanged.

The unmodified IREE baseline passes5/9 natural cases. Repaired baseline passes9/9 in A. In B, repaired baseline, repair+commit and repair+commit+aggressive fusion each pass9/9 natural cases and18/18 forced-winner checks. Maximum natural-case absolute discrepancy across repaired outputs is3.62396240234375e-5, compared with0.0016870498657226562 across unmodified IREE. CPU/GPU Torch reference checks remain successful. Cache-commit and repaired baseline outputs are numerically identical on all9 natural cases.

Median of per-repetition medians, all3 prompts per context, synchronous resident-output full-model step:

|Context|Repaired IREE baseline|Repair+cache commit|Latency reduction|Repair+cache+aggressive fusion|
|---|---:|---:|---:|---:|
|2048|44.52ms|32.02ms|28.1%|33.16ms|
|3072|63.20ms|60.77ms|3.8%|60.40ms|
|4096|82.86ms|78.51ms|5.2%|78.41ms|

Aggressive fusion is3.6% slower at2048, only0.6% faster at3072 and0.1% faster at4096; these small positive changes are not established significant gains. Do not enable it by default on this evidence.

Accuracy repair overhead against unmodified IREE on matched valid prompt/repetition pairs: median time ratio1.0007 in A,1.0025 in B. This suggests little observed overhead on these tests, not proof of zero cost on other models/hardware.

IMPORTANT RUNNER LIMITATION: SetB repair runner's strict forced-winner timing gate had reports only for IREE, which excluded Torch eager, SDPA and compiled Torch from timing. Their natural correctness checks passed but they were never timed or post-timing checked in B. Therefore do not infer a same-run B comparison to Torch from this archive or combine numbers from older runs as a controlled ablation. SetA Torch timing remains present.

Version2 produces forced-winner reports for Torch as well, keeping the gate strict, and writes separate repair_results_v2 directories. CPU regression confirms validated eager/SDPA backends are timed. GPU Inductor timing awaits the B-only rerun. No original result files are overwritten.

Interpretation: actual numerical repair is now supported on T4 and cache-commit acceleration is measured against a numerically valid IREE baseline on all three prompts. Scope remains one f32 pretrained135M model, T4, custom branching single-step inference and repeated prompt templates. Full-sequence generation, representative branching workloads, other dtypes/GPUs, serving benchmarks and novelty remain unestablished.
