# Evaluation reanalysis

No new GPU execution. Intervals describe the recorded session only.

| Context | Speedup | Within-session 95% interval | Paired-repeat range | Without first repetition |
|---:|---:|---:|---:|---:|
| 2048 | 1.360x | 1.346–1.363 | 1.343–1.374 | 1.359x |
| 3072 | 1.049x | 1.037–1.058 | 1.034–1.084 | 1.047x |
| 4096 | 1.058x | 1.056–1.064 | 1.052–1.065 | 1.057x |

Bootstrap: 10,000 draws (seed 6886), resample the three prompts, then paired repetition medians within each selected prompt. The statistic is baseline median / optimized median. The 31 calls within a repetition are not independent replications. Three fixed prompts and one session cannot support an across-session confidence claim.

* 2048 improvement appears in all nine paired repetition medians and persists when repetition zero is excluded. This does not explain the kernel-level cause.
* Only 3.91% less whole-step DRAM at 2048 accompanies 26.46% less latency; bytes alone cannot explain the gap.
* Dense control changes direction across repetitions. Equal canonical IR/codegen narrows the cause to runtime/measurement or other unexamined effects; no pass regression mechanism established.
* Additional contexts and same-binary placebo tests require a new GPU run.
* Dense session A: paired ratios 0.178, 3.396, 0.413.
* Dense session B: paired ratios 0.207, 2.293, 0.831.

Canonical dense input equal: True; all retained fresh codegen files equal: True. Full VMFB hashes differ, so bitwise whole-executable equivalence is not asserted.
