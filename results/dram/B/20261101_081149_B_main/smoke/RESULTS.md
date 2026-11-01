# Attention DRAM study results

Valid profiles: 7; retained failed profiles: 0.

| Case | Cache conditioning | Replay | Baseline MB | Optimized MB | DRAM reduction |
|---|---|---|---:|---:|---:|
| B_n2048_p0 | warm | app-range | 2415.256 | 2315.637 | 4.12% |
| cache_large | warm | app-range | 182.919 | 0.109 | 99.94% |

Only rows with completed before/after validation and unchanged inputs are included.
A negative reduction is increased traffic. No speedup is inferred from bytes alone.
See summary.json and latency/*/latency.json for unprofiled latency and all backend eligibility reports.
