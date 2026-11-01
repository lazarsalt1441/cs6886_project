# Attention DRAM study results

Valid profiles: 51; retained failed profiles: 0.

| Case | Cache conditioning | Replay | Baseline MB | Optimized MB | DRAM reduction |
|---|---|---|---:|---:|---:|
| cache_dense | warm | app-range | 0.262 | 0.271 | -3.19% |
| cache_large | warm | app-range | 178.228 | 0.132 | 99.93% |
| cache_large | warm | application | 0.222 | 0.137 | 38.11% |
| cache_small | warm | app-range | 5.837 | 0.120 | 97.95% |

Only rows with completed before/after validation and unchanged inputs are included.
A negative reduction is increased traffic. No speedup is inferred from bytes alone.
See summary.json and latency/*/latency.json for unprofiled latency and all backend eligibility reports.
