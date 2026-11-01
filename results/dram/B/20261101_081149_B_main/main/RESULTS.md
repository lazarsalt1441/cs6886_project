# Attention DRAM study results

Valid profiles: 61; retained failed profiles: 0.

| Case | Cache conditioning | Replay | Baseline MB | Optimized MB | DRAM reduction |
|---|---|---|---:|---:|---:|
| B_n2048_p0 | warm | app-range | 2417.039 | 2322.571 | 3.91% |
| B_n2048_p0 | warm | application | 1864.251 | 1836.541 | 1.49% |
| B_n3072_p0 | warm | app-range | 3093.272 | 3131.547 | -1.24% |
| B_n4096_p0 | warm | app-range | 3880.712 | 3870.097 | 0.27% |
| cache_dense | warm | app-range | 0.285 | 0.275 | 3.71% |
| cache_large | warm | app-range | 178.897 | 0.137 | 99.92% |
| cache_large | warm | application | 0.226 | 0.137 | 39.52% |
| cache_small | warm | app-range | 5.831 | 0.128 | 97.80% |

Only rows with completed before/after validation and unchanged inputs are included.
A negative reduction is increased traffic. No speedup is inferred from bytes alone.
See summary.json and latency/*/latency.json for unprofiled latency and all backend eligibility reports.
