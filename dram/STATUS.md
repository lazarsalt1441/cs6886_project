# DRAM study: completed evaluation

The implementation includes owner-context selection, NVTX application ranges,
range counters, separate kernel attribution, Nsight Systems copy tracing,
randomized unprofiled latency, forced-winner gates, input-immutability checks,
strict counter parsing, plots and periodic result backups.

Validation: 24 unit tests, 18 synthetic CPU numerical checks, six CPU Runner
checks and nine sm_75 compilations. Completed final GPU datasets contain 51 valid
profiles in track A and 61 in track B, with raw CSV exports and unprofiled timing
samples. All 112 exported profile totals were independently recomputed.

Track B IREE latency reductions at contexts2048/3072/4096 are 26.46%, 4.64%
and 5.52%. Full-model total DRAM changes are -3.91%, +1.24% and -0.27%,
respectively. Thus full-model traffic reduction is not consistent across contexts.
The sparse synthetic large-cache fixture shows about 99.92% lower IREE traffic.

The single-fork ownership probe changes caller K/V inputs under stock lowering
and remains excluded from performance claims. The dense-demand no-match control
and clock variation limit interpretation of small timing gains. Counter/replay
durations remain separate from ordinary latency.

Nsight2025.3.1 compatibility uses `--print-units base`; the details-only
`--print-metric-name` is omitted from raw CSV export. Both long and wide layouts
are supported, and completed captures can be re-exported without recapture.
