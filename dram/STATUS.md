# Attention DRAM study status

Completed: isolated profiling worker, owner-context selection, process-wide NVTX
range, app-range counters, separate kernel attribution, Nsight Systems copy
trace, randomized unprofiled latency, all-backend forced-winner gates, strict
counter parsing, plots, notebooks, automatic partial/final archival and Drive
backup. Assets and compiler configurations are pinned and hash checked.

Local validation: 17 unit tests; 18 CPU numerical executions; six CPU Runner
checks; nine sm_75 compilations. See validation/ for records. The single-fork
CPU control fails input immutability and is excluded; the dense-demand control
passes. No fresh GPU traffic or latency result exists in this kit.

Next: run Colab track B main; optionally run track A on a second account. Send
the entire archive, including failures. Evaluate paired DRAM bytes and separate
latency before claiming memory traffic reduction or inference improvement.

Export compatibility fix: the user's T4 streaming application-range capture
succeeded, but CSV export failed on obsolete --units. Replaced it with
--print-units, added long/wide CSV support and safe reuse of completed captures.
24 unit tests now pass, including export-command and recovery regressions.
Full-model range collection remains pending. No model setup or IR changed.

Second returned export diagnosis: Nsight 2025.3.1 rejects --print-metric-name on the raw page. Removed that details-only option; raw CSV already names metric columns. The 24-test suite now explicitly rejects both obsolete --units and raw-page --print-metric-name. Capture and numerical computation are unchanged.
