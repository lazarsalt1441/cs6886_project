# Returned T4 smoke diagnosis

Archive attention_dram_20261101_074854_B_main.tar.gz contains a completed, valid
Torch streaming application-range capture and profile.ncu-rep. GPU before/after
checks pass. The export command failed with:

    ==ERROR== unrecognised option '--units'.

Fix: --print-units base, explicit metric names, support both long and wide CSV
layouts, preserve completed reports on export retry, and check export before
expensive model setup. 24 local unit tests pass, including the command spelling,
wide CSV units and failed-export recovery without another capture. Original
CPU numerical and CUDA compilation validation records remain unchanged.

The corrected command still requires execution on the user's profiler. No DRAM
byte total or full-model GPU result is inferred from the failed CSV export.

Second returned export diagnosis: Nsight 2025.3.1 rejects --print-metric-name on the raw page. Removed that details-only option; raw CSV already names metric columns. The 24-test suite now explicitly rejects both obsolete --units and raw-page --print-metric-name. Capture and numerical computation are unchanged.
