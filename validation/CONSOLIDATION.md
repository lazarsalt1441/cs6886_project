# Reproducibility validation

The original experiment records and source snapshots are preserved. Repository validation ran the fixed 24-test DRAM unit suite and actual IREE CPU compilation/execution: 18 numerical checks and 6 Runner checks passed. The separate single-fork ownership probe remains ineligible because stock lowering changes arguments; it is not timed as a successful optimization.

Raw exported Nsight CSVs were independently parsed for all 112 final A/B profiles and compared with the saved byte totals. The aggregate generator reproduced the final model latency medians from raw per-repetition timing records. Archive extraction verifies part and complete-archive SHA256, restores original paths, and refuses to replace a modified restored file. Interrupted archive assembly is rebuilt through a temporary file.

C++ build and structural execution evidence, CPU checks, offline CUDA compilation and GPU results are associated with their original environment records. Repository validation does not constitute a new GPU run or a full source-compiler rebuild. The added registration of StableTrig in the standalone drivers uses the same previously compiled implementation and registration pattern.

Weights, SDKs, VMFBs and profiler report binaries are excluded from Git. SOURCE_MANIFEST.json identifies the committed source/evidence bytes; historical package manifests describe their historical releases, not this consolidated layout. Reproduction may need Internet for fresh SDK, package and checkpoint download; reanalysis needs no network.
