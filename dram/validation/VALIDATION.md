# Validation record

The fixed suite passes24 automated unit tests,18 actual CPU numerical checks
over three seeds and three synthetic fixtures, and six CPU Runner checks.
All nine sm_75 variants compile; repaired model codegen retains f64 operations.
The excluded single-fork ownership probe changes caller K/V arguments under both
variants despite numerical agreement and remains a negative finding.

Final T4 range collection completed with51 valid trackA and61 valid trackB
profiles. Exported counter totals were independently re-parsed for all112
profiles. GPU correctness, raw timing samples, counter exports and environment
records are under [results/dram](../../results/dram) from this document's directory.

Unit coverage includes counter units and missing/duplicate/empty-data rejection,
forced-winner eligibility, exact old-prefix preservation, profile orchestration,
completed-capture recovery and separate logical memcpy reporting. Hardware
counter totals and unprofiled execution latency are distinct measurements.
