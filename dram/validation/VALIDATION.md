# Validation before delivery

17 automated unit tests pass. Actual IREE CPU compilation and execution passed 18
numerical checks over three seeds and three synthetic fixtures, plus six tests
of the same fixture/Runner interface used by the profiler worker. All nine
sm_75 variants compile; repaired model codegen retains f64 operations.

The excluded single-fork CPU ownership probe agreed numerically but changed K/V
arguments 1 and 2, for both variants. This is retained as a negative finding.

GPU execution, application-range context capture, actual counter collection,
Nsight Systems export, and the new full-model Torch latency comparison have NOT
been run in the authoring environment. The user must execute the Colab smoke
checks and full suite. No new GPU bytes, latency, or speedup is claimed here.

Unit tests include counter units, missing/duplicate/empty data rejection, forced
winner eligibility for all Torch backends, exact old-prefix preservation, profile
orchestration, and separate logical memcpy reporting.
