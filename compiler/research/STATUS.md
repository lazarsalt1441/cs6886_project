# Research continuation status — 22 October 2026

## R0 — stronger baseline and novelty audit, committed

IREE O3 + matmul generalization + aggressive reshape movement removes the original head-copy expansion. Workspace8192 bytes, same as the original custom pass. Original measured data is retained with a prominent correction; that example alone is not novel.

## R1 — projection placement, committed

Compiled C++ reassociation pass, static f32 shape/use/body checks, explicit contract, MAC filter, key/value placement, native scaling, partial rotary subspace and shared-source concatenation. Weight absorption is prior work.16 decomposed and7 native cases checked, plus4 partial-rotary cases. Native SDPA exports are genuine native attention; earlier decomposed-SDPA trial is retained under its corrected filename.

## R2 — independent semantic bug and guard, committed

20 original native CPU shape/alias cases;14 fail reference checks, independent of projection rewriting. Wrong partition simulated within3e-7. Real C++ configured-IR feature guard plus legal-tile resource adjustment. Naive feature untile and separate operand-bound approaches failed vector constraints; logs retained. Optional caller-bound range screen rejects an all-finite overflow counterexample. Standard-Linalg fallback implemented.

## R3 — final validation and timing, complete

- Guard20/20 correct; max abs9.267186557859741e-7. Final stricter C++ guard reproduces all20 measured lowering configurations.
- Seed-free analytic output0.599999964 original,0.899999976 repaired; correct real0.9.
-36 structural tests pass (9 original,16 projection,11 guard). Explicit positive/negative/zero scale across3 alternatives:9 additional numerical executions pass.
- Final timing27 groups,21 observations each, randomized order, CPU0,5warmups, no output download.16 decomposed geomean3.928x;4 rotary11.404x;7 frozen portfolio15.835x vs correct native and4.819x vs correct materialized. Raw samples/hashes/errors retained.
- Separate final timing confirms6 ablations and3 correct fallback comparisons. Forced long-query absorption regresses; corrected fused can be much slower than materialization.
-54/54 offline CUDA sm_80 binaries compile. LLVM/PTX inventory retained; kernel counts sometimes increase. No GPU execution or hardware traffic result.
-9 explicit-scale numerical executions pass. GPU runner supports optional eager CUDA PyTorch, but that GPU-only branch remains unexecuted.

## R4 — manuscript, reproduction, handoff

Final PDF compiled twice, all10 pages rendered and visually inspected, no overfull boxes or undefined references. Final integrity checks verified68 primary VMFB hashes and no empty IR/VMFB artifacts. Paper and generated tables/plots are in `paper/`; `make_paper.py` regenerates figures/macros from actual JSON. Research guide, upstream issue draft (not posted), claim audit and future evaluation checklist are supplied. Root README/REPORT carry the stronger-baseline correction. SDK/build/vendor are reconstructed, not packaged.

Publication status: research manuscript draft, not submitted/accepted or proven NeurIPS-level. Missing GPU execution, matched upstream compiler build, independent reproduction, complete models, lower precision, broader numerical contracts and general scheduling contribution. Do not promote known absorption algebra as new.
