# Version2: retain Torch baselines after forced-winner validation

GPU repair result: all9 natural cases and18 forced-winner checks per repaired B variant passed. The first repair runner mistakenly excluded Torch baselines from SetB timing because forced-winner reports were produced only for IREE. This version validates both forced winners for Torch eager, SDPA and compiled Torch before applying the same eligibility gate. Original raw IREE timings remain usable; prior B repair results cannot establish a same-run Torch performance comparison.

The updated cell tests contexts2048/4096 to shorten the control rerun; all three prompt templates remain. Use the updated RUN_REPAIR_CELL.txt in the existing SET=B Colab notebook; SetA need not be rerun just to fix this gate. Version2 writes repair_results_v2_A/B and preserves old result directories. Return repair_results_v2_B.tar.gz. All tolerance/IR/performance options remain unchanged. Torch compile failures stay explicitly unavailable. If no Torch compilation succeeds, there will still be no compiled-Torch timing; review errors and TIMED BACKENDS.

--- Original repair instructions ---

# Numerical repair and controlled performance experiment

This is a new extension to the existing real-LLM project; original modules remain intact. It does not loosen tolerance or claim an unmeasured GPU speedup.

## Run

Download Attention_Repair_Tests.zip and RUN_REPAIR_CELL.txt. Paste the text cell below your existing Colab cells. Set SET='A' for the ordinary decoder account, SET='B' for the lookahead account. Upload the REPAIR ZIP when prompted. Keep original project/environment/Drive mount. If the runtime was reset, run original upload/environment/Drive cells first; the extension downloads the checkpoint and regenerates parameters if missing. Do not rerun the old benchmark.

The extension compiles fresh variants and tests contexts2048/3072/4096 with prompts0/1/2. It saves output under repair_results_v2_A/B; original results are not overwritten. Backups every5minutes go to the already-mounted Drive, then final archive download is attempted. A hard runtime kill may prevent final callbacks. Return repair_results_v2_B.tar.gz, including failures. This run may take longer because it includes more variants and both-winner correctness tests; compilation is excluded from timing.

## Concrete finding and fix

The original generated PTX uses f32 quadrant reduction for sin/cos: floor(x*(2/pi)), then x-q*float(pi/2), followed by a polynomial. Rounding/cancellation grows with the original phase. At actual long-position RoPE angles, emulating the emitted PTX gives sin/cos errors around1e-4. This is a reproduced compiler numerical discrepancy under our fixed reference criterion; it is not yet a conclusion about IREE's documented mathematical error contract.

The C++ attention-stable-trig pass performs bounded argument reduction in f64, then applies the existing f32 sin/cos to the small remainder. It only matches f32 trig inside static unencoded linalg kernels with at most4096 elements. A runtime ordered range guard selects the repair only for 32<abs(x)<=8192; small and out-of-domain values keep the original operation. Markers prevent recursive rewriting. Lowered kernels, f64 operations and large tensors are rejected.

The already-computed f32 phase is preserved. We do not recompute the position/frequency product in higher precision or change attention/cache semantics. Only argument reduction is widened. Crucially IREE initially demoted the repair back to f32. The isolated flag ablation shows --iree-input-demote-f64-to-f32=false is sufficient to retain the accuracy gain; disabling numeric precision reduction alone is not. Both repair and control variants in this experiment use these fixed protective settings:

--iree-input-demote-f64-to-f32=false
--iree-opt-numeric-precision-reduction=false

All variants in this extension receive these SAME precision settings. These settings alone do not implement the repair; the prepared IR contains actual C++-generated f64 reduction operations. Offline repaired PTX is checked for f64 instructions. The original public weight checkpoint, dtypef32, toleranceatol5e-4/rtol3e-3 and strong baseline flags remain fixed.

## Variant matrix

A: original baseline, repaired baseline.
B: original baseline, original cache pass, repaired baseline, repair+cache pass, repair+cache pass+aggressive fusion.

The last variant adds only --iree-dispatch-creation-enable-aggressive-fusion. It is an experimental performance option, not a confirmed improvement. Compare repair+cache against repaired baseline to isolate the cache pass. Compare repair+cache+fusion against repair+cache to isolate fusion. Do not compare a combined variant only against the weakest baseline and attribute all gains to one pass. Torch eager, SDPA and optional Torch compile remain reference performance baselines.

At every natural case, both cache winners are additionally forced for GPU correctness using separated log priors. Forced cases are never timed. Normal timings use natural priors. New forced-winner checks are reported independently; a backend must pass both forced winners as well as the natural case before being timed. Each passing backend is timed with randomized backend order,31trials x3repetitions and5warmups; full output and input immutability are rechecked afterward. Only final valid=true latency rows support claims. Reference-failing cases remain in validation.json and are not timed. No TTFT or complete-generation claim is made.

## Local evidence and limits

Real C++ pass compiled with MLIR19.1.1. Structural checks cover two rewrite sites, idempotence, lowering-config rejection, f64 rejection and large-kernel rejection. Twenty compiled random-angle seeds compare against float64 NumPy trig; the repair gives errors on the order1e-7 rather than1e-4. Small/out-of-domain finite fallback is checked for exact output bits. Source PTX emulation is additional explanatory evidence, not a substitute for compiled tests.

The CPU full pretrained model reproduces the uploaded2048/prompt1 failure and the repair passes unchanged tolerance, reducing maximum discrepancy from about0.000762 to0.0000414. Further CPU checks and exact logs are included. All four repaired sm_75 variants compile; GPU numerical execution and speed remain pending. Existing unmodified variants retain the original tests/results.

## Rebuild source

Compiler changes are in source/StableTrig.cpp plus the supplied CMakeLists and registration main. Copy them into the original compiler tree, build normally, then run prepare_repair.py from research_next/repair. The supplied prepared/*.mlir avoid any C++ build on Colab. test_trig.py executes actual before/after compiled kernels; validate_model.py reproduces CPU model cases; run_repair.py is the GPU driver. Root determination in reproduction scripts assumes this directory is research_next/repair under the original project.
