# Bounded trig argument-reduction repair

This extension preserves the computed FP32 phase while using bounded FP64 argument reduction for supported sine/cosine kernels. Natural and both forced-winner cases use the same fixed correctness gate. The historical repair/fusion ablation and final DRAM evaluation retain separate run identities.

## Reproduction

This directory records the bounded trig repair and the original repair/fusion
ablation. The final controlled evaluation is implemented in [dram/](../../../dram); use the root README or its Colab notebooks
for the submitted IREE comparison. Earlier archived scripts retain the original
ablation protocol and its failures.

For that earlier protocol, run from `real_model/` after restoring artifacts,
installing dependencies and downloading the pinned checkpoint:

```bash
python experiments/stable_trig/run_repair.py --project . --set B --device cuda
```

The wrapper saves separate repair-results directories, validates natural and
forced-winner outputs and keeps the numerical tolerances fixed. The companion
RUN_REPAIR_CELL.txt is a legacy attachment-based bootstrap; current notebooks
clone this repository directly. Earlier and final runs must retain separate
identities and should not be pooled as one controlled experiment.

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

The last variant adds only --iree-dispatch-creation-enable-aggressive-fusion. It is an experimental performance option, not a confirmed improvement. Compare repair+cache against repaired baseline to isolate the cache pass. Compare repair+cache+fusion against repair+cache to isolate fusion. Do not compare a combined variant only against the weakest baseline and attribute all gains to one pass. The presented performance comparison is within IREE; Torch supplies the numerical reference.

At every natural case, both cache winners are additionally forced for GPU correctness using separated log priors. Forced cases are never timed. Normal timings use natural priors. New forced-winner checks are reported independently; a backend must pass both forced winners as well as the natural case before being timed. Each passing backend is timed with randomized backend order,31trials x3repetitions and5warmups; full output and input immutability are rechecked afterward. Only final valid=true latency rows support claims. Reference-failing cases remain in validation.json and are not timed. No TTFT or complete-generation claim is made.

## Local evidence and limits

Real C++ pass compiled with MLIR19.1.1. Structural checks cover two rewrite sites, idempotence, lowering-config rejection, f64 rejection and large-kernel rejection. Twenty compiled random-angle seeds compare against float64 NumPy trig; the repair gives errors on the order1e-7 rather than1e-4. Small/out-of-domain finite fallback is checked for exact output bits. Source PTX emulation is additional explanatory evidence, not a substitute for compiled tests.

The CPU full pretrained model reproduces the observed2048/prompt1 failure and the repair passes unchanged tolerance, reducing maximum discrepancy from about0.000762 to0.0000414. Further CPU checks and exact logs are included. All four repaired sm_75 variants compile; GPU numerical execution and speed remain pending. Existing unmodified variants retain the original tests/results.

## Rebuild source

Compiler changes are in source/StableTrig.cpp plus the supplied CMakeLists and registration main. Copy them into the original compiler tree, build normally, then run prepare_repair.py from research_next/repair. The supplied prepared/*.mlir avoid any C++ build on Colab. test_trig.py executes actual before/after compiled kernels; validate_model.py reproduces CPU model cases; run_repair.py is the GPU driver. Root determination in reproduction scripts assumes this directory is research_next/repair under the original project.
