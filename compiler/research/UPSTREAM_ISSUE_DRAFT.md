# Draft: CPU online attention normalizes partial feature reductions

This issue draft has **not been posted**. It describes the measured binary, not all current IREE builds.

## Reproduction environment

IREE base compiler/runtime 3.13.0, compiler revision `89ece87b89852100c0a4ac3d515e14842d090967`, LLVM24git; Python3.12.14, PyTorch2.14.1+cpu, Turbine3.9.0; AMD EPYC9V74, CPU host target. The separately inspected source checkout is a different revision.

```bash
source scripts/env.sh
python research/analytic_counterexample.py
```

This script exports a plain `scaled_dot_product_attention` with 4D Q/K/V and `enable_gqa=True` to preserve native attention. It does **not** apply projection placement. It compiles the same configured module with and without the C++ schedule guard. Full exported, configured and guarded IR is under `research/results/analytic-counterexample/`.

Shapes: Q=[1,1,1,256], K=V=[1,1,2,256]. Q and K start zero. Set Q[...,0]=sqrt(256)*log(9), K[...,token0,0]=1. V token0 is all ones; token1 all zeros. With standard 1/sqrt(256) scaling, logits are [log(9),0]. The exact output is 0.9 at every output coordinate.

Observed original compiled output: **0.5999999642372131**.
Observed guarded compiled output: **0.8999999761581421**.

## Localization

Configured online-attention IR assigns a feature-reduction tile of 16. The observed output equals normalization over (feature tile, key token), rather than summing all feature contributions before normalization. For sixteen tiles, partial logits are [log(9),0] in one tile and [0,0] in the other fifteen. The resulting erroneous quotient is (9+15)/(9+1+30)=0.6.

A separate random-input simulation reproduces a reduced original output within 3e-7 using this feature-partition interpretation. See `backend-localization.json` and `localize_native_bug.py`. O0/O3 and transposed-value toggles did not repair the independently reproduced mismatch. The evidence supports a feature-completion boundary violation; feature tiling inside a dot-product accumulation remains legal.

The independent audit covers20 shape/alias cases, head widths32/128/256/512/768, with14 original failures. Both shared K/V SSA operands and separate equal-shaped K/V inputs are included. Configured C++ guarding repairs all20 with worst max absolute error9.27e-7 against float64.

## Repair prototype

`src/OnlineAttentionGuard.cpp` infers K1 feature axes from Q/K/V/output indexing maps. It untiled feature reductions at the online update and jointly shrinks legal query/token tiles to bound complete vector contraction volume. Simply untile K1, or bound the operand vectors separately, failed compiler vector-resource checks in other cases. A full upstream solution may instead stage complete logit accumulation with a better resource model.

The tested bridge isolates compatible tensor IR because MLIR19 SDK cannot parse the whole IREE24 HAL module. C++ owns the schedule change. `verify_final_guard.py` checks that the final stricter pass reproduces all20 measured configurations. Unsupported forms are left untouched; this is not a generic verifier for arbitrary attention variants.

The corrected fused CPU schedule can be very slow. A correct standard-Linalg score-materializing fallback is supplied and often wins. Wrong-kernel latency must not be presented as a valid optimization baseline.

## Source clue, not a claim of revision identity

The separately inspected checkout's `compiler/src/iree/compiler/Codegen/LLVMCPU/KernelDispatch.cpp` selects attention reduction tiles around a head-width threshold128. The authoritative evidence is the measured compiler's emitted IR and executable counterexample. Reproduce on a matched source build and current upstream before assigning an upstream regression range.
