# GPU follow-up: correctness and implementation selection

31 October 2026

## What changed

* Added an independent native-attention width probe (NumPy float64 references, fresh inputs per invocation, random and tail-only features). It isolates query/key width from value width.
* Added a decomposed projection fallback for native cases where equivalent saved decomposed IR exists. The original native binary remains compiled and tested; failures remain visible.
* Added 21 calibration observations, independent of the subsequent 101 holdout observations, and a 5% minimum measured improvement for selecting a candidate over the baseline. This is empirical per-shape selection, not a general compiler cost model or a new kernel. It is tuned from this study and requires independent validation/repeatability testing.
* Cache compatibility preference now tries the conditional form before the masked form. This changes preference only; the calibrated selection is reported separately and still requires the multi-seed correctness audit before deployment.
* Added optional eager CUDA PyTorch comparison in the follow-up cache/block/winner run using the existing CUDA-enabled Torch installation.

## Executed locally

IREE compiler/runtime 3.13.0, LLVM 24 build; CPU local-sync, 10 seeds per case, 21 host timing observations. No local GPU.

Native probes (20 checks each: random + only last feature active, 10 seeds):

| Q/K width | V width | CPU status |
|---|---|---|
|64|64|PASS|
|128|128|PASS|
|256|128|FAIL|
|512|128|FAIL|
|128|512|PASS|
|512|512|FAIL|

Large projected case h8_m1_n2048_l512_d128_v128_s1, identical inputs and reference:

* Native original: 10/10 PASS, worst max-absolute error 3.71e-7.
* Native reassociated: fails; worst max-absolute error 0.196.
* Decomposed reassociated fallback: 10/10 PASS, worst max-absolute error 3.70e-7.

All nine diagnostic/candidate programs compiled to sm_75 CUDA binaries. Six selection tests passed. Python compilation checks passed. These are compilation and CPU evidence, not new GPU runtime measurements. CPU wide-head failure was already known in the project; its relation to the T4 failure remains a hypothesis until the isolated T4 probe executes.

## Interpreting the first T4 run

144/145 compiled variants passed ten-seed validation. A further irregular masked form did not compile. Cache forwarding: 1.23x geometric mean across nine shapes, best 3.44x. Winner factoring: 1.57x. Update chains: only 1.01–1.07x; do not headline. Valid projection cases: 1.24x geometric mean, median 1.06x, best 6.49x; one incorrect native case excluded. These are invocation + execution + output-download timings, not kernel-only timings. Allocation counters are invocation requests, not measured DRAM traffic.

## Reproduction

Restore artifacts from the repository root and configure the CUDA environment
described in the root README. From `compiler/`, run:

```bash
python run_gpu_followup.py --mode probe --arch sm_75 --seeds 10 --trials 101
python run_gpu_followup.py --mode benchmarks --arch sm_75 --seeds 10 --trials 101
```

The probe runs nine compilations rather than the full variant sweep. Expected
native failures remain recorded; baseline or decomposed-fallback failures cause
a nonzero exit. Each run writes a result archive with commands, errors,
validation and raw samples. Recorded GPU outputs are under `../results`.

## Interpretation

Fallback routing is distinct from repairing the native schedule. Calibration
and holdout measurements are per-shape observations, not a general compiler cost
model. Timing includes synchronized invocation and output download and must not
be equated with kernel-only latency or DRAM bytes. Small timing differences,
clock variation and no-match controls limit performance claims. The final
pretrained-model/DRAM evaluation uses the separate `../dram` protocol.
