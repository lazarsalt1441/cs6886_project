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

## Colab: first run

Keep the existing /content/attention_gpu project and working environment. Upload Attention_GPU_Followup.zip, then extract it over the project (no deleting/re-extracting the old project). Run:

```python
import zipfile, subprocess, os
from google.colab import files
uploaded = files.upload()  # choose Attention_GPU_Followup.zip
with zipfile.ZipFile("Attention_GPU_Followup.zip") as z:
    z.extractall("/content/attention_gpu")
os.chdir("/content/attention_gpu")
env = os.environ.copy()
env["PATH"] = "/content/attention_env/bin:" + env["PATH"]
result = subprocess.run([
    "/content/attention_env/bin/python", "run_gpu_followup.py",
    "--mode", "probe", "--arch", "sm_75", "--seeds", "10", "--trials", "101"
], env=env)
print("Exit:", result.returncode)
files.download("/content/attention_gpu/gpu_followup_probe.tar.gz")
```

The probe is nine compilations rather than the entire 146-variant sweep. Expected native failures are retained as findings; the command fails if the original baseline or decomposed fallback fails. Inspect probe.json for all statuses, including errors in the standalone probes. Send back gpu_followup_probe.tar.gz. Do not loosen numerical tolerances.

## Second independent measurements

After the probe, repeat the same subprocess command with `--mode benchmarks`. This runs cache/block/winner with calibration + holdout timing, eager CUDA PyTorch, and ten-seed validation. Download gpu_followup_benchmarks.tar.gz. This leaves chains out because they showed little GPU benefit; native projection diagnosis comes first. Avoid concurrent runs in the same project. Old JSON may be copied into artifacts; commands.json identifies the new commands.

## Remaining research work

1. Use the isolated T4 probe to establish whether Q/K width, V width, or another property triggers the native failure; obtain pass-by-pass IR before patching upstream code.
2. Validate the decomposed fallback on T4, then integrate a guarded lowering or fix the native schedule. Do not claim root-cause repair from fallback routing.
3. Repeat cache/winner timing with independent inputs/session and eager CUDA reference. Small <5% differences remain inconclusive.
4. Measure device-only execution separately and profile launches/copies before claiming hardware traffic reduction.
5. Export natural model cache-update/branch graphs and compare optimized IREE/PyTorch on end-to-end blocks. Establish prior-art position before a publication claim.
