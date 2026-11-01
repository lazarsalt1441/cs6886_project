# Attention DRAM traffic evaluation

This study measures the C++ MLIR cache transformations using hardware counters
and separately timed IREE baseline/optimized execution. Completed datasets are
under `../results/dram/A` and `../results/dram/B`. The root README summarizes
the latency improvements, DRAM tradeoffs and experimental limitations.

TrackB evaluates winner-commit factoring in a custom two-candidate pretrained
step. TrackA is ordinary decoding with the same numerical repair and no matching
commit rewrite. Both tracks include synthetic mechanism and no-match controls.
Run separate sessions for independent measurements; avoid concurrent workloads
on one GPU.

The model is FP32 SmolLM2-135M-Instruct, revision
`12fd25f77366fa6b3b4b768ec3050bf629380bac`. The workload is not a complete
Hugging Face beam-search or speculative-generation implementation.

## Colab reproduction

Select a CUDA GPU runtime with Torch2.11 and Nsight Compute counter permission.
The notebooks `Colab_DRAM_A.ipynb` and `Colab_DRAM_B.ipynb` contain these steps.

Cell1:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Cell2:

```python
from pathlib import Path
import subprocess, sys
repo = Path('/content/cs6886_project')
if not (repo / '.git').is_dir():
    subprocess.run(['git', 'clone', 'https://github.com/lazarsalt1441/cs6886_project.git', str(repo)], check=True)
subprocess.run([sys.executable, str(repo / 'scripts/unpack_artifacts.py')], check=True)
```

Cell3:

```python
TRACK = 'B'                 # 'A' for the ordinary-decode control
PLAN = 'main'               # Main runs smoke first; extended adds cases
DRAM_KIT_ROOT = str(repo / 'dram')
exec(Path(DRAM_KIT_ROOT, 'COLAB_RUN.py').read_text())
```

The runner checks pinned dependencies in a subprocess and rejects mismatched
Torch for matched replication. It downloads and verifies the pinned checkpoint
if needed; a verified existing model can be reused. Model preparation, CPU
references and compilation occur outside the measured range. Runtime depends
on setup and the profile plan; an extended sweep may take hours.

Partial backups occur every five minutes and on completion/interruption under
`MyDrive/Attention_DRAM_Results`. A browser download is attempted at completion;
browser permissions or abrupt runtime termination can prevent it. Colab session
lifetime is controlled by the service. Preserve the archive and logs on failure.
Set `DRAM_RUN_ID` only to intentionally resume an existing run; an independent
session needs a new ID.

Nsight2025.3.1 raw export uses `--print-units base`, without the details-only
`--print-metric-name`. A streaming preflight verifies capture/export before model
setup, and matching completed captures can be re-exported without recapture.

## Displayed IREE comparisons

Default plots and regenerated model summary tables show IREE variants only.
Raw experiment records retain all originally collected backends and numerical
references; the collection protocol below continues to describe those full runs.

| Test | Purpose | Comparators |
|---|---|---|
| Random 64 MiB streaming kernel | Check the marked window and byte units; no initialization in range | Known one read and one write per element |
| Small forked cache | Measure the conditional read-forwarding mechanism | Strong IREE, C++ forwarded IR |
| Large forked cache | Test whether avoided history copies actually reduce DRAM | Same two IREE comparators |
| Dense-demand control | Test a case where the forwarding cost gate does not fire | Same comparators; zero forwarding markers |
| Full model B | Measure winner-commit factoring on real weights | Repaired strong IREE, repaired C++ commit IR |
| Full model A | No-match control for ordinary decode | Repaired strong IREE; no optimized variant manufactured |
| Kernel diagnostic | Attribute traffic to kernels and inspect L2 request sectors | Separate application replay report |
| Nsight Systems trace | Attribute CUDA copies and kernel launches within the window | Same isolated call window, if nsys is installed |

Main model correctness and latency cover contexts **2048, 3072, 4096**, three
prompts each. All Torch and IREE backends must pass natural and both forced
winners in B. Full logits, exact old-cache prefixes and appended cache rows are
checked before and after measurement; inputs must remain unchanged. Model
tolerance stays atol=5e-4, rtol=3e-3. Synthetic reference is independent FP64 NumPy
attention, with atol=5e-5, rtol=2e-3.

Main counter collection uses prompt 0 at each context and three separate profile
processes per backend. Each range has three calls with the same immutable inputs;
reported bytes are divided by three. These are independent one-step calls, not
three sequential generated tokens. Latency uses all prompts, three repetitions
of 31 samples, randomized backend order and resident GPU outputs. Compilation and
first-use setup are excluded. No profiler is attached during latency measurement.

Extended additionally profiles all prompts, includes contexts 128 and 512, and
adds a one-call eviction-conditioned experiment. The latter touches 128 MiB
outside the range; it is a conditioning sensitivity test, not proof of a perfectly
cold GPU. Main is the recommended first study.

## Why the profiling setup matters

The primary Nsight command uses `--replay-mode app-range --cache-control none`
with a process-wide NVTX start/end range. This preserves the application launch
order and concurrency better than per-kernel replay. Model loading, weight
upload, CPU reference, validation, warmup and optional eviction happen outside
the marker. The worker discovers the actual CUDA context owning its input and
makes it current during the range, which matters when mixing IREE and Torch.

No `--launch-count 1` is used for the model: that would profile only a fragment
in kernel mode. A one-action result is expected for an application-range report,
because that action represents the whole marked range. A separate kernel report
contains many actions and is never pooled with the range total.

The marked range includes completion synchronization but not host downloads. The
raw hardware counters are device traffic during that window. Other GPU work can
contaminate global DRAM counters, so run alone. Dirty cache-line writebacks can
fall beyond the window; synchronization does not flush L2. In particular, the
streaming control need not report exactly its analytical payload. Three repeated
calls, repeated processes, a large random buffer and the conditioning experiment
help expose boundary/cache effects rather than hiding them.

`--clock-control none` avoids requesting privileged clock changes. GPU clock,
temperature and power telemetry are recorded around latency repetitions. The
clock warning is expected; do not use profiler-instrumented duration as ordinary
latency or interpret small timing differences as robust gains.

Nsight Systems records **logical CUDA memcpy payload bytes**, not physical DRAM
bytes. It can reveal copy-engine activity that per-kernel Nsight Compute sums
miss. Do not add memcpy payload bytes to range DRAM counters: that can double
count transfers and ignores cache effects. If nsys is absent, the suite records
that limitation rather than inventing zero copies.

## Implementation and validation

The kit includes hash-checked before/after IR, the C++ cache/trig source, original
model adapters, compiler flags, numerical gates, raw CSV parsing, compilation
records, randomized latency sampling, resumable profiles and result packaging.
Prepared model IR comes from the previously compiled C++ passes; it is not a new
handwritten approximation. Python manages experiments, not the compiler rewrite.
`iree_commit` is B's commit pass; `iree_forwarded` is the synthetic read pass.

Validation records include24 unit tests,18 CPU numerical checks, six CPU
Runner checks and nine sm_75 compilations. The final GPU runs contain51 valid
trackA profiles and61 valid trackB profiles. All112 exported CSV totals were
independently recomputed. A streaming-range preflight and IREE-range preflight
stop a sweep if capture fails; a kernel-only diagnostic is not promoted to a
whole-step result.

The single-fork fixture is kept as a separate local ownership probe, not a
performance control. Local CPU lowering agreed numerically but changed caller
K/V input buffers. It is excluded from the default measured cases; the dense
two-fork fixture is the validated no-rewrite control. This is an observed
contract limitation for that fixture, not proof of a universal upstream bug.

## Recorded outputs

Each result archive preserves the following records, including failures:

* `summary.json`, `dram_summary.csv`, `RESULTS.md` and `plan.json`;
* `environment.json`, `compilation.json` and `completion.json`;
* `profiles/*/metrics.csv`, `profile.ncu-rep`, `worker.json`, `task.json`, commands
  and `ncu.log`;
* `latency/*/latency.json`, which includes forced-winner eligibility, input
  checks, raw samples and telemetry;
* `nsys/*/trace.nsys-rep`, SQLite exports, copy summaries and logs when available.

Cache/weights and input fixture files are excluded from the archive to avoid
gigabytes of redundant model data. Raw report files in a
periodic partial backup are included only after their collection record exists.

The useful outcome is a paired table of **DRAM read bytes, write bytes, total
bytes, and unprofiled latency** for the same correct workload. Traffic reductions are evaluated alongside regressions and no-match controls. Allocation
reduction alone remains insufficient; persistent KV compression, serving
throughput, mixed precision and global novelty are not established by this kit.

## Command line and local checks

```bash
python -m unittest discover -s tests -v
python tests/cpu_integration.py --cache ./cpu_test_cache
python study.py --track B --plan smoke --output ./results_B_smoke
python study.py --track B --plan main --output ./results_B_main
python analyze.py ./results_B_main
```

The compiler/runtime must be IREE 3.13.0 for replication. The experiment applies
O3, generalize-matmul and aggressive reshape movement to both IREE variants, and
disables f64 input demotion and numeric precision reduction equally. CUDA split
argmax flags match the repaired model study. All compile commands are saved.

Completed profiles are reused only when their task, fixture, compiler build and
stable environment identity match. For a new independent session, choose a new
run ID. Do not count resumed cached profiles as fresh repetitions.

## Primary profiler documentation

* https://docs.nvidia.com/nsight-compute/ProfilingGuide/ — application-range
  replay, context requirements, cache state, global traffic and measurement limits.
* https://docs.nvidia.com/nsight-compute/NsightComputeCli/ — NVTX filtering,
  metrics, replay, CSV export and cache/clock control.
* https://docs.nvidia.com/nsight-systems/UserGuide/ — NVTX capture and trace export.

The recorded profiler is Nsight Compute 2025.3.1 on a T4. The suite saves
actual tool versions and does not require upgrading the GPU or driver.
