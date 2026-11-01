# Attention DRAM traffic study

This kit profiles the saved, genuine C++ MLIR transformations from the CS6886
attention project. It adds hardware traffic evidence and separately timed
IREE baseline/optimized comparisons. Completed GPU results are now preserved in
`../results/dram/A` and `../results/dram/B`; see the repository root README for
the measured improvements, traffic tradeoffs and timing limitations. The ZIP
upload instructions below remain available; the root README also gives a direct
Git checkout workflow using `DRAM_KIT_ROOT`.

The first counter test successfully profiled a Torch initialization kernel. That
establishes access, not an optimization benefit. This study marks only inference.

### CSV export fix for the returned T4 smoke run

The returned run `20261101_074854_B_main` successfully captured the streaming
application range and passed its numerical checks. Export then failed because
the runner used `--units`, which Nsight Compute 2025.3.1 rejects. The corrected
command uses `--print-units base` without the details-only `--print-metric-name` option. The parser accepts both long metric-row
CSV and wide raw CSV with a units row. A completed, matching capture can be
re-exported without running it again. A cheap counter/export check now precedes
model setup. No model or numerical tolerance was changed.

Extract this updated ZIP over the existing `/content/Attention_DRAM_Tests`
directory. Keep cache/ and runs/. For the returned B run, set
`DRAM_RUN_ID = '20261101_074854_B_main'`, `TRACK='B'`, `PLAN='main'` and execute
COLAB_RUN.py again. Matching model fixtures and compiler outputs are reused.
The friend on track A can apply the same update and retain their own run ID.

## What to run

Use **track B** for the main result: the pretrained two-candidate workload with
stable trig reduction, before/after winner-cache commit factoring. A friend can
run **track A**: ordinary decode with the same repair and no matching commit
rewrite, plus independent repeats of the synthetic mechanism tests. Both accounts
may run B instead if cross-session confirmation is the priority. Do not run two
studies simultaneously on one GPU.

The model is pinned SmolLM2-135M-Instruct, revision
`12fd25f77366fa6b3b4b768ec3050bf629380bac`, FP32. Track B is a custom two-candidate
one-step driver, not complete HF beam search or speculative generation.

### Colab cell 1

Select a GPU runtime. Keep the original Torch 2.11 CUDA environment. Mount Drive
for persistent backups; this cell will prompt for authorization if needed.

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Colab cell 2

Download Attention_DRAM_Tests.zip from the chat, then upload it here.

```python
from google.colab import files
from pathlib import Path
import zipfile
uploaded = files.upload()
archives = [Path(n) for n in uploaded if n.lower().endswith('.zip')]
assert len(archives) == 1, 'Upload only Attention_DRAM_Tests.zip'
with zipfile.ZipFile(archives[0]) as z:
    assert all(not Path(n).is_absolute() and '..' not in Path(n).parts
               for n in z.namelist())
    z.extractall('/content')
```

### Colab cell 3

The main plan first runs its own smoke stage, then continues automatically. It
reuses `/content/attention_real_llm/model` if that verified checkpoint is present.
Otherwise it downloads the pinned checkpoint. It installs missing pinned IREE,
Turbine, Transformers and numerical dependencies, leaving Torch unchanged.

```python
TRACK = 'B'        # Friend: 'A', or 'B' for an independent confirmation
PLAN = 'main'      # 'smoke' for a shorter first attempt; 'extended' is much larger
exec(Path('/content/Attention_DRAM_Tests/COLAB_RUN.py').read_text())
```

Dependencies are checked in a fresh subprocess, so a notebook kernel restart is
not needed merely for these subprocess workers. A different Torch version is
rejected for matched replication rather than silently changing the baseline.

The runner prints progress and estimates remaining time from observed profile
durations. Expect a substantial run: model preparation, CPU references, Inductor
compilation and separate profiler processes can take tens of minutes or longer.
An extended run may take hours. There is no fixed runtime guarantee. Avoid
disconnecting before a backup is saved. Colab still controls session lifetime.

Backups occur every five minutes and on completion/interruption. The final cell
attempts a browser download. Browser downloads cannot be guaranteed; the Drive
copy is the durable fallback. Return the entire `attention_dram_*.tar.gz`.

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

## What is included and what remains untested

The kit includes hash-checked before/after IR, the C++ cache/trig source, original
model adapters, compiler flags, numerical gates, raw CSV parsing, compilation
records, randomized latency sampling, resumable profiles and result packaging.
Prepared model IR comes from the previously compiled C++ passes; it is not a new
handwritten approximation. Python manages experiments, not the compiler rewrite.
`iree_commit` is B's commit pass; `iree_forwarded` is the synthetic read pass.

Local validation is recorded in `validation/`: unit tests, actual CPU compilation
and execution, and sm_75 compilation. No GPU/profiler execution was available to
the authoring environment. Your Colab run is required to validate this exact
Nsight integration. A streaming-range preflight and real IREE-range preflight
stop a long sweep early if range capture fails. A kernel-only diagnostic is
attempted on IREE preflight failure, but is not promoted to a whole-step result.

The single-fork fixture is kept as a separate local ownership probe, not a
performance control. Local CPU lowering agreed numerically but changed caller
K/V input buffers. It is excluded from the default measured cases; the dense
two-fork fixture is the validated no-rewrite control. This is an observed
contract limitation for that fixture, not proof of a universal upstream bug.

## Output to return

Send the complete archive, including failures. In particular:

* `summary.json`, `dram_summary.csv`, `RESULTS.md` and `plan.json`;
* `environment.json`, `compilation.json` and `completion.json`;
* `profiles/*/metrics.csv`, `profile.ncu-rep`, `worker.json`, `task.json`, commands
  and `ncu.log`;
* `latency/*/latency.json`, which includes forced-winner eligibility, input
  checks, raw samples and telemetry;
* `nsys/*/trace.nsys-rep`, SQLite exports, copy summaries and logs when available.

Do not send cache/weights or input fixture files: these are excluded from the
archive to avoid gigabytes of redundant model data. Raw report files in a
periodic partial backup are included only after their collection record exists.

The useful outcome is a paired table of **DRAM read bytes, write bytes, total
bytes, and unprofiled latency** for the same correct workload. We will examine
traffic reductions alongside regressions and no-match controls. Allocation
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

The user's observed profiler is Nsight Compute 2025.3.1 on a T4. The suite saves
actual tool versions and does not require upgrading the GPU or driver.
