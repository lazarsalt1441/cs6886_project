# Evaluation checks and follow-up

[Saved-run analysis](results/ANALYSIS.md) and [error-bar plot](results/latency_uncertainty.png) distinguish within-session variation from independent GPU replication. Three fixed prompts and one session do not establish general confidence intervals.

The 2048 gain appears in every paired repetition and persists after dropping the first repetition. That rules out a single bad repetition as the explanation; it does not identify the responsible kernels. Whole-step DRAM falls only 3.91%, so the 26.46% latency change cannot be attributed to the synthetic cache-copy mechanism without more evidence.

The dense control has identical canonical input and matching fresh CUDA codegen, including PTX. Its paired timing ratios change direction across repetitions. Whole VMFB hashes differ (embedded metadata is included), so whole-executable byte identity is not asserted. The recorded slowdown is retained; its exact runtime cause is not resolved.

Recompute the saved statistics after unpacking, with the Python standard library:

```bash
python3 evaluation/analyze_saved.py
python3 -m unittest discover -s evaluation -p 'test_*.py' -v
```

Add `--plot` with Matplotlib installed. To repeat the code comparison, first run the root CUDA compile-only test, then add `--compiled-cache local_validation/compiled/cuda --iree-opt /path/to/iree-opt`. Original source/compiler flags remain in the DRAM build records.

## New Colab GPU experiment

This runner is prepared and its plan/statistical helpers are unit-tested. It has **not been executed on a GPU in this revision**. It adds contexts 512/1024/1536/1792/2048/2304/2560/3072/4096, three prompts, seven repetition blocks and 100 warmup calls per backend. It releases previous outputs before the timer; do not pool these new timings with the original protocol. The dense placebo loads byte-identical copies of one baseline VMFB under two labels, testing runtime/context/measurement effects without a compiler transformation. Its true role is recorded in the plan and build manifests.

Use the working Torch 2.11 CUDA environment from the main GPU instructions. Install the remaining pins with `python -m pip install -r real_model/requirements-colab.txt`. Nsight Compute permissions are still required; Systems traces, when available, cover contexts 1792, 2048 and 2304 for kernel-level investigation. Inspect the plan without a GPU:

```bash
python3 evaluation/run_followup.py --print-plan
```

In Colab, after cloning/unpacking and mounting Drive, run this cell. It retains local JSON checkpoints and attempts a final Drive backup and download. Timing/profiling output stays local; archive compression and Drive copying happen outside the worker process's measurement windows, but simultaneous compression can still disturb CPU timing. For final measurements use the runner directly and back up between cases or after completion if such interference is a concern.

```python
from pathlib import Path
from datetime import datetime, timezone
from google.colab import drive, files
import os, subprocess, sys, tarfile, shutil, time

drive.mount('/content/drive')
repo = Path('/content/cs6886_project')
if not repo.exists():
    subprocess.run(['git','clone','https://github.com/lazarsalt1441/cs6886_project.git',str(repo)],check=True)
subprocess.run([sys.executable,str(repo/'scripts/unpack_artifacts.py')],check=True)
run_id = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
out = Path('/content/evaluation_runs') / run_id
out.mkdir(parents=True,exist_ok=True)
archive = out.parent / (run_id + '.tar.gz')
backup = Path('/content/drive/MyDrive/Attention_Evaluation')
backup.mkdir(parents=True,exist_ok=True)
env = os.environ.copy()
env['OMP_NUM_THREADS'] = env['OPENBLAS_NUM_THREADS'] = '1'
# Put the installed ncu directory on PATH if the GPU setup requires it.
cmd = [sys.executable,str(repo/'evaluation/run_followup.py'),
       '--output',str(out),'--cache','/content/evaluation_cache']
try:
    with (out/'run.log').open('w') as log:
        proc = subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
        # Worker JSON checkpoints are written after repetitions; avoid compressing
        # archives concurrently with CPU-sensitive timing.
        while proc.poll() is None:
            try: proc.wait(timeout=30)
            except subprocess.TimeoutExpired: pass
            print('\n'.join((out/'run.log').read_text(errors='replace').splitlines()[-2:]),flush=True)
        if proc.returncode:
            print('Run failed; keeping partial diagnostics. Exit:',proc.returncode)
finally:
    if 'proc' in globals() and proc.poll() is None:
        proc.terminate()
        try:proc.wait(timeout=10)
        except subprocess.TimeoutExpired:proc.kill();proc.wait()
    with tarfile.open(archive,'w:gz') as tar:
        tar.add(out,arcname=out.name)
    shutil.copy2(archive,backup/archive.name)
    files.download(str(archive))
```

The final backup callback cannot survive abrupt runtime termination. Preserve the local run directory or copy it between cases if necessary. Run separate independent sessions with fresh output/cache directories. Do not interpret a same-binary placebo offset as an optimization benefit. Return the full archive, including `latency`, `compilation.json`, `nsys`, `profiles` and any errors. New context results and the causal explanation remain open until these GPU runs are completed.
