import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def cell(kind,text):return dict(cell_type=kind,metadata={},source=text.splitlines(keepends=True),**(dict(execution_count=None,outputs=[]) if kind=='code' else {}))
for which in 'AB':
 title='Ordinary full-model decode: applicability control' if which=='A' else 'Full-model two-candidate lookahead: cache commit factoring'
 cells=[cell('markdown',f'''# Set {which}: {title}
Choose **Runtime → Change runtime type → T4 GPU** (or another NVIDIA GPU).
Upload `Attention_Real_LLM_Tests.zip` in the first cell. Each person runs one notebook.
Baselines: Torch eager, natural Torch SDPA, optional Torch compile and strong IREE.
Pinned pretrained SmolLM2-135M-Instruct, f32, original HF eager attention.
Set A has zero pass matches; Set B has 60. No GPU speedup is assumed.
Run cells in order. The final cell periodically backs up to Drive and attempts an automatic browser download at completion, including failure. Browser/session restrictions can prevent download; use Drive backup then.
'''),cell('code','''from google.colab import files
from pathlib import Path
import zipfile
uploaded = files.upload()  # Select Attention_Real_LLM_Tests.zip
candidates = [Path(name) for name in uploaded if Path(name).suffix.lower() == ".zip"]
assert len(candidates) == 1, "Upload only the common Attention_Real_LLM_Tests ZIP"
bundle = candidates[0]
project = Path("/content/attention_real_llm")
project.mkdir(exist_ok=True)
with zipfile.ZipFile(bundle) as z:
    assert "prepared/A/audit.json" in z.namelist() and "prepared/B/audit.json" in z.namelist(), "Wrong or incomplete test bundle"
    assert all(not Path(n).is_absolute() and ".." not in Path(n).parts for n in z.namelist())
    z.extractall(project)
print("Project:", project)
'''),cell('code','''import os, subprocess, sys
venv = Path("/content/attention_real_env")
# Bundled pip avoids the previous ensurepip failure. Retain Colab CUDA Torch.
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "virtualenv"], check=True)
subprocess.run([sys.executable, "-m", "virtualenv", "--system-site-packages", "--no-download", str(venv)], check=True)
python = str(venv / "bin/python")
env = os.environ.copy()
env["PATH"] = str(venv / "bin") + os.pathsep + env["PATH"]
env["OMP_NUM_THREADS"] = "1"
env["OPENBLAS_NUM_THREADS"] = "1"
with (project / "setup.log").open("w") as log:
    result = subprocess.run([python, "-m", "pip", "install", "-r", str(project / "requirements-colab.txt")], env=env, stdout=log, stderr=subprocess.STDOUT)
if result.returncode:
    print((project / "setup.log").read_text()[-8000:])
    raise RuntimeError("Setup failed; upload setup.log")
subprocess.run([python, "-c", "import text_only; import torch, iree.runtime as r; from transformers import LlamaForCausalLM; print('Torch:', torch.__version__, 'CUDA:', torch.version.cuda); assert torch.cuda.is_available(), 'Choose GPU runtime'; print('GPU:', torch.cuda.get_device_name()); print('IREE:', r.get_driver('cuda').query_available_devices())"], cwd=project, env=env, check=True)
print("Setup PASS")
'''),cell('code','''from google.colab import drive
drive.mount("/content/drive")
backup_root = Path("/content/drive/MyDrive/Attention_Real_LLM_Results")
backup_root.mkdir(parents=True, exist_ok=True)
print("Drive backup ready")
'''),cell('markdown','''Default: contexts 128/512/2048, two prompts, 31 trials × 3 repetitions, plus 16-token continuations. Torch compilation is outside measured trials and may take several minutes per shape. Set `QUICK=True` for a first pipeline check; use `False` for final measurements. Keep `SKIP_TORCH_COMPILE=False` for the strongest baseline. Unsupported Torch compilation is logged and marked unavailable.

Leave the next cell running. It prints a heartbeat every minute and saves a partial Drive archive every five minutes. This does not keep a disconnected Colab runtime alive.
'''),cell('code',f'''SET = "{which}"
QUICK = False
SKIP_TORCH_COMPILE = False

import signal, shutil, time
from datetime import datetime, timezone
stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
backup_dir = backup_root / ("set_" + SET + "_" + stamp)
backup_dir.mkdir(parents=True, exist_ok=True)
archive = project / ("real_llm_set_" + SET + "_" + stamp + ".tar.gz")
log_path = project / ("runner_" + SET + ".log")

def save_snapshot():
    subprocess.run([python, "scripts/archive_results.py", "--set", SET, "--destination", str(archive)], cwd=project, env=env, check=True, stdout=subprocess.DEVNULL)
    shutil.copy2(archive, backup_dir / archive.name)
    if log_path.exists(): shutil.copy2(log_path, backup_dir / log_path.name)
    if (project / "setup.log").exists(): shutil.copy2(project / "setup.log", backup_dir / "setup.log")

cmd = [python, "-u", "scripts/run_colab_set.py", "--set", SET]
if QUICK: cmd.append("--quick")
if SKIP_TORCH_COMPILE: cmd.append("--skip-torch-compile")
proc = None
try:
    with log_path.open("w", buffering=1) as log:
        proc = subprocess.Popen(cmd, cwd=project, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        start = last_backup = time.monotonic()
        while proc.poll() is None:
            try: proc.wait(timeout=60)
            except subprocess.TimeoutExpired: pass
            print("Elapsed minutes:", round((time.monotonic()-start)/60, 1), flush=True)
            print("\\n".join(log_path.read_text(errors="replace").splitlines()[-3:]), flush=True)
            if time.monotonic()-last_backup >= 300:
                save_snapshot(); last_backup=time.monotonic()
                print("Partial backup saved", flush=True)
        print("Exit code:", proc.returncode)
        if proc.returncode: print(log_path.read_text(errors="replace")[-10000:])
except BaseException:
    if proc is not None and proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try: proc.wait(timeout=10)
        except subprocess.TimeoutExpired: os.killpg(proc.pid, signal.SIGKILL)
    raise
finally:
    save_snapshot()
    print("Saved to Drive:", backup_dir, flush=True)
    print("Starting browser download:", archive.name, flush=True)
    files.download(str(archive))
'''),cell('markdown','''Upload both resulting `real_llm_set_*.tar.gz` archives here. They include correctness/errors, randomized timings, compilation logs, generation tokens, environment, MLIR and PTX when available. Model weights are excluded. If browser download is blocked, use Drive or rerun the next cell.
'''),cell('code','''# Run only to repeat the download.
files.download(str(archive))
''')]
 for i,c in enumerate(cells):c['id']=f'cell-{i:02d}'
 nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(colab=dict(name=f'Colab_Set_{which}.ipynb'),accelerator='GPU',kernelspec=dict(display_name='Python 3',language='python',name='python3')),cells=cells)
 (ROOT/f'Colab_Set_{which}.ipynb').write_text(json.dumps(nb,indent=2))
