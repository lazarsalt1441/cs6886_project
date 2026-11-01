"""Execute from a Colab cell after uploading/extracting this kit. See README.md."""
from pathlib import Path
from datetime import datetime, timezone
import json, os, shutil, signal, subprocess, sys, tarfile, time
from google.colab import files

kit=Path(globals().get('DRAM_KIT_ROOT','/content/Attention_DRAM_Tests')).resolve()
assert (kit/'study.py').is_file(), 'Set DRAM_KIT_ROOT to the repository dram directory'
track=globals().get('TRACK','B');plan=globals().get('PLAN','main')
assert track in ['A','B'] and plan in ['smoke','main','extended']
candidates=[Path('/content/attention_real_env/bin/python'),Path('/content/attention_env/bin/python'),Path(sys.executable)]
python=None
for candidate in candidates:
    if not candidate.exists():continue
    test=subprocess.run([str(candidate),'-c','import torch; assert torch.cuda.is_available()'],capture_output=True)
    if test.returncode==0:python=str(candidate);break
assert python is not None,'A CUDA PyTorch Python interpreter is required'
print('Python:',python,flush=True)
probe=subprocess.run([python,'-c','import iree.compiler,iree.runtime,iree.turbine,transformers,numpy; from importlib.metadata import version; assert version("iree-base-compiler")==version("iree-base-runtime")=="3.13.0"; assert version("iree-turbine")=="3.9.0"; assert transformers.__version__=="4.57.1"'],capture_output=True)
if probe.returncode:
    subprocess.run([python,'-m','pip','install','--disable-pip-version-check','iree-base-compiler==3.13.0',
                    'iree-base-runtime==3.13.0','iree-turbine==3.9.0','transformers==4.57.1',
                    'numpy==2.3.5','ml_dtypes==0.6.0','safetensors','huggingface_hub'],check=True)
subprocess.run([python,'-c','import torch; print("Torch:",torch.__version__); assert torch.__version__.startswith("2.11."), "Use the original Torch 2.11 environment for matched replication"'],check=True)
env=os.environ.copy();env['PATH']=str(Path(python).parent)+':'+env['PATH']
env['OMP_NUM_THREADS']=env['OPENBLAS_NUM_THREADS']='1'
if not shutil.which('ncu',path=env['PATH']):
    found=list(Path('/usr/local').glob('cuda*/**/ncu'))+list(Path('/opt/nvidia').glob('**/ncu')) if Path('/opt/nvidia').exists() else list(Path('/usr/local').glob('cuda*/**/ncu'))
    if found:env['PATH']=str(found[0].parent)+':'+env['PATH']
assert shutil.which('ncu',path=env['PATH']),'ncu not found in this GPU session'
run_id=globals().get('DRAM_RUN_ID')
if globals().get('DRAM_RUN_CONFIG') not in [None,(track,plan)]:run_id=None
if not run_id:run_id=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'_'+track+'_'+plan
globals()['DRAM_RUN_ID']=run_id
globals()['DRAM_RUN_CONFIG']=(track,plan)
base=kit/'runs'/run_id;base.mkdir(parents=True,exist_ok=True)
archive=kit/('attention_dram_'+run_id+'.tar.gz')
drive=Path('/content/drive/MyDrive/Attention_DRAM_Results')
print('Results:',base,flush=True)
print('Drive backup enabled:',Path('/content/drive/MyDrive').is_dir(),flush=True)

def snapshot():
    tmp=archive.with_suffix('.tmp')
    with tarfile.open(tmp,'w:gz') as tar:
        for path in base.rglob('*'):
            if not path.is_file() or path.suffix=='.tmp':continue
            if path.suffix in ['.ncu-rep','.nsys-rep','.sqlite'] and not (path.parent/'collection.json').exists():continue
            try:tar.add(path,arcname=path.relative_to(base.parent),recursive=False)
            except (FileNotFoundError,OSError):pass
    tmp.replace(archive)
    if Path('/content/drive/MyDrive').is_dir():
        drive.mkdir(parents=True,exist_ok=True);shutil.copy2(archive,drive/archive.name)
    return archive

proc=None
try:
    plans=['smoke',plan] if plan!='smoke' else ['smoke']
    for stage in plans:
        log=base/(stage+'.log')
        cmd=[python,'-u',str(kit/'study.py'),'--track',track,'--plan',stage,
             '--cache',str(kit/'cache'),'--output',str(base/stage)]
        print('RUN',stage,'track',track,flush=True)
        with log.open('a') as f:
            proc=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            last_backup=time.monotonic()
            while proc.poll() is None:
                try:proc.wait(timeout=30)
                except subprocess.TimeoutExpired:pass
                print('\n'.join(log.read_text(errors='replace').splitlines()[-3:]),flush=True)
                if time.monotonic()-last_backup>=300:
                    snapshot();last_backup=time.monotonic();print('Partial backup saved',flush=True)
        snapshot()
        if proc.returncode:raise RuntimeError(f'{stage} failed with exit {proc.returncode}; the partial archive contains diagnostics')
        completion=json.loads((base/stage/'completion.json').read_text())
        if completion.get('profile_failures'):print('Some profiles failed; retain them for analysis:',completion['profile_failures'],flush=True)
    print('COMPLETED. Read RESULTS.md and summary.json; a successful process exit alone is not a performance conclusion.',flush=True)
finally:
    if proc is not None and proc.poll() is None:
        os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    snapshot()
    print('Saved:',archive,flush=True)
    if Path('/content/drive/MyDrive').is_dir():print('Drive copy:',drive/archive.name,flush=True)
    print('Attempting browser download',flush=True)
    files.download(str(archive))
