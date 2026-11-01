"""Run from the extracted project root; archive results even on failure."""
import argparse, os, subprocess, sys, tarfile
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--mode',choices=['probe','benchmarks'],default='probe')
p.add_argument('--arch',default='sm_75')
p.add_argument('--trials',default='101')
p.add_argument('--seeds',default='10')
a=p.parse_args()
root=Path(__file__).resolve().parent;os.chdir(root)
out=root/('followup_'+a.mode);out.mkdir(exist_ok=True)
env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
commands=[]
if a.mode=='probe':
    commands=[[sys.executable,'research/native_width_probe.py','--arch',a.arch,'--seeds',a.seeds,'--trials',a.trials,'--output',str(out/'probe')]]
else:
    for family in ['attention','block','winner']:
        commands.append([sys.executable,'research2/gpu_suite.py','--arch',a.arch,'--family',family,'--trials',a.trials,'--torch-cuda'])
    for family in ['cache','winner']:
        commands.append([sys.executable,'pbs_gpu/validate.py','--arch',a.arch,'--family',family,'--seeds',a.seeds,'--output',str(out/('validation_'+family+'.json'))])
statuses=[]
try:
    for i,cmd in enumerate(commands):
        print('Running:',' '.join(cmd),flush=True)
        with (out/f'step_{i+1}.log').open('w') as log:
            rc=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        statuses.append((rc,cmd));print('Exit:',rc,flush=True)
finally:
    import json
    (out/'commands.json').write_text(json.dumps(statuses,indent=2))
    # Existing collector adds package protocol, timing JSON and generated logs/PTX.
    subprocess.run([sys.executable,'pbs_gpu/collect.py',str(out)],env=env)
    for name,cmd in [('compiler',['iree-compile','--version']),('hardware',['nvidia-smi']),('packages',[sys.executable,'-m','pip','freeze'])]:
        with (out/(name+'.txt')).open('w') as f:
            try:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
            except OSError as e:f.write(str(e))
    archive=root/f'gpu_followup_{a.mode}.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:tar.add(out,arcname=out.name)
    print('Saved:',archive,flush=True)
if len(statuses)!=len(commands) or any(rc for rc,_ in statuses):raise SystemExit(1)
