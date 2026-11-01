import argparse,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--set',choices=['A','B'],required=True);p.add_argument('--quick',action='store_true');p.add_argument('--skip-torch-compile',action='store_true');a=p.parse_args()
def run(cmd):print('RUN',*cmd,flush=True);subprocess.run(cmd,cwd=ROOT,check=True)
run([sys.executable,str(ROOT/'scripts/download_model.py')])
cmd=[sys.executable,str(ROOT/'benchmark.py'),'--set',a.set,'--output','results_'+a.set]
cmd+=['--contexts','128','512','--trials','7','--repetitions','1','--prompts','1','--steps','4'] if a.quick else []
if not a.skip_torch_compile:cmd+=['--torch-compile']
run(cmd);run([sys.executable,str(ROOT/'analyze.py'),str(ROOT/('results_'+a.set))]);print('ALL TASKS COMPLETED',flush=True)
