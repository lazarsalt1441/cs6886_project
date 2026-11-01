"""Sequential timing-only confirmation of ablations and two correct repairs."""
import sys,os,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from final_timings import measure
from projection_study import ROOT,create_inputs
from rotary_projection_study import rotate
from run_suite import load
import torch

def main():
 torch.set_num_threads(1);os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});out=[]
 exploratory=json.loads(Path('research/results/placement-ablations.json').read_text())
 for row in exploratory:
  name=row['case'];w=Path('research/results/ablations')/name
  # Decode shapes from the recorded case name, not from timing values.
  import re
  match=re.search(r'h(\d+)_m(\d+)_n(\d+)_l(\d+)_d(\d+)_v(\d+)',name)
  h,m,n,l,d,dv=map(int,match.groups())
  if name.startswith('dense'):
   shared='_s1_' in name;x=create_inputs((h,m,n,l,d,dv,shared,False))
  else:
   r=int(re.search(r'_r(\d+)',name).group(1));x=create_inputs((h,m,n,l,d,dv,True,False));x=(*x,rotate(torch.randn(h,m,r),torch.arange(n-m,n)),rotate(torch.randn(1,n,r),torch.arange(n)))
  loaded={};functions={}
  for variant in row['variants']:
   cfg,ctx,fn,args=load(w/(variant+'.vmfb'),x);loaded[variant]=(cfg,ctx,fn,args);functions[variant]=lambda fn=fn,args=args:fn(*args)
  result=measure(functions);result.update(case=name,family='ablation');out.append(result)
  Path('research/results/final-ablation-timings.json').write_text(json.dumps(out,indent=2));print(name,flush=True)
 for h,m,n,d in [(1,1,128,256),(1,16,256,512),(1,128,2048,256)]:
  name=f'h{h}_m{m}_n{n}_d{d}';w=Path('research/results/fallback-comparison')/name;torch.manual_seed(88)
  x=(torch.randn(1,h,m,d),torch.randn(1,h,n,d),torch.randn(1,h,n,d));loaded={};functions={}
  for v in ['guarded','fallback']:
   cfg,ctx,fn,args=load(w/(v+'.vmfb'),x);loaded[v]=(cfg,ctx,fn,args);functions[v]=lambda fn=fn,args=args:fn(*args)
  result=measure(functions);result.update(case=name,family='fallback');out.append(result)
  Path('research/results/final-ablation-timings.json').write_text(json.dumps(out,indent=2));print(name,flush=True)
 print('Completed',len(out),flush=True)
if __name__=='__main__':main()
