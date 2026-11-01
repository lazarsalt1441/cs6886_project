"""Recheck the entire no-custom-pass backend audit after C++ schedule guarding."""
import os,sys,json,statistics,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from projection_study import ROOT,STRONG
from cpu_compile import compile_guarded,FLAGS
from run_suite import load,run
from native_attention_audit import Attention
from iree.compiler import ir
import torch,numpy as np

def main():
 torch.set_num_threads(1);os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
 audit=json.loads(Path('research/results/backend-audit.json').read_text())['rows'];out=[]
 for old in audit:
  if 'shape' not in old:continue
  h,m,n,d=old['shape'];alias=old['alias'];w=Path('research/results/backend-audit')/old['case']
  torch.manual_seed(77);x=(torch.randn(1,h,m,d),torch.randn(1,h,n,d),torch.randn(1,h,n,d));model=Attention(alias)
  with torch.inference_mode():ref=model(*[a.double() for a in x]).numpy()
  changed=compile_guarded(w/'torch.mlir',w/'guarded.vmfb',w,'fix',baseline_destination=w/'original.vmfb')
  row=dict(case=old['case'],shape=old['shape'],alias=alias,guarded_ops=changed,variants={})
  for v in ['original','guarded']:
   cfg,ctx,fn,args=load(w/(v+'.vmfb'),x);before=cfg.device.allocator.statistics;y=fn(*args).to_host();after=cfg.device.allocator.statistics
   correct=bool(np.allclose(y,ref,atol=4e-5,rtol=1e-3));samples=[]
   for _ in range(5):fn(*args)
   for _ in range(9):
    start=time.perf_counter_ns()
    for _ in range(3):fn(*args)
    samples.append((time.perf_counter_ns()-start)/3000)
   row['variants'][v]=dict(correct=correct,max_abs=float(np.max(np.abs(y-ref))),samples_us=samples,median_us=statistics.median(samples),allocated_bytes=after['device_bytes_allocated']-before['device_bytes_allocated'])
   if v=='guarded':assert correct,(old['case'],row)
  out.append(row);Path('research/results/guard-validation.json').write_text(json.dumps({'rows':out,'note':'Original wrong-kernel latency is diagnostic, never a valid performance baseline.'},indent=2));print(old['case'],'guarded',changed,'original_correct',row['variants']['original']['correct'],'fixed_error',row['variants']['guarded']['max_abs'],flush=True)
 print('Completed',len(out),flush=True)
if __name__=='__main__':main()
