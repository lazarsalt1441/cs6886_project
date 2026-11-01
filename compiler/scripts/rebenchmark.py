"""Timing-only confirmation, single pinned CPU, interleaved variants, raw samples."""
import json, os, random, statistics, time
from pathlib import Path
import numpy as np
import torch
from run_suite import ROOT,inputs,GQA,load

def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 cpu=min(os.sched_getaffinity(0));os.sched_setaffinity(0,{cpu})
 base=json.loads(Path('results/benchmark.json').read_text())['rows'];out=[]
 for r in base:
  case=[r[k] for k in ['b','hq','hkv','m','n','d']];x=inputs(*case);w=Path('results/cases')/r['case']
  loaded={};fns={};model=GQA().eval()
  for v in ['baseline','optimized','compact','aggressive-fusion','compact-fusion']:
   if (w/(v+'.vmfb')).exists():
    cfg,ctx,fn,args=load(w/(v+'.vmfb'),x);loaded[v]=(cfg,ctx,fn,args)
    fns[v]=lambda fn=fn,args=args:fn(*args)
  fns['torch_eager']=lambda:model(*x)
  samples={v:[] for v in fns};rng=random.Random(42)
  # At least 21 observations. Each timed batch targets 10ms using original latency.
  inner=max(1,min(100,int(10000/max(r['baseline_median_us'],r['optimized_median_us']))))
  with torch.inference_mode():
   for fn in fns.values():
    for _ in range(10):fn()
   for _ in range(21):
    order=list(fns);rng.shuffle(order)
    for v in order:
     start=time.perf_counter_ns()
     for _ in range(inner):fns[v]()
     samples[v].append((time.perf_counter_ns()-start)/inner/1000)
  row={'case':r['case'],'inner':inner,'samples_us':samples,'median_us':{v:statistics.median(s) for v,s in samples.items()}}
  row['speedup']=row['median_us']['baseline']/row['median_us']['optimized'];out.append(row)
  Path('results/pinned_benchmark.json').write_text(json.dumps(dict(cpu=cpu,trials=21,warmup=10,rows=out,method='Pinned single CPU, local-sync, preuploaded IREE inputs, no output copy; randomized variant order each trial. Torch eager uses one thread. Wrapper+execution timing; compilation excluded.'),indent=2))
  print(r['case'],'speedup=%.3f'%row['speedup'],flush=True)
if __name__=='__main__':main()
