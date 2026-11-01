"""Pinned comparison of explicit attention and native SDPA on the original shape."""
import json,os,random,statistics,time
from pathlib import Path
import torch
from run_suite import GQA,inputs,load
from investigate_native_gqa import NativeGQA

def main():
 torch.set_num_threads(1);os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
 x=inputs();loaded=[];fns={}
 for name,path in [('iree_explicit','results/cases/b1_q8_kv2_m1_n128_d32/baseline.vmfb'),('iree_reuse','results/cases/b1_q8_kv2_m1_n128_d32/optimized.vmfb'),('iree_native','results/native-gqa/native.vmfb')]:
  cfg,ctx,fn,args=load(Path(path),x);loaded.append((cfg,ctx,fn,args));fns[name]=lambda fn=fn,args=args:fn(*args)
 explicit=GQA().eval();native=NativeGQA().eval()
 fns['torch_explicit']=lambda:explicit(*x);fns['torch_native']=lambda:native(*x)
 samples={k:[] for k in fns};rng=random.Random(42)
 with torch.inference_mode():
  for fn in fns.values():
   for _ in range(10):fn()
  for _ in range(31):
   order=list(fns);rng.shuffle(order)
   for name in order:
    start=time.perf_counter_ns()
    for _ in range(100):fns[name]()
    samples[name].append((time.perf_counter_ns()-start)/100/1000)
 out=dict(cpu=min(os.sched_getaffinity(0)),trials=31,inner=100,samples_us=samples,median_us={k:statistics.median(s) for k,s in samples.items()})
 Path('results/native-gqa/pinned-comparison.json').write_text(json.dumps(out,indent=2));print(out['median_us'])
if __name__=='__main__':main()
