"""Actual export and controlled projection-placement experiments."""
import sys,os,json,time,statistics,random,subprocess,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT);sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,torch
from iree.turbine import aot
from run_suite import run,load,allocations

class ProjectedAttention(torch.nn.Module):
 def __init__(self,shared=False,masked=False):super().__init__();self.shared=shared;self.masked=masked
 def forward(self,q,c,wk,wv):
  if self.shared:c=c.expand(q.shape[0],-1,-1)
  k=torch.bmm(c,wk);v=torch.bmm(c,wv)
  s=torch.bmm(q,k.transpose(1,2))*q.shape[-1]**-0.5
  if self.masked:
   m,n=s.shape[-2:];mask=torch.arange(n)[None,:]>torch.arange(m)[:,None]+n-m
   s=s.masked_fill(mask,float('-inf'))
  return torch.bmm(s.softmax(-1),v)

CASES=[# headbatch, querylen, kvlen, latentwidth, keywidth, valuewidth, shared, causal
 (4,1,128,64,32,32,False,False),(8,1,512,128,64,64,True,False),
 (8,4,1024,128,64,64,True,False),(4,16,512,128,32,64,False,False),
 (4,64,512,128,32,32,True,False),(4,128,512,128,32,32,True,False),
 (4,256,512,128,32,32,True,False),(4,512,512,128,32,32,True,False),
 (8,1,2048,512,128,128,True,False),(8,16,2048,512,128,128,True,False),
 (4,1,4096,128,64,64,False,False),(4,8,1024,128,64,32,True,True),
 (2,7,257,96,40,24,True,False),(4,32,512,32,128,64,False,False),
 (4,128,512,32,128,64,True,False),(2,1,256,64,64,64,True,False)]

STRONG=['--iree-opt-level=O3','--iree-opt-generalize-matmul','--iree-dispatch-creation-enable-aggressive-reshape-movement']

def create_inputs(case):
 h,m,n,l,d,dv,shared,masked=case;torch.manual_seed(123)
 return (torch.randn(h,m,d),torch.randn(1 if shared else h,n,l),torch.randn(h,l,d)/l**.5,torch.randn(h,l,dv)/l**.5)

def annotate(text):
 # Insert a standard discardable function attribute using the compiler's parser.
 from iree.compiler import ir
 from iree.compiler.dialects import func
 with ir.Context():
  module=ir.Module.parse(text)
  for op in module.body.operations:
   if op.operation.name=='func.func':op.attributes['attention.allow_reassociation']=ir.UnitAttr.get()
  return str(module)

def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
 rows=[]
 for case in CASES:
  h,m,n,l,d,dv,shared,masked=case
  name=f'h{h}_m{m}_n{n}_l{l}_d{d}_v{dv}_s{int(shared)}_mask{int(masked)}'
  w=Path('research/results/projections')/name;w.mkdir(parents=True,exist_ok=True)
  x=create_inputs(case);model=ProjectedAttention(shared,masked).eval()
  with torch.inference_mode():ref=model(*[a.double() for a in x]).numpy();torchref=model(*x).numpy()
  aot.export(torch.export.export(model,x)).save_mlir(str(w/'torch.mlir'))
  run(['iree-opt',w/'torch.mlir','--torch-function-to-torch-backend-pipeline','--torch-backend-to-linalg-on-tensors-backend-pipeline','--canonicalize','-o',w/'baseline.mlir'],w/'lower.log')
  baseline=(w/'baseline.mlir').read_text();(w/'contract.mlir').write_text(annotate(baseline))
  pass_s=run(['build/gqa-opt',w/'contract.mlir','--attention-project-reassociate','--canonicalize','-o',w/'optimized.mlir'],w/'pass.log')
  row=dict(case=name,shape=case,ir_changed=baseline.replace(' ', '')!=(w/'optimized.mlir').read_text().replace(' ',''),pass_s=pass_s,variants={})
  loaded={}
  for v,source,flags in [('baseline','baseline',[]),('strong','baseline',STRONG),('optimized','optimized',STRONG)]:
   compile_flags=['--iree-hal-target-backends=llvm-cpu','--iree-llvmcpu-target-cpu=host',*flags]
   compile_s=run(['iree-compile',w/(source+'.mlir'),*compile_flags,'-o',w/(v+'.vmfb')],w/(v+'-compile.log'))
   run(['iree-compile',w/(source+'.mlir'),*compile_flags,'--compile-to=hal','-o',w/(v+'-hal.mlir')],w/(v+'-hal.log'))
   cfg,ctx,fn,args=load(w/(v+'.vmfb'),x);before=cfg.device.allocator.statistics;y=fn(*args).to_host();after=cfg.device.allocator.statistics
   delta=y-ref;maxabs=float(np.max(np.abs(delta)));rel=float(np.linalg.norm(delta)/max(np.linalg.norm(ref),1e-30))
   np.testing.assert_allclose(y,ref,atol=4e-5,rtol=1e-3)
   row['variants'][v]=dict(compile_s=compile_s,max_abs_vs_f64=maxabs,relative_l2_vs_f64=rel,hal_alloc_bytes=sum(allocations((w/(v+'-hal.mlir')).read_text())),runtime_alloc_bytes=after['device_bytes_allocated']-before['device_bytes_allocated'],runtime_peak_bytes=after['device_bytes_peak'])
   loaded[v]=(cfg,ctx,fn,args)
  samples={v:[] for v in loaded};samples['torch']=[]
  with torch.inference_mode():
   for cfg,ctx,fn,args in loaded.values():
    for _ in range(5):fn(*args)
   randomizer=random.Random(42)
   for trial in range(13):
    order=list(loaded);randomizer.shuffle(order)
    for v in order:
     _,_,fn,args=loaded[v];start=time.perf_counter_ns()
     for _ in range(5):fn(*args)
     samples[v].append((time.perf_counter_ns()-start)/5000)
    start=time.perf_counter_ns()
    for _ in range(5):model(*x)
    samples['torch'].append((time.perf_counter_ns()-start)/5000)
  row['samples_us']=samples;row['median_us']={v:statistics.median(s) for v,s in samples.items()};row['speedup_vs_strong']=row['median_us']['strong']/row['median_us']['optimized']
  rows.append(row);Path('research/results/projection-study.json').write_text(json.dumps(dict(cpu=min(os.sched_getaffinity(0)),trials=13,inner=5,strong_flags=STRONG,rows=rows),indent=2))
  print(name,'PASS','vs_strong=%.3f'%row['speedup_vs_strong'],'bytes',row['variants']['strong']['runtime_alloc_bytes'],'->',row['variants']['optimized']['runtime_alloc_bytes'],flush=True)
  del loaded;gc.collect()
 print('Completed',len(rows),flush=True)
if __name__=='__main__':main()
