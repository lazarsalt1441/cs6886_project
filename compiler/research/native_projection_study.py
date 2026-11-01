"""Preserve fused SDPA while relocating latent K/V projections."""
import os,sys,json,statistics,random,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from projection_study import ROOT,create_inputs,annotate,STRONG
from run_suite import run,load,allocations
from cpu_compile import compile_guarded
import torch,numpy as np
from iree.turbine import aot
from iree.compiler import ir
class NativeProjected(torch.nn.Module):
 def __init__(self,shared):super().__init__();self.shared=shared
 def forward(self,q,c,wk,wv):
  if self.shared:c=c.expand(q.shape[0],-1,-1)
  return torch.nn.functional.scaled_dot_product_attention(q.unsqueeze(0),torch.bmm(c,wk).unsqueeze(0),torch.bmm(c,wv).unsqueeze(0),enable_gqa=True).squeeze(0)

def main():
 torch.set_num_threads(1);os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
 cases=[(4,1,128,64,32,32,False,False),(8,1,512,128,64,64,True,False),
 (8,16,512,128,64,64,True,False),(4,128,512,128,32,32,True,False),
 (8,1,2048,512,128,128,True,False),(2,7,257,96,40,24,True,False),
 (4,32,512,32,128,64,False,False)]
 rows=[]
 for case in cases:
  h,m,n,l,d,dv,shared,_=case;name=f'h{h}_m{m}_n{n}_l{l}_d{d}_v{dv}_s{int(shared)}'
  w=Path('research/results/native-projections')/name;w.mkdir(parents=True,exist_ok=True)
  x=create_inputs(case);model=NativeProjected(shared).eval()
  with torch.inference_mode():ref=model(*[v.double() for v in x]).numpy()
  aot.export(torch.export.export(model,x)).save_mlir(str(w/'torch.mlir'))
  run(['iree-opt',w/'torch.mlir','--torch-function-to-torch-backend-pipeline','--torch-backend-to-linalg-on-tensors-backend-pipeline','--canonicalize','-o',w/'baseline.mlir'],w/'lower.log')
  with ir.Context():
   module=ir.Module.parse(annotate((w/'baseline.mlir').read_text()).replace('attention.allow_reassociation','attention.enable_native_projection, attention.allow_reassociation'))
   (w/'contract-generic.mlir').write_text(module.operation.get_asm(print_generic_op_form=True))
  run(['build/gqa-opt',w/'contract-generic.mlir','--allow-unregistered-dialect','--attention-project-reassociate','--canonicalize','-o',w/'optimized.mlir'],w/'pass.log')
  row={'case':name,'shape':case,'placements':(w/'optimized.mlir').read_text().count('attention.projection_placement'),'variants':{}}
  loaded={}
  for variant,source in [('native','baseline'),('native-optimized','optimized')]:
   flags=['--iree-hal-target-backends=llvm-cpu','--iree-llvmcpu-target-cpu=host',*STRONG]
   guarded=compile_guarded(w/(source+'.mlir'),w/(variant+'.vmfb'),w,variant)
   run(['iree-compile',w/(source+'.mlir'),*flags,'--compile-to=hal','-o',w/(variant+'-hal.mlir')],w/(variant+'-hal.log'))
   cfg,ctx,fn,args=load(w/(variant+'.vmfb'),x);before=cfg.device.allocator.statistics;y=fn(*args).to_host();after=cfg.device.allocator.statistics
   np.testing.assert_allclose(y,ref,atol=4e-5,rtol=1e-3)
   row['variants'][variant]={'guarded_ops':guarded,'max_abs':float(np.max(np.abs(y-ref))),'relative_l2':float(np.linalg.norm(y-ref)/np.linalg.norm(ref)),'alloc_bytes':after['device_bytes_allocated']-before['device_bytes_allocated'],'hal_bytes':sum(allocations((w/(variant+'-hal.mlir')).read_text()))}
   loaded[variant]=(cfg,ctx,fn,args)
  samples={v:[] for v in loaded};rng=random.Random(42)
  for _,_,fn,args in loaded.values():
   for _ in range(5):fn(*args)
  for _ in range(15):
   order=list(loaded);rng.shuffle(order)
   for v in order:
    _,_,fn,args=loaded[v];start=time.perf_counter_ns()
    for _ in range(5):fn(*args)
    samples[v].append((time.perf_counter_ns()-start)/5000)
  row['samples_us']=samples;row['median_us']={v:statistics.median(s) for v,s in samples.items()};row['speedup']=row['median_us']['native']/row['median_us']['native-optimized'];rows.append(row)
  Path('research/results/native-projection-study.json').write_text(json.dumps({'cpu':min(os.sched_getaffinity(0)),'rows':rows},indent=2));print(name,'PASS','speedup=%.3f'%row['speedup'],flush=True)
 print('Completed',len(rows),flush=True)
if __name__=='__main__':main()
