"""Partial-RoPE latent attention, with the rotary subspace left untouched."""
import os,sys,json,time,random,statistics
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from projection_study import ROOT,create_inputs,annotate,STRONG
from run_suite import run,load,allocations
import torch,numpy as np
from iree.turbine import aot
from iree.compiler import ir
class RotaryProjected(torch.nn.Module):
 def forward(self,q,c,wk,wv,qr,kr):
  c=c.expand(q.shape[0],-1,-1);kr=kr.expand(q.shape[0],-1,-1)
  q=torch.cat([q,qr],dim=-1)
  k=torch.cat([torch.bmm(c,wk),kr],dim=-1)
  return torch.nn.functional.scaled_dot_product_attention(q,k,torch.bmm(c,wv))
def rotate(x,positions):
 r=x.shape[-1];freq=10000**(-torch.arange(r//2)*2/r)
 angle=positions[:,None]*freq[None,:];angle=torch.cat([angle,angle],-1)
 return x*angle.cos()+torch.cat([-x[...,r//2:],x[...,:r//2]],-1)*angle.sin()
def main():
 torch.set_num_threads(1);os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
 cases=[(4,1,128,64,32,32,16),(8,4,512,128,64,64,32),(16,1,1024,512,128,128,64),(4,256,512,128,32,32,16)]
 rows=[]
 for h,m,n,l,d,dv,r in cases:
  name=f'h{h}_m{m}_n{n}_l{l}_d{d}_v{dv}_r{r}';w=Path('research/results/rotary-projections')/name;w.mkdir(parents=True,exist_ok=True)
  x=create_inputs((h,m,n,l,d,dv,True,False));qr=rotate(torch.randn(h,m,r),torch.arange(n-m,n));kr=rotate(torch.randn(1,n,r),torch.arange(n));x=(*x,qr,kr)
  model=RotaryProjected().eval()
  with torch.inference_mode():ref=model(*[a.double() for a in x]).numpy()
  aot.export(torch.export.export(model,x)).save_mlir(str(w/'torch.mlir'))
  run(['iree-opt',w/'torch.mlir','--torch-function-to-torch-backend-pipeline','--torch-backend-to-linalg-on-tensors-backend-pipeline','--canonicalize','-o',w/'baseline.mlir'],w/'lower.log')
  with ir.Context():
   module=ir.Module.parse(annotate((w/'baseline.mlir').read_text()));(w/'contract.mlir').write_text(module.operation.get_asm(print_generic_op_form=True))
  run(['build/gqa-opt',w/'contract.mlir','--allow-unregistered-dialect','--attention-project-reassociate','--canonicalize','-o',w/'optimized.mlir'],w/'pass.log')
  row=dict(case=name,shape=[h,m,n,l,d,dv,r],placements=(w/'optimized.mlir').read_text().count('attention.projection_placement'),variants={});loaded={}
  for v,source in [('native','baseline'),('optimized','optimized')]:
   flags=['--iree-hal-target-backends=llvm-cpu','--iree-llvmcpu-target-cpu=host',*STRONG]
   run(['iree-compile',w/(source+'.mlir'),*flags,'-o',w/(v+'.vmfb')],w/(v+'-compile.log'))
   cfg,ctx,fn,args=load(w/(v+'.vmfb'),x);before=cfg.device.allocator.statistics;y=fn(*args).to_host();after=cfg.device.allocator.statistics
   np.testing.assert_allclose(y,ref,atol=5e-5,rtol=2e-3)
   row['variants'][v]=dict(max_abs=float(np.max(np.abs(y-ref))),relative_l2=float(np.linalg.norm(y-ref)/np.linalg.norm(ref)),alloc_bytes=after['device_bytes_allocated']-before['device_bytes_allocated']);loaded[v]=(cfg,ctx,fn,args)
  samples={v:[] for v in loaded};rng=random.Random(42)
  for _,_,fn,args in loaded.values():
   for _ in range(5):fn(*args)
  for _ in range(13):
   order=list(loaded);rng.shuffle(order)
   for v in order:
    _,_,fn,args=loaded[v];start=time.perf_counter_ns()
    for _ in range(3):fn(*args)
    samples[v].append((time.perf_counter_ns()-start)/3000)
  row['samples_us']=samples;row['median_us']={v:statistics.median(s) for v,s in samples.items()};row['speedup']=row['median_us']['native']/row['median_us']['optimized'];rows.append(row)
  Path('research/results/rotary-projection-study.json').write_text(json.dumps({'rows':rows},indent=2));print(name,'PASS','placements',row['placements'],'speedup=%.3f'%row['speedup'],flush=True)
 print('Completed',len(rows),flush=True)
if __name__=='__main__':main()
