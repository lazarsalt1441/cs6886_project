"""Compare two correct repairs: guarded online schedule vs score materialization."""
import sys,os,json,statistics,time,random
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from projection_study import ROOT,STRONG
from native_attention_audit import Attention
from cpu_compile import compile_guarded
from run_suite import run,load
from iree.turbine import aot
from iree.compiler import ir
import torch,numpy as np

def main():
 torch.set_num_threads(1);os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});rows=[]
 cases=[(1,1,128,256),(1,16,256,512),(1,128,2048,256)]
 for h,m,n,d in cases:
  name=f'h{h}_m{m}_n{n}_d{d}';w=Path('research/results/fallback-comparison')/name;w.mkdir(parents=True,exist_ok=True)
  torch.manual_seed(88);x=(torch.randn(1,h,m,d),torch.randn(1,h,n,d),torch.randn(1,h,n,d))
  model=Attention(False)
  with torch.inference_mode():ref=model(*[a.double() for a in x]).numpy()
  aot.export(torch.export.export(model,x)).save_mlir(str(w/'torch.mlir'))
  run(['iree-opt',w/'torch.mlir','--torch-function-to-torch-backend-pipeline','--torch-backend-to-linalg-on-tensors-backend-pipeline','--canonicalize','-o',w/'linalg.mlir'],w/'lower.log')
  with ir.Context():module=ir.Module.parse((w/'linalg.mlir').read_text());(w/'generic.mlir').write_text(module.operation.get_asm(print_generic_op_form=True))
  run(['build/gqa-opt',w/'generic.mlir','--allow-unregistered-dialect','--attention-cpu-safe-lowering','--canonicalize','-o',w/'fallback.mlir'],w/'fallback-pass.log')
  compile_guarded(w/'torch.mlir',w/'guarded.vmfb',w,'online')
  run(['iree-compile',w/'fallback.mlir','--iree-hal-target-backends=llvm-cpu','--iree-llvmcpu-target-cpu=host',*STRONG,'-o',w/'fallback.vmfb'],w/'fallback-compile.log')
  row=dict(case=name,variants={});loaded={}
  for v in ['guarded','fallback']:
   cfg,ctx,fn,args=load(w/(v+'.vmfb'),x);before=cfg.device.allocator.statistics;y=fn(*args).to_host();after=cfg.device.allocator.statistics
   np.testing.assert_allclose(y,ref,atol=5e-5,rtol=2e-3);row['variants'][v]=dict(max_abs=float(np.max(np.abs(y-ref))),allocated_bytes=after['device_bytes_allocated']-before['device_bytes_allocated']);loaded[v]=(cfg,ctx,fn,args)
  samples={v:[] for v in loaded};rng=random.Random(9)
  for _,_,fn,args in loaded.values():
   for _ in range(5):fn(*args)
  for _ in range(21):
   order=list(loaded);rng.shuffle(order)
   for v in order:
    _,_,fn,args=loaded[v];start=time.perf_counter_ns();fn(*args);samples[v].append((time.perf_counter_ns()-start)/1000)
  row['samples_us']=samples;row['median_us']={v:statistics.median(s) for v,s in samples.items()};rows.append(row)
  Path('research/results/fallback-comparison.json').write_text(json.dumps(rows,indent=2));print(name,row['median_us'],flush=True)
 print('Completed',len(rows),flush=True)
if __name__=='__main__':main()
