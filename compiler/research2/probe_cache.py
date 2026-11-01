"""Probe actual frontend cache update + read dataflow before designing a pass."""
import sys,os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT);sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'research')]
import torch,numpy as np
from iree.turbine import aot
from run_suite import run,load,allocations
STRONG=['--iree-opt-level=O3','--iree-opt-generalize-matmul','--iree-dispatch-creation-enable-aggressive-reshape-movement']
class CacheRead(torch.nn.Module):
 def __init__(self,returned=False):super().__init__();self.returned=returned
 def forward(self,q,k,v,nk,nv,ids):
  k=torch.slice_scatter(k,nk,dim=0,start=37,end=39)
  v=torch.slice_scatter(v,nv,dim=0,start=37,end=39)
  rk=k.index_select(0,ids).permute(1,0,2).unsqueeze(0)
  rv=v.index_select(0,ids).permute(1,0,2).unsqueeze(0)
  out=torch.nn.functional.scaled_dot_product_attention(q,rk,rv,enable_gqa=True)
  return (out,k,v) if self.returned else out

def inputs(pool=4096,n=128,h=2,g=4,d=32):
 torch.manual_seed(155)
 q=torch.randn(1,h*g,1,d);k=torch.randn(pool,h,d);v=torch.randn_like(k);nk=torch.randn(2,h,d);nv=torch.randn_like(nk)
 ids=torch.cat([torch.tensor([37,38,0,pool-1]),torch.randint(pool,(n-4,))]);return(q,k,v,nk,nv,ids)
def main():
 torch.set_num_threads(1);rows=[]
 for returned in [False,True]:
  name='return-cache' if returned else 'attention-only';w=Path('research2/results/probe')/name;w.mkdir(parents=True,exist_ok=True);model=CacheRead(returned);x=inputs()
  aot.export(torch.export.export(model,x)).save_mlir(str(w/'torch.mlir'))
  run(['iree-opt',w/'torch.mlir','--torch-function-to-torch-backend-pipeline','--torch-backend-to-linalg-on-tensors-backend-pipeline','--canonicalize','-o',w/'baseline.mlir'],w/'lower.log')
  row={'case':name,'variants':{}}
  for mode,extra in [('default',[]),('strong',STRONG),('aggressive',STRONG+['--iree-dispatch-creation-enable-aggressive-fusion'])]:
   flags=['--iree-hal-target-backends=llvm-cpu','--iree-llvmcpu-target-cpu=host',*extra]
   run(['iree-compile',w/'baseline.mlir',*flags,'-o',w/(mode+'.vmfb')],w/(mode+'-compile.log'))
   run(['iree-compile',w/'baseline.mlir',*flags,'--compile-to=hal','-o',w/(mode+'-hal.mlir')],w/(mode+'-hal.log'))
   cfg,ctx,fn,dev=load(w/(mode+'.vmfb'),x);before=cfg.device.allocator.statistics;y=fn(*dev);after=cfg.device.allocator.statistics
   actual=y[0].to_host() if returned else y.to_host()
   with torch.inference_mode():ref=model(*[a.double() if a.dtype==torch.float32 else a for a in x]);ref=(ref[0] if returned else ref).numpy()
   np.testing.assert_allclose(actual,ref,atol=4e-5,rtol=1e-3)
   hal=(w/(mode+'-hal.mlir')).read_text();row['variants'][mode]={'alloc_bytes':after['device_bytes_allocated']-before['device_bytes_allocated'],'hal_allocs':allocations(hal),'dispatches':hal.count('hal.command_buffer.dispatch'),'max_abs':float(np.max(np.abs(actual-ref)))}
  rows.append(row);print(row,flush=True)
 Path('research2/results/cache-probe.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
