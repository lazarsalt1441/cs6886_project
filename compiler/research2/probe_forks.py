import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from probe_cache import *
class Forked(torch.nn.Module):
 def forward(self,q,k,v,nk,nv,ids):
  a=CacheRead()(q,k,v,nk,nv,ids)
  b=CacheRead()(q,k,v,-nk,-nv,ids)
  return a,b

def main():
 torch.set_num_threads(1);w=Path('research2/results/forks-probe');w.mkdir(exist_ok=True);x=inputs();model=Forked()
 with torch.inference_mode():ref=[y.numpy() for y in model(*[a.double() if a.is_floating_point() else a for a in x])]
 aot.export(torch.export.export(model,x)).save_mlir(str(w/'torch.mlir'))
 run(['iree-opt',w/'torch.mlir','--torch-function-to-torch-backend-pipeline','--torch-backend-to-linalg-on-tensors-backend-pipeline','--canonicalize','-o',w/'baseline.mlir'],w/'lower.log')
 out={}
 for mode,extra in [('strong',STRONG),('aggressive',STRONG+['--iree-dispatch-creation-enable-aggressive-fusion'])]:
  flags=['--iree-hal-target-backends=llvm-cpu','--iree-llvmcpu-target-cpu=host',*extra]
  run(['iree-compile',w/'baseline.mlir',*flags,'-o',w/(mode+'.vmfb')],w/(mode+'-compile.log'))
  run(['iree-compile',w/'baseline.mlir',*flags,'--compile-to=hal','-o',w/(mode+'-hal.mlir')],w/(mode+'-hal.log'))
  cfg,ctx,fn,dev=load(w/(mode+'.vmfb'),x);before=cfg.device.allocator.statistics;y=fn(*dev);after=cfg.device.allocator.statistics
  errors=[]
  for yy,rr in zip(y,ref):actual=yy.to_host();np.testing.assert_allclose(actual,rr,atol=4e-5,rtol=1e-3);errors.append(float(np.max(np.abs(actual-rr))))
  hal=(w/(mode+'-hal.mlir')).read_text();out[mode]={'alloc_bytes':after['device_bytes_allocated']-before['device_bytes_allocated'],'hal_allocs':allocations(hal),'dispatches':hal.count('hal.command_buffer.dispatch'),'max_abs':errors}
 print(out,flush=True);Path('research2/results/forks-probe.json').write_text(json.dumps(out,indent=2))
if __name__=='__main__':main()
