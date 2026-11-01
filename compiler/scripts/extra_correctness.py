"""Masked attention and nonzero destination-seed regression tests."""
import json
from pathlib import Path
import numpy as np,torch
from iree.turbine import aot
from run_suite import inputs,run,load
class BiasedGQA(torch.nn.Module):
 def forward(self,q,k,v,bias):
  r=q.shape[1]//k.shape[1]
  score=q@k.repeat_interleave(r,1).transpose(-2,-1)*q.shape[-1]**-0.5
  return (score+bias).softmax(-1)@v.repeat_interleave(r,1)
def main():
 torch.set_num_threads(1);out=[]
 for b,hq,hkv,m,n,d in [(1,8,2,8,32,32),(2,6,2,7,37,24)]:
  x=inputs(b,hq,hkv,m,n,d)
  for kind in ['causal','padding']:
   bias=torch.zeros(1,1,m,n)
   if kind=='causal':bias.masked_fill_(torch.arange(n)[None,:]>torch.arange(m)[:,None],float('-inf'))
   else:bias[:,:,:,-n//3:]=float('-inf')
   args=(*x,bias);model=BiasedGQA();w=Path(f'results/extra/{kind}_b{b}_q{hq}');w.mkdir(parents=True,exist_ok=True)
   aot.export(torch.export.export(model,args)).save_mlir(str(w/'torch.mlir'))
   run(['iree-opt',w/'torch.mlir','--torch-function-to-torch-backend-pipeline','--torch-backend-to-linalg-on-tensors-backend-pipeline','--canonicalize','-o',w/'baseline.mlir'],w/'lower.log')
   run(['build/gqa-opt',w/'baseline.mlir','--gqa-head-reuse','--canonicalize','-o',w/'optimized.mlir'],w/'pass.log')
   with torch.inference_mode():ref=model(*args).numpy()
   for v in ['baseline','optimized']:
    run(['iree-compile',w/(v+'.mlir'),'--iree-hal-target-backends=llvm-cpu','--iree-llvmcpu-target-cpu=host','-o',w/(v+'.vmfb')],w/(v+'.log'))
    cfg,ctx,fn,dev=load(w/(v+'.vmfb'),args);y=fn(*dev).to_host()
    np.testing.assert_allclose(y,ref,atol=2e-5,rtol=2e-4);out.append(dict(case=w.name,variant=v,max_abs=float(np.max(np.abs(y-ref)))))
 # Preserve arbitrary destination init, rather than assuming zero-filled attention.
 s=Path('tests/dynamic.mlir').read_text().replace('func.func @dynamic','func.func @main').replace('?', '128').replace('tensor.empty(%n)','tensor.empty()')
 w=Path('results/extra/nonzero-init');w.mkdir(exist_ok=True);(w/'baseline.mlir').write_text(s)
 rng=np.random.default_rng(1);q=rng.standard_normal((8,1,128),dtype=np.float32);kv=rng.standard_normal((1,2,128,32),dtype=np.float32);init=rng.standard_normal((8,1,32),dtype=np.float32)
 ref=q@np.repeat(kv,4,axis=1).reshape(8,128,32)+init
 run(['build/gqa-opt',w/'baseline.mlir','--gqa-head-reuse','--canonicalize','-o',w/'optimized.mlir'],w/'pass.log')
 run(['iree-compile',w/'optimized.mlir','--iree-hal-target-backends=llvm-cpu','--iree-llvmcpu-target-cpu=host','-o',w/'optimized.vmfb'],w/'compile.log')
 cfg,ctx,fn,dev=load(w/'optimized.vmfb',[torch.from_numpy(a) for a in [q,kv,init]]);y=fn(*dev).to_host()
 np.testing.assert_allclose(y,ref,atol=2e-5,rtol=2e-4);out.append(dict(case='nonzero-init',variant='optimized',max_abs=float(np.max(np.abs(y-ref)))))
 Path('results/extra-correctness.json').write_text(json.dumps(out,indent=2));print('PASS',len(out),'comparisons')
if __name__=='__main__':main()
