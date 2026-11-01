"""Exercise the common unsqueeze/expand/reshape frontend spelling of GQA."""
from pathlib import Path
import json,torch,numpy as np
from iree.turbine import aot
from run_suite import inputs,GQA,run,load
class BroadcastGQA(torch.nn.Module):
 def forward(self,q,k,v):
  b,hq,m,d=q.shape;hkv=k.shape[1];n=k.shape[2];r=hq//hkv
  k=k[:,:,None,:,:].expand(b,hkv,r,n,d).reshape(b,hq,n,d)
  v=v[:,:,None,:,:].expand(b,hkv,r,n,d).reshape(b,hq,n,d)
  return (q@k.transpose(-2,-1)*d**-0.5).softmax(-1)@v

def main():
 torch.set_num_threads(1);x=inputs();w=Path('results/broadcast-frontend');w.mkdir(exist_ok=True)
 aot.export(torch.export.export(BroadcastGQA(),x)).save_mlir(str(w/'torch.mlir'))
 run(['iree-opt',w/'torch.mlir','--torch-function-to-torch-backend-pipeline','--torch-backend-to-linalg-on-tensors-backend-pipeline','--canonicalize','-o',w/'baseline.mlir'],w/'lower.log')
 run(['build/gqa-opt',w/'baseline.mlir','--gqa-head-reuse','--canonicalize','-o',w/'optimized.mlir'],w/'pass.log')
 before=(w/'baseline.mlir').read_text();after=(w/'optimized.mlir').read_text()
 removed=before.count('linalg.batch_matmul')-after.count('linalg.batch_matmul')
 run(['iree-compile',w/'optimized.mlir','--iree-hal-target-backends=llvm-cpu','--iree-llvmcpu-target-cpu=host','-o',w/'optimized.vmfb'],w/'compile.log')
 cfg,ctx,fn,args=load(w/'optimized.vmfb',x);y=fn(*args).to_host()
 with torch.inference_mode():ref=GQA()(*x).numpy()
 np.testing.assert_allclose(y,ref,atol=2e-5,rtol=2e-4)
 result=dict(contractions_rewritten=removed,max_abs=float(np.max(np.abs(y-ref))),status='PASS')
 (w/'result.json').write_text(json.dumps(result,indent=2));print(result)
if __name__=='__main__':main()
