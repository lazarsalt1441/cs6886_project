"""Check the high-level PyTorch SDPA alternative; do not assume all GQA lowers alike."""
from pathlib import Path
import json,torch,numpy as np
from iree.turbine import aot
from run_suite import GQA,inputs,run,load,bench
class NativeGQA(torch.nn.Module):
 def forward(self,q,k,v):
  return torch.nn.functional.scaled_dot_product_attention(q,k,v,enable_gqa=True)
def main():
 torch.set_num_threads(1);x=inputs();w=Path('results/native-gqa');w.mkdir(exist_ok=True)
 result={};model=NativeGQA()
 with torch.inference_mode():
  ref=GQA()(*x).numpy();v=model(*x).numpy();result['torch_max_abs']=float(np.max(np.abs(v-ref)))
  result['torch_native_samples_us']=bench(lambda:model(*x),21,100)
 aot.export(torch.export.export(model,x)).save_mlir(str(w/'torch.mlir'))
 try:
  run(['iree-opt',w/'torch.mlir','--torch-function-to-torch-backend-pipeline','--torch-backend-to-linalg-on-tensors-backend-pipeline','--canonicalize','-o',w/'linalg.mlir'],w/'lower.log')
  run(['iree-compile',w/'torch.mlir','--iree-hal-target-backends=llvm-cpu','--iree-llvmcpu-target-cpu=host','--iree-hal-dump-executable-files-to='+str(w/'dumps'),'-o',w/'native.vmfb'],w/'compile.log')
  cfg,ctx,fn,args=load(w/'native.vmfb',x);v=fn(*args).to_host();np.testing.assert_allclose(v,ref,atol=2e-5,rtol=2e-4)
  result['iree_max_abs']=float(np.max(np.abs(v-ref)));result['iree_samples_us']=bench(lambda:fn(*args),21,100);result['status']='PASS'
 except Exception as e:result.update(status='BLOCKED',error=str(e))
 (w/'result.json').write_text(json.dumps(result,indent=2));print(result['status'])
if __name__=='__main__':main()
