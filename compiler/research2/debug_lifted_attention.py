"""Determine whether a lifted-gather comparator failure is in helper or backend."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from probe_cache import *
from lift_gather_baseline import lift
class Read(torch.nn.Module):
 def __init__(self,mode):super().__init__();self.mode=mode
 def forward(self,q,k,v,ids):
  kk=k.index_select(0,ids);vv=v.index_select(0,ids)
  if self.mode=='gather-only':return kk,vv
  out=torch.nn.functional.scaled_dot_product_attention(q,kk.permute(1,0,2).unsqueeze(0),vv.permute(1,0,2).unsqueeze(0),enable_gqa=True)
  return (out,kk,vv) if self.mode=='observed-gathers' else out

def main():
 torch.set_num_threads(1);out=[]
 for h in [1,2]:
  for mode in ['gather-only','attention','observed-gathers']:
   q=torch.zeros(1,h,1,8);k=torch.zeros(3,h,8);v=torch.arange(3*h,dtype=torch.float32).reshape(3,h,1).expand(3,h,8).contiguous();ids=torch.tensor([2,0]);x=(q,k,v,ids);model=Read(mode)
   with torch.inference_mode():ref=model(*x);ref=[ref.numpy()] if isinstance(ref,torch.Tensor) else [a.numpy() for a in ref]
   w=Path('research2/results/debug-lifted')/(f'h{h}-'+mode);w.mkdir(parents=True,exist_ok=True)
   aot.export(torch.export.export(model,x)).save_mlir(str(w/'torch.mlir'))
   run(['iree-opt',w/'torch.mlir','--torch-function-to-torch-backend-pipeline','--torch-backend-to-linalg-on-tensors-backend-pipeline','--canonicalize','-o',w/'baseline.mlir'],w/'lower.log');lift(w/'baseline.mlir',w/'lifted.mlir')
   for variant,source,extra in [('original','baseline',STRONG),('lifted','lifted',STRONG),('no-vtranspose','lifted',STRONG+['--iree-global-opt-enable-attention-v-transpose=false'])]:
    run(['iree-compile',w/(source+'.mlir'),'--iree-hal-target-backends=llvm-cpu','--iree-llvmcpu-target-cpu=host',*extra,'-o',w/(variant+'.vmfb')],w/(variant+'-compile.log'))
    cfg,ctx,fn,dev=load(w/(variant+'.vmfb'),x);yy=fn(*dev);ys=[yy] if hasattr(yy,'to_host') else list(yy);actual=[a.to_host() for a in ys]
    row={'h':h,'mode':mode,'variant':variant,'correct':all(np.allclose(a,b,atol=1e-5) for a,b in zip(actual,ref)),'max_abs':[float(np.max(np.abs(a-b))) for a,b in zip(actual,ref)],'first_values':actual[0].reshape(-1,8)[:,0].tolist(),'expected_first':ref[0].reshape(-1,8)[:,0].tolist()};out.append(row);print(row,flush=True)
 Path('research2/results/debug-lifted-attention.json').write_text(json.dumps(out,indent=2))
if __name__=='__main__':main()
