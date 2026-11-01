"""Independent CPU backend audit: no custom compiler pass is used."""
import sys,os,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from projection_study import ROOT,STRONG
from run_suite import run,load
import torch,numpy as np
from iree.turbine import aot
class Attention(torch.nn.Module):
 def __init__(self,alias):super().__init__();self.alias=alias
 def forward(self,q,k,v):
  return torch.nn.functional.scaled_dot_product_attention(q,k,k if self.alias else v,enable_gqa=True)
def main():
 torch.set_num_threads(1);os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
 cases=[(1,1,32,32),(1,1,128,128),(1,1,128,256),(1,1,128,512),(8,1,128,512),(8,1,2048,128),(8,1,2048,256),(8,1,2048,512),(1,16,256,512),(1,1,128,768)]
 rows=[]
 for h,m,n,d in cases:
  for alias in [False,True]:
   torch.manual_seed(77);x=(torch.randn(1,h,m,d),torch.randn(1,h,n,d),torch.randn(1,h,n,d));model=Attention(alias)
   with torch.inference_mode():ref=model(*[a.double() for a in x]).numpy()
   name=f'h{h}_m{m}_n{n}_d{d}_alias{int(alias)}';w=Path('research/results/backend-audit')/name;w.mkdir(parents=True,exist_ok=True)
   aot.export(torch.export.export(model,x)).save_mlir(str(w/'torch.mlir'))
   try:
    run(['iree-compile',w/'torch.mlir','--iree-hal-target-backends=llvm-cpu','--iree-llvmcpu-target-cpu=host',*STRONG,'-o',w/'model.vmfb'],w/'compile.log')
    # Export may remove unused input v in the alias case, but torch exporter
    # normally preserves public input signature. Inspect and use actual arity.
    cfg,ctx,fn,args=load(w/'model.vmfb',x);y=fn(*args).to_host();error=float(np.max(np.abs(y-ref)))
    valid=bool(np.allclose(y,ref,atol=4e-5,rtol=1e-3));row=dict(case=name,shape=[h,m,n,d],alias=alias,max_abs=error,correct=valid)
   except Exception as e:row=dict(case=name,status='ERROR',error=str(e))
   rows.append(row);Path('research/results/backend-audit.json').write_text(json.dumps({'compiler':'89ece87b89852100c0a4ac3d515e14842d090967','custom_pass_used':False,'rows':rows},indent=2));print(row,flush=True)
 print('Completed',len(rows),flush=True)
if __name__=='__main__':main()
