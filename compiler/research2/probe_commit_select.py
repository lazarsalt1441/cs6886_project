"""Inspect actual lowering of runtime winner cache selection."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from study_cache import *
class WinnerAttention(torch.nn.Module):
 def __init__(self,case):super().__init__();self.case=case
 def forward(self,qs,k,v,nks,nvs,ids):
  pool,n,h,g,d,m,f,p,t=self.case;outs=[];ks=[];vs=[]
  for b in range(f):
   kb=torch.slice_scatter(k,nks[b],dim=0,start=p,end=p+t);vb=torch.slice_scatter(v,nvs[b],dim=0,start=p,end=p+t);ks.append(kb);vs.append(vb)
   kk=kb.index_select(0,ids).permute(1,0,2).unsqueeze(0);vv=vb.index_select(0,ids).permute(1,0,2).unsqueeze(0);outs.append(torch.nn.functional.scaled_dot_product_attention(qs[b],kk,vv,enable_gqa=True))
  winner=outs[0].sum()>outs[1].sum()
  return *outs,torch.where(winner,ks[0],ks[1]),torch.where(winner,vs[0],vs[1]),winner

def main():
 case=(16384,128,2,4,32,1,2,37,2);w=Path('research2/results/winner');w.mkdir(parents=True,exist_ok=True);aot.export(torch.export.export(WinnerAttention(case),inputs(case))).save_mlir(str(w/'torch.mlir'));run(['iree-opt',w/'torch.mlir','--torch-function-to-torch-backend-pipeline','--torch-backend-to-linalg-on-tensors-backend-pipeline','--canonicalize','-o',w/'baseline.mlir'],w/'lower.log')
if __name__=='__main__':main()
