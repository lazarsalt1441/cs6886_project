import sys,torch,numpy as np,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from projection_study import ROOT
from run_suite import load

def main():
 torch.set_num_threads(1);torch.manual_seed(77);h,m,n,d=1,1,128,256
 x=(torch.randn(1,h,m,d),torch.randn(1,h,n,d),torch.randn(1,h,n,d))
 w=Path('research/results/backend-audit/h1_m1_n128_d256_alias0');cfg,ctx,fn,args=load(Path('research/results/backend-bug.vmfb'),x);y=fn(*args).to_host();out=[]
 for start,end in [(0,256),(0,128),(128,256),(0,64),(192,256)]:
  with torch.inference_mode():r=torch.nn.functional.scaled_dot_product_attention(x[0][...,start:end].double(),x[1][...,start:end].double(),x[2].double(),scale=d**-.5).numpy()
  out.append(dict(slice=[start,end],max_abs=float(np.max(np.abs(y-r)))))
 for chunk in [16,32,64,128]:
  with torch.inference_mode():
   q=x[0].double().reshape(1,1,1,d//chunk,chunk)
   k=x[1].double().reshape(1,1,n,d//chunk,chunk)
   score=torch.einsum('bhmct,bhnct->bhmcn',q,k)*(d**-.5)
   weights=score.flatten(-2).softmax(-1).reshape(1,1,1,d//chunk,n)
   r=torch.einsum('bhmcn,bhnv->bhmv',weights,x[2].double()).numpy()
  out.append(dict(wrong_partitioned_softmax_chunk=chunk,max_abs=float(np.max(np.abs(y-r)))))
 print(out);Path('research/results/backend-localization.json').write_text(json.dumps(out,indent=2))
if __name__=='__main__':main()
