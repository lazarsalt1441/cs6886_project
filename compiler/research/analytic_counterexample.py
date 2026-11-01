"""A seed-free 2-token counterexample with exact real outputs 0.9 and 0.6."""
import sys,json,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from projection_study import ROOT
from cpu_compile import compile_guarded
from native_attention_audit import Attention
from run_suite import load
from iree.turbine import aot
import torch,numpy as np

def main():
 torch.set_num_threads(1);d=256;q=torch.zeros(1,1,1,d);q[...,0]=math.sqrt(d)*math.log(9)
 k=torch.zeros(1,1,2,d);k[...,0,0]=1
 v=torch.zeros(1,1,2,d);v[...,0,:]=1;x=(q,k,v)
 w=Path('research/results/analytic-counterexample');w.mkdir(exist_ok=True)
 aot.export(torch.export.export(Attention(False),x)).save_mlir(str(w/'torch.mlir'))
 compile_guarded(w/'torch.mlir',w/'guarded.vmfb',w,'analytic',baseline_destination=w/'original.vmfb')
 out={}
 for variant in ['original','guarded']:
  cfg,ctx,fn,args=load(w/(variant+'.vmfb'),x);y=fn(*args).to_host()
  out[variant]=dict(min=float(y.min()),max=float(y.max()),first=float(y.flat[0]),error_from_correct_0_9=float(np.max(np.abs(y.astype(np.float64)-.9))))
 out['correct_real']=.9;out['wrong_feature_partition_real']=.6
 Path('research/results/analytic-counterexample.json').write_text(json.dumps(out,indent=2));print(out)
 assert abs(out['guarded']['first']-.9)<1e-5
if __name__=='__main__':main()
