import sys,json,torch,numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from native_projection_study import NativeProjected
from projection_study import create_inputs,STRONG
from run_suite import load,run

def main():
 torch.set_num_threads(1);case=(8,1,2048,512,128,128,True,False);x=create_inputs(case)
 with torch.inference_mode():
  ref=NativeProjected(True)(*[a.double() for a in x]).numpy()
  q,c,wk,wv=x;c=c.expand(8,-1,-1);qa=torch.bmm(q,wk.transpose(1,2))*2
  latent=torch.nn.functional.scaled_dot_product_attention(qa.unsqueeze(0),c.unsqueeze(0),c.unsqueeze(0),enable_gqa=True).squeeze(0)
  manual=torch.bmm(latent,wv).numpy();print('torch absorption error',np.max(np.abs(manual-ref)),flush=True)
 w=Path('research/results/native-projections/h8_m1_n2048_l512_d128_v128_s1');results=[]
 for name,flags in [('o0',[]),('o3',['--iree-opt-level=O3']),('strong-no-v-transpose',STRONG+['--iree-global-opt-enable-attention-v-transpose=false'])]:
  path=w/('debug-'+name+'.vmfb');run(['iree-compile',w/'optimized.mlir','--iree-hal-target-backends=llvm-cpu','--iree-llvmcpu-target-cpu=host',*flags,'-o',path],w/('debug-'+name+'.log'))
  cfg,ctx,fn,args=load(path,x);y=fn(*args).to_host();err=float(np.max(np.abs(y-ref)));print(name,err,flush=True);results.append(dict(config=name,max_abs=err))
 Path('research/results/native-mismatch-debug.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()
