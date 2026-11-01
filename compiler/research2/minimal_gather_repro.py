"""Standalone native IREE counterexample; accepts a patched source compiler."""
import sys,argparse,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from study_cache import ROOT,run,normalize,STRONG
import iree.runtime as rt,numpy as np

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--compiler',default='iree-compile');ap.add_argument('--tag',default='wheel');args=ap.parse_args();w=Path('research2/results/minimal-gather')/args.tag;w.mkdir(parents=True,exist_ok=True)
 source=Path('research2/patches/minimal_gather_attention.mlir');normalize(source,w/'normalized.mlir',w/'normalize.log');out={}
 q=np.zeros((2,1,32),dtype=np.float32);k=np.zeros((257,2,32),dtype=np.float32);v=np.broadcast_to(np.arange(514,dtype=np.float32).reshape(257,2,1),(257,2,32)).copy();ids=(np.arange(128,dtype=np.int64)*2).reshape(128,1)
 for variant,p in [('original',source),('normalized',w/'normalized.mlir')]:
  run([args.compiler,p,'--iree-hal-target-backends=llvm-cpu','--iree-llvmcpu-target-cpu=host',*STRONG,'-o',w/(variant+'.vmfb')],w/(variant+'-compile.log'))
  cfg=rt.Config('local-sync');ctx=rt.SystemContext(config=cfg);ctx.add_vm_module(rt.VmModule.copy_buffer(cfg.vm_instance,(w/(variant+'.vmfb')).read_bytes()));y=ctx.modules.module.main(*[rt.asdevicearray(cfg.device,a) for a in [q,k,v,ids]]).to_host()
  out[variant]={'first':y[:,0,0].tolist(),'max_abs':float(np.max(np.abs(y-np.array([254,255],dtype=np.float32).reshape(2,1,1)))),'correct':bool(np.allclose(y,np.array([254,255]).reshape(2,1,1),atol=1e-5))}
 print(out,flush=True);(w/'result.json').write_text(json.dumps(out,indent=2));assert out['normalized']['correct']
if __name__=='__main__':main()
