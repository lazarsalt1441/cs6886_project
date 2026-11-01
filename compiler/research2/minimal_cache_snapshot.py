"""Self-contained snapshot counterexample and executable C++ repairs."""
import sys,json,argparse,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from study_cache import run,STRONG
from snapshot_study import protect
from repair_snapshot_cow import compile_repaired
import numpy as np,iree.runtime as rt

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--compiler',default='iree-compile');ap.add_argument('--repair',choices=['none','early','cow'],default='none');ap.add_argument('--expect-correct',action='store_true');ap.add_argument('--tag',default='wheel');a=ap.parse_args();w=Path('research2/results/minimal-snapshot')/(a.tag+'-'+a.repair);w.mkdir(parents=True,exist_ok=True);source=Path('research2/patches/minimal_cache_snapshot.mlir')
 if a.repair=='cow':compile_repaired(source,w/'module.vmfb')
 else:
  if a.repair=='early':protect(source,w/'protected.mlir',w/'protect.log');source=w/'protected.mlir'
  run([a.compiler,source,'--iree-hal-target-backends=llvm-cpu','--iree-llvmcpu-target-cpu=host',*STRONG,'-o',w/'module.vmfb'],w/'compile.log')
 cfg=rt.Config('local-sync');ctx=rt.SystemContext(config=cfg);ctx.add_vm_module(rt.VmModule.copy_buffer(cfg.vm_instance,(w/'module.vmfb').read_bytes()));x=[np.arange(64,dtype=np.float32),np.array([101,102],np.float32),np.array([201,202],np.float32)];dev=[rt.asdevicearray(cfg.device,z.copy()) for z in x];ys=ctx.modules.module.main(*dev);updated=x[0].copy();updated[7:9]=x[1];updated[9:11]=x[2];result={'compiler':a.compiler,'repair':a.repair,'updated_correct':ys[0].to_host().tobytes()==updated.tobytes(),'snapshot_correct':ys[1].to_host().tobytes()==x[0].tobytes(),'observed_snapshot':ys[1].to_host()[7:11].tolist(),'expected_snapshot':[7,8,9,10],'vmfb_sha256':hashlib.sha256((w/'module.vmfb').read_bytes()).hexdigest()};(w/'result.json').write_text(json.dumps(result,indent=2));print(result)
 assert result['updated_correct']
 if a.expect_correct or a.repair!='none':assert result['snapshot_correct']
if __name__=='__main__':main()
