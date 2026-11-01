"""Validate the refined ratio4 cost gate and preserve ratio2 timing artifacts."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from study_cache import *
from add_patch_variant import patch

def main():
 rows=[]
 for row in json.loads(Path('research2/results/cache-study.json').read_text())['rows']:
  w=Path('research2/results/study')/row['case'];path=w/'ratio4.mlir';patch(w/'lifted-normalized.mlir',path,w/'ratio4.log',ratio=4)
  before=(w/'patch-forwarded.mlir').read_text();after=path.read_text();entry={'case':row['case'],'ratio4_patches':after.count('cache_gather_forwarded'),'ratio2_patches':before.count('cache_gather_forwarded'),'identical_to_ratio2_ir':before==after}
  if not entry['identical_to_ratio2_ir']:
   flags=['--iree-hal-target-backends=llvm-cpu','--iree-llvmcpu-target-cpu=host',*STRONG,'--iree-dispatch-creation-enable-aggressive-fusion']
   run(['iree-compile',path,*flags,'-o',w/'ratio4.vmfb'],w/'ratio4-compile.log');cfg,ctx,fn,args=load(w/'ratio4.vmfb',inputs(row['shape']));y=fn(*args);ys=list(y) if not hasattr(y,'to_host') else [y];refs=reference(BranchAttention(row['shape']),inputs(row['shape']));assert len(ys)==len(refs)
   for a,b in zip(ys,refs):np.testing.assert_allclose(a.to_host(),b,atol=4e-5,rtol=1e-3)
   entry['correct']=True;entry['binary_identical_to_fused_baseline']=(w/'ratio4.vmfb').read_bytes()==(w/'lifted-normalized.vmfb').read_bytes()
  rows.append(entry)
 Path('research2/results/cost-gate-validation.json').write_text(json.dumps(rows,indent=2));print(rows,flush=True)
if __name__=='__main__':main()
