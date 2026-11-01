"""Correctness plus copy elimination for ephemeral updates with old-cache outputs."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from study_cache import *
from snapshot_study import protect
from repair_snapshot_cow import compile_repaired
from add_patch_variant import patch
class SnapshotReadAttention(torch.nn.Module):
 def __init__(self,case):super().__init__();self.model=BranchAttention(case)
 def forward(self,q,k,v,nk,nv,ids):return (*self.model(q,k,v,nk,nv,ids),k,v)
CASE=(4096,128,2,4,32,1,1,37,2)
def main():
 torch.set_num_threads(1);w=Path('research2/results/snapshot-read');w.mkdir(parents=True,exist_ok=True);model=SnapshotReadAttention(CASE);x=inputs(CASE);aot.export(torch.export.export(model,x)).save_mlir(str(w/'torch.mlir'));run(['iree-opt',w/'torch.mlir','--torch-function-to-torch-backend-pipeline','--torch-backend-to-linalg-on-tensors-backend-pipeline','--canonicalize','-o',w/'baseline.mlir'],w/'lower.log');protect(w/'baseline.mlir',w/'protected.mlir',w/'protect.log');forward(w/'protected.mlir',w/'forwarded.mlir',w/'forward.log');lift(w/'protected.mlir',w/'lifted.mlir');normalize(w/'lifted.mlir',w/'normalized.mlir',w/'normalize.log');patch(w/'normalized.mlir',w/'masked.mlir',w/'masked.log',ratio=4)
 rows={}
 for v,source,extra in [('stock-diagnostic','baseline',[]),('protected','protected',[]),('aggressive','protected',['--iree-dispatch-creation-enable-aggressive-fusion']),('fused','normalized',[]),('fused-aggressive','normalized',['--iree-dispatch-creation-enable-aggressive-fusion']),('forwarded','forwarded',[]),('masked','masked',[]),('root-cow','baseline',[])]:
  flags=['--iree-hal-target-backends=llvm-cpu','--iree-llvmcpu-target-cpu=host',*STRONG,*extra]
  if v=='root-cow':compile_repaired(w/'baseline.mlir',w/(v+'.vmfb'))
  else:run(['iree-compile',w/(source+'.mlir'),*flags,'-o',w/(v+'.vmfb')],w/(v+'-compile.log'));run(['iree-compile',w/(source+'.mlir'),*flags,'--compile-to=hal','-o',w/(v+'-hal.mlir')],w/(v+'-hal.log'))
  checks=[]
  for seed in [171,613,811]:
   x=inputs(CASE,seed);refs=reference(model,x);cfg,ctx,fn,args=load(w/(v+'.vmfb'),[a.clone() for a in x]);before=cfg.device.allocator.statistics;ys=fn(*args);after=cfg.device.allocator.statistics;assert len(ys)==3;np.testing.assert_allclose(ys[0].to_host(),refs[0],atol=4e-5,rtol=1e-3);bits=all(ys[i].to_host().tobytes()==x[i].numpy().tobytes() for i in [1,2]);checks.append({'seed':seed,'snapshots_bitwise_equal':bits,'max_abs':float(np.max(np.abs(ys[0].to_host()-refs[0])))})
   if v!='stock-diagnostic':assert bits,(v,seed)
  rows[v]={'correct':all(c['snapshots_bitwise_equal'] for c in checks),'checks':checks,'allocated_bytes':after['device_bytes_allocated']-before['device_bytes_allocated'],'vmfb_sha256':hashlib.sha256((w/(v+'.vmfb')).read_bytes()).hexdigest()};Path('research2/results/snapshot-read-study.json').write_text(json.dumps({'case':CASE,'variants':rows},indent=2));print(v,rows[v]['correct'],rows[v]['allocated_bytes'],flush=True)
 assert not rows['stock-diagnostic']['correct']
if __name__=='__main__':main()
