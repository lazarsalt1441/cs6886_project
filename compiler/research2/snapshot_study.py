"""Reproduce and repair original-cache snapshot corruption in stock IREE."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from study_cache import *
from test_update_coalescing import fixture

def protect(source,dest,log):
 with ir.Context():mod=ir.Module.parse(Path(source).read_text());temp=Path(dest).with_name(Path(dest).stem+'-input.mlir');temp.write_text(mod.operation.get_asm(print_generic_op_form=True))
 return run(['build/gqa-opt',temp,'--allow-unregistered-dialect','--attention-protect-cache-snapshots','--canonicalize','-o',dest],log)
class SnapshotAttention(torch.nn.Module):
 def __init__(self,case):super().__init__();self.model=BranchAttention(case,returned='one')
 def forward(self,q,k,v,nk,nv,ids):return (*self.model(q,k,v,nk,nv,ids),k,v)

def main():
 torch.set_num_threads(1);case=list(CASES[0]);case[6]=1;w=Path('research2/results/snapshot-attention/single');w.mkdir(parents=True,exist_ok=True);model=SnapshotAttention(case);x=inputs(case)
 aot.export(torch.export.export(model,x)).save_mlir(str(w/'torch.mlir'));run(['iree-opt',w/'torch.mlir','--torch-function-to-torch-backend-pipeline','--torch-backend-to-linalg-on-tensors-backend-pipeline','--canonicalize','-o',w/'baseline.mlir'],w/'lower.log');protect(w/'baseline.mlir',w/'protected.mlir',w/'protect.log');assert (w/'protected.mlir').read_text().count('cache_snapshot_guard')==2
 rows={}
 for v in ['baseline','protected']:
  flags=['--iree-hal-target-backends=llvm-cpu','--iree-llvmcpu-target-cpu=host',*STRONG];run(['iree-compile',w/(v+'.mlir'),*flags,'-o',w/(v+'.vmfb')],w/(v+'-compile.log'));run(['iree-compile',w/(v+'.mlir'),*flags,'--compile-to=hal','-o',w/(v+'-hal.mlir')],w/(v+'-hal.log'));checks=[]
  for seed in [171,613,811]:
   x=inputs(case,seed);refs=reference(model,x)
   with torch.inference_mode():exact=model(*x)
   cfg,ctx,fn,dev=load(w/(v+'.vmfb'),[a.clone() for a in x]);before=cfg.device.allocator.statistics;ys=fn(*dev);after=cfg.device.allocator.statistics;assert len(ys)==case[6]+4
   floats=[];bits=[]
   for i,(actual,ref) in enumerate(zip(ys,refs)):
    arr=actual.to_host()
    if i<case[6]:np.testing.assert_allclose(arr,ref,atol=4e-5,rtol=1e-3);floats.append(float(np.max(np.abs(arr-ref))))
    else:bits.append(arr.tobytes()==exact[i].numpy().tobytes())
   check={'seed':seed,'attention_max_abs':max(floats),'cache_bitwise_equal':bits,'correct':all(bits)};checks.append(check)
   if v=='protected':assert check['correct'],check
  rows[v]={'checks':checks,'allocated_bytes':after['device_bytes_allocated']-before['device_bytes_allocated'],'vmfb_sha256':hashlib.sha256((w/(v+'.vmfb')).read_bytes()).hexdigest()}
 Path('research2/results/snapshot-attention.json').write_text(json.dumps(rows,indent=2));print(rows,flush=True);assert any(not c['correct'] for c in rows['baseline']['checks'])
if __name__=='__main__':main()
