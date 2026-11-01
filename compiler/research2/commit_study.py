"""Check compiler obligations when an updated cache version escapes the block."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from study_cache import *
from add_patch_variant import patch

def main():
 torch.set_num_threads(1);rows=[]
 for mode,case in [('one',(16384,128,2,4,32,1,4,37,2)),('all',CASES[0])]:
  w=Path('research2/results/commit')/mode;w.mkdir(parents=True,exist_ok=True);x=inputs(case);model=BranchAttention(case,returned='one' if mode=='one' else True);refs=reference(model,x)
  aot.export(torch.export.export(model,x)).save_mlir(str(w/'torch.mlir'));run(['iree-opt',w/'torch.mlir','--torch-function-to-torch-backend-pipeline','--torch-backend-to-linalg-on-tensors-backend-pipeline','--canonicalize','-o',w/'baseline.mlir'],w/'lower.log');lift(w/'baseline.mlir',w/'lifted.mlir');normalize(w/'lifted.mlir',w/'lifted-normalized.mlir',w/'normalize.log');forward(w/'baseline.mlir',w/'forwarded.mlir',w/'forward.log');patch(w/'lifted-normalized.mlir',w/'patch-forwarded.mlir',w/'patch.log')
  row={'case':mode,'shape':case,'returned_cache_mode':mode,'variants':{}}
  for v,source in [('strong','baseline'),('lifted-normalized','lifted-normalized'),('forwarded','forwarded'),('patch-forwarded','patch-forwarded')]:
   flags=['--iree-hal-target-backends=llvm-cpu','--iree-llvmcpu-target-cpu=host',*STRONG];cs=run(['iree-compile',w/(source+'.mlir'),*flags,'-o',w/(v+'.vmfb')],w/(v+'-compile.log'));run(['iree-compile',w/(source+'.mlir'),*flags,'--compile-to=hal','-o',w/(v+'-hal.mlir')],w/(v+'-hal.log'))
   cfg,ctx,fn,args=load(w/(v+'.vmfb'),[a.clone() for a in x]);before=cfg.device.allocator.statistics;ys=fn(*args);after=cfg.device.allocator.statistics;error=0
   for i,(a,b) in enumerate(zip(ys,refs)):
    y=a.to_host();np.testing.assert_allclose(y,b,atol=4e-5,rtol=1e-3);error=max(error,float(np.max(np.abs(y-b))))
    if i>=case[6]:assert y.tobytes()==b.astype(np.float32).tobytes()
   text=(w/(v+'-hal.mlir')).read_text();row['variants'][v]={'compiled':True,'correct':True,'returned_caches_bitwise_equal':True,'max_abs':error,'compile_s':cs,'allocated_bytes':after['device_bytes_allocated']-before['device_bytes_allocated'],'dispatches':text.count('hal.command_buffer.dispatch'),'copy_command_bytes':copy_stats(text),'patches':(w/(source+'.mlir')).read_text().count('cache_gather_forwarded'),'vmfb_sha256':hashlib.sha256((w/(v+'.vmfb')).read_bytes()).hexdigest()}
  rows.append(row);Path('research2/results/commit-study.json').write_text(json.dumps(rows,indent=2));print(mode,{v:(r['allocated_bytes'],r['patches']) for v,r in row['variants'].items()},flush=True)
if __name__=='__main__':main()
