"""Runtime attention-derived winner commit: factoring plus observation forwarding."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from study_cache import *
from add_patch_variant import patch
from probe_commit_select import WinnerAttention
CASE=(16384,128,2,4,32,1,2,37,2)
def factor(source,dest,log):
 with ir.Context():mod=ir.Module.parse(Path(source).read_text());temp=Path(dest).with_name(Path(dest).stem+'-input.mlir');temp.write_text(mod.operation.get_asm(print_generic_op_form=True))
 return run(['build/gqa-opt',temp,'--allow-unregistered-dialect','--attention-factor-cache-commit','--canonicalize','-o',dest],log)
def winner_inputs(seed=171,sign=1):
 x=list(inputs(CASE,seed));x[4][0]+=sign*8;x[4][1]-=sign*8;return tuple(x)
def main():
 torch.set_num_threads(1);w=Path('research2/results/winner');w.mkdir(parents=True,exist_ok=True);model=WinnerAttention(CASE);x=winner_inputs()
 aot.export(torch.export.export(model,x)).save_mlir(str(w/'torch.mlir'));run(['iree-opt',w/'torch.mlir','--torch-function-to-torch-backend-pipeline','--torch-backend-to-linalg-on-tensors-backend-pipeline','--canonicalize','-o',w/'baseline.mlir'],w/'lower.log')
 factor(w/'baseline.mlir',w/'factored.mlir',w/'factor.log');assert (w/'factored.mlir').read_text().count('commit_factored')==2
 lift(w/'baseline.mlir',w/'lifted.mlir');normalize(w/'lifted.mlir',w/'lifted-normalized.mlir',w/'normalize.log')
 forward(w/'factored.mlir',w/'factor-conditional.mlir',w/'factor-conditional.log');lift(w/'factored.mlir',w/'factor-lifted.mlir');normalize(w/'factor-lifted.mlir',w/'factor-normalized.mlir',w/'factor-normalize.log');patch(w/'factor-normalized.mlir',w/'factor-masked.mlir',w/'factor-masked.log',ratio=4)
 row={'case':'attention-derived-winner','shape':CASE,'variants':{}}
 for v,source,extra in [('strong','baseline',[]),('aggressive','baseline',['--iree-dispatch-creation-enable-aggressive-fusion']),('lifted-normalized','lifted-normalized',[]),('lifted-aggressive','lifted-normalized',['--iree-dispatch-creation-enable-aggressive-fusion']),('factored','factored',[]),('factor-conditional','factor-conditional',[]),('factor-masked','factor-masked',[])]:
  flags=['--iree-hal-target-backends=llvm-cpu','--iree-llvmcpu-target-cpu=host',*STRONG,*extra];cs=run(['iree-compile',w/(source+'.mlir'),*flags,'-o',w/(v+'.vmfb')],w/(v+'-compile.log'));run(['iree-compile',w/(source+'.mlir'),*flags,'--compile-to=hal','-o',w/(v+'-hal.mlir')],w/(v+'-hal.log'));checks=[]
  for seed in [171,811]:
   for sign in [-1,1]:
    x=winner_inputs(seed,sign);refs=reference(model,x)
    with torch.inference_mode():exact=model(*x)
    cfg,ctx,fn,args=load(w/(v+'.vmfb'),[a.clone() for a in x]);before=cfg.device.allocator.statistics;ys=fn(*args);after=cfg.device.allocator.statistics;assert len(ys)==5;errors=[]
    for i,(actual,ref) in enumerate(zip(ys,refs)):
     arr=actual.to_host()
     if i>=2:assert arr.tobytes()==exact[i].numpy().tobytes(),(v,seed,sign,i)
     else:np.testing.assert_allclose(arr,ref,atol=4e-5,rtol=1e-3);errors.append(float(np.max(np.abs(arr-ref))))
    checks.append({'seed':seed,'sign':sign,'winner':bool(ys[-1].to_host()),'max_abs':max(errors),'cache_and_decision_bitwise_equal':True})
  assert {c['winner'] for c in checks}=={True,False}
  text=(w/(v+'-hal.mlir')).read_text();row['variants'][v]={'correct':True,'checks':checks,'allocated_bytes':after['device_bytes_allocated']-before['device_bytes_allocated'],'copy_command_bytes':copy_stats(text),'dispatches':text.count('hal.command_buffer.dispatch'),'compile_s':cs,'vmfb_sha256':hashlib.sha256((w/(v+'.vmfb')).read_bytes()).hexdigest()};Path('research2/results/winner-study.json').write_text(json.dumps(row,indent=2));print(v,row['variants'][v]['allocated_bytes'],flush=True)
if __name__=='__main__':main()
