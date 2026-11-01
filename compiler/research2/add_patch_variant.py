"""Evaluate a fused-form alternative: base gather + masked gathered update."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from study_cache import *
def patch(source,dest,log,ratio=2):
 with ir.Context():
  mod=ir.Module.parse(Path(source).read_text());temp=Path(dest).with_name(Path(dest).stem+'-input.mlir');temp.write_text(mod.operation.get_asm(print_generic_op_form=True))
 return run(['build/gqa-opt',temp,'--allow-unregistered-dialect','--attention-forward-gather-updates=min-pool-demand-ratio='+str(ratio),'--canonicalize','-o',dest],log)
def main():
 torch.set_num_threads(1);path=Path('research2/results/cache-study.json');data=json.loads(path.read_text())
 for row in data['rows']:
  case=row['shape'];w=Path('research2/results/study')/row['case'];x=inputs(case);refs=reference(BranchAttention(case),x);v='patch-forwarded'
  ps=patch(w/'lifted-normalized.mlir',w/(v+'.mlir'),w/'patch-pass.log')
  try:
   flags=['--iree-hal-target-backends=llvm-cpu','--iree-llvmcpu-target-cpu=host',*STRONG]
   cs=run(['iree-compile',w/(v+'.mlir'),*flags,'-o',w/(v+'.vmfb')],w/(v+'-compile.log'))
   run(['iree-compile',w/(v+'.mlir'),*flags,'--compile-to=hal','-o',w/(v+'-hal.mlir')],w/(v+'-hal.log'))
   cfg,ctx,fn,dev=load(w/(v+'.vmfb'),[a.clone() for a in x]);before=cfg.device.allocator.statistics;res=fn(*dev);after=cfg.device.allocator.statistics;ys=[res] if hasattr(res,'to_host') else list(res);errors=[]
   for a,b in zip(ys,refs):arr=a.to_host();np.testing.assert_allclose(arr,b,atol=4e-5,rtol=1e-3);errors.append(float(np.max(np.abs(arr-b))))
   text=(w/(v+'-hal.mlir')).read_text();row['variants'][v]={'compiled':True,'correct':True,'compile_s':cs,'pass_s':ps,'max_abs':max(errors),'allocated_bytes':after['device_bytes_allocated']-before['device_bytes_allocated'],'hal_alloc_bytes':sum(allocations(text)),'dispatches':text.count('hal.command_buffer.dispatch'),'copy_command_bytes':copy_stats(text),'patches':(w/(v+'.mlir')).read_text().count('cache_gather_forwarded'),'vmfb_sha256':hashlib.sha256((w/(v+'.vmfb')).read_bytes()).hexdigest()}
  except Exception as e:row['variants'][v]={'error':str(e)}
  path.write_text(json.dumps(data,indent=2));print(row['case'],row['variants'][v].get('allocated_bytes',row['variants'][v].get('error')),flush=True)
if __name__=='__main__':main()
