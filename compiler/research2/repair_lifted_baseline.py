"""Add a corrected fused-gather comparator, preserving the failed raw experiment."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from study_cache import *
def main():
 torch.set_num_threads(1);path=Path('research2/results/cache-study.json');prior=path.with_name('cache-study-raw-lifted.json')
 if not prior.exists():prior.write_bytes(path.read_bytes())
 data=json.loads(prior.read_text())
 for row in data['rows']:
  case=row['shape'];w=Path('research2/results/study')/row['case'];x=inputs(case);refs=reference(BranchAttention(case),x)
  normalize(w/'lifted.mlir',w/'lifted-normalized.mlir',w/'normalize.log');v='lifted-normalized'
  flags=['--iree-hal-target-backends=llvm-cpu','--iree-llvmcpu-target-cpu=host',*STRONG,'--iree-dispatch-creation-enable-aggressive-fusion']
  cs=run(['iree-compile',w/(v+'.mlir'),*flags,'-o',w/(v+'.vmfb')],w/(v+'-compile.log'))
  run(['iree-compile',w/(v+'.mlir'),*flags,'--compile-to=hal','-o',w/(v+'-hal.mlir')],w/(v+'-hal.log'))
  cfg,ctx,fn,dev=load(w/(v+'.vmfb'),[a.clone() for a in x]);before=cfg.device.allocator.statistics;y=fn(*dev);after=cfg.device.allocator.statistics;ys=[y] if hasattr(y,'to_host') else list(y);errors=[]
  for a,b in zip(ys,refs):actual=a.to_host();np.testing.assert_allclose(actual,b,atol=4e-5,rtol=1e-3);errors.append(float(np.max(np.abs(actual-b))))
  text=(w/(v+'-hal.mlir')).read_text();row['variants'][v]={'compiled':True,'correct':True,'compile_s':cs,'max_abs':max(errors),'allocated_bytes':after['device_bytes_allocated']-before['device_bytes_allocated'],'hal_alloc_bytes':sum(allocations(text)),'dispatches':text.count('hal.command_buffer.dispatch'),'copy_command_bytes':copy_stats(text),'vmfb_sha256':hashlib.sha256((w/(v+'.vmfb')).read_bytes()).hexdigest()}
  row['variants']['lifted-raw-diagnostic']=row['variants'].pop('lifted');path.write_text(json.dumps(data,indent=2));print(row['case'],'PASS',row['variants'][v]['allocated_bytes'],flush=True)
if __name__=='__main__':main()
