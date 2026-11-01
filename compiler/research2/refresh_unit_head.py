import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from study_cache import *
from add_patch_variant import patch
def main():
 torch.set_num_threads(1);path=Path('research2/results/cache-study.json');data=json.loads(path.read_text())
 for row in data['rows']:
  if row['shape'][2]!=1:continue
  w=Path('research2/results/study')/row['case'];row['lifted_gathers']=lift(w/'baseline.mlir',w/'lifted.mlir');normalize(w/'lifted.mlir',w/'lifted-normalized.mlir',w/'normalize.log');patch(w/'lifted-normalized.mlir',w/'patch-forwarded.mlir',w/'patch-pass.log');x=inputs(row['shape']);refs=reference(BranchAttention(row['shape']),x)
  for v in ['lifted-normalized','patch-forwarded']:
   flags=['--iree-hal-target-backends=llvm-cpu','--iree-llvmcpu-target-cpu=host',*STRONG]+(['--iree-dispatch-creation-enable-aggressive-fusion'] if v=='lifted-normalized' else [])
   cs=run(['iree-compile',w/(v+'.mlir'),*flags,'-o',w/(v+'.vmfb')],w/(v+'-compile.log'));run(['iree-compile',w/(v+'.mlir'),*flags,'--compile-to=hal','-o',w/(v+'-hal.mlir')],w/(v+'-hal.log'))
   cfg,ctx,fn,args=load(w/(v+'.vmfb'),[a.clone() for a in x]);before=cfg.device.allocator.statistics;ys=fn(*args);after=cfg.device.allocator.statistics;err=0
   for a,b in zip(ys,refs):y=a.to_host();np.testing.assert_allclose(y,b,atol=4e-5,rtol=1e-3);err=max(err,float(np.max(np.abs(y-b))))
   text=(w/(v+'-hal.mlir')).read_text();row['variants'][v]={'compiled':True,'correct':True,'compile_s':cs,'max_abs':err,'allocated_bytes':after['device_bytes_allocated']-before['device_bytes_allocated'],'hal_alloc_bytes':sum(allocations(text)),'dispatches':text.count('hal.command_buffer.dispatch'),'copy_command_bytes':copy_stats(text),'vmfb_sha256':hashlib.sha256((w/(v+'.vmfb')).read_bytes()).hexdigest()}
  print(row['case'],row['lifted_gathers'],{v:r.get('allocated_bytes') for v,r in row['variants'].items()},flush=True)
 path.write_text(json.dumps(data,indent=2))
if __name__=='__main__':main()
