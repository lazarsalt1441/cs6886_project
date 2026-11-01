import sys,time,random,statistics,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from probe_forks import Forked
from probe_cache import *
def main():
 torch.set_num_threads(1);os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});w=Path('research2/results/forks-probe');x=inputs();model=Forked()
 with torch.inference_mode():ref=[a.numpy() for a in model(*[t.double() if t.is_floating_point() else t for t in x])]
 flags=['--iree-hal-target-backends=llvm-cpu','--iree-llvmcpu-target-cpu=host',*STRONG]
 run(['iree-compile',w/'optimized.mlir',*flags,'--compile-to=hal','-o',w/'optimized-hal.mlir'],w/'optimized-hal.log')
 loaded={};out={}
 for v in ['strong','aggressive','optimized']:
  cfg,ctx,fn,args=load(w/(v+'.vmfb'),[a.clone() for a in x]);before=cfg.device.allocator.statistics;result=fn(*args);after=cfg.device.allocator.statistics
  errors=[]
  for a,b in zip(result,ref):aa=a.to_host();np.testing.assert_allclose(aa,b,atol=4e-5,rtol=1e-3);errors.append(float(np.max(np.abs(aa-b))))
  out[v]={'correct':True,'max_abs':errors,'allocated_bytes':after['device_bytes_allocated']-before['device_bytes_allocated'],'hal_allocs':allocations((w/(v+'-hal.mlir')).read_text()),'samples_us':[]};loaded[v]=(cfg,ctx,fn,args)
  for _ in range(5):fn(*args)
 rng=random.Random(42)
 for _ in range(21):
  order=list(loaded);rng.shuffle(order)
  for v in order:
   _,_,fn,args=loaded[v];start=time.perf_counter_ns();fn(*args);out[v]['samples_us'].append((time.perf_counter_ns()-start)/1000)
 for v,r in out.items():r['median_us']=statistics.median(r['samples_us'])
 Path('research2/results/forwarding-smoke.json').write_text(json.dumps(out,indent=2));print({v:{k:x for k,x in r.items() if k!='samples_us'} for v,r in out.items()})
if __name__=='__main__':main()
