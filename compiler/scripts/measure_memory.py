"""Fresh-process peak RSS evidence; distinct from static HAL allocation plans."""
import argparse, ctypes, errno, json, resource, statistics, subprocess, sys
from pathlib import Path
import numpy as np
import iree.runtime as rt

def worker(path,case):
 b,hq,hkv,m,n,d=case
 rng=np.random.default_rng(42)
 xs=[rng.standard_normal(s,dtype=np.float32) for s in [(b,hq,m,d),(b,hkv,n,d),(b,hkv,n,d)]]
 cfg=rt.Config('local-sync');vm=rt.VmModule.mmap(cfg.vm_instance,str(path))
 ctx=rt.SystemContext(config=cfg);ctx.add_vm_module(vm)
 args=[rt.asdevicearray(cfg.device,x) for x in xs]
 stats_before=cfg.device.allocator.statistics
 before=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
 y=ctx.modules.module.main(*args).to_host()
 stats_after=cfg.device.allocator.statistics
 after=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
 print(json.dumps(dict(peak_rss_before_bytes=before,peak_rss_after_bytes=after,peak_rss_increase_bytes=max(0,after-before),finite=bool(np.all(np.isfinite(y))),allocator_statistics_available=cfg.device.allocator.has_statistics,allocator_before=stats_before,allocator_after=stats_after)))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--worker');ap.add_argument('--shape',nargs=6,type=int);a=ap.parse_args()
 if a.worker:return worker(a.worker,a.shape)
 root=Path(__file__).resolve().parents[1]
 rows=json.loads((root/'results/benchmark.json').read_text())['rows'];out=[]
 for r in rows:
  case=[r[k] for k in ['b','hq','hkv','m','n','d']]
  item={'case':r['case']}
  for v in ['baseline','optimized']:
   samples=[]
   for _ in range(3):
    p=subprocess.run([sys.executable,__file__,'--worker',str(root/'results/cases'/r['case']/(v+'.vmfb')),'--shape',*map(str,case)],text=True,capture_output=True,check=True)
    samples.append(json.loads(p.stdout))
   item[v]={'samples':samples,'median_peak_rss_bytes':statistics.median(s['peak_rss_after_bytes'] for s in samples),'median_peak_rss_increase_bytes':statistics.median(s['peak_rss_increase_bytes'] for s in samples)}
  out.append(item);(root/'results/memory.json').write_text(json.dumps(out,indent=2));print(r['case'],'measured',flush=True)
 # Try ordinary per-process hardware counters. Record the kernel's answer.
 class Attr(ctypes.Structure):
  _fields_=[('type',ctypes.c_uint32),('size',ctypes.c_uint32),('config',ctypes.c_uint64),('sample_period',ctypes.c_uint64),('sample_type',ctypes.c_uint64),('read_format',ctypes.c_uint64),('flags',ctypes.c_uint64),('wakeup_events',ctypes.c_uint32),('bp_type',ctypes.c_uint32),('config1',ctypes.c_uint64)]
 attr=Attr();attr.size=ctypes.sizeof(attr);attr.type=0;attr.config=3
 libc=ctypes.CDLL(None,use_errno=True);fd=libc.syscall(298,ctypes.byref(attr),0,-1,-1,0)
 err=ctypes.get_errno()
 probe={'platform':'Linux x86_64','event':'PERF_COUNT_HW_CACHE_MISSES','opened':fd>=0,'errno':err if fd<0 else 0,'error':__import__('os').strerror(err) if fd<0 else None}
 if fd>=0:__import__('os').close(fd)
 (root/'results/perf-counter-probe.json').write_text(json.dumps(probe,indent=2))
if __name__=='__main__':main()
