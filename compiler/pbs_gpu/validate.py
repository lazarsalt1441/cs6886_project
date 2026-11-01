"""Fresh-input CUDA audit of already compiled binaries. Never times incorrect kernels."""
import argparse,json,sys,hashlib,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'research2'),str(ROOT/'research'),str(ROOT/'scripts')]
import os
os.chdir(ROOT)
import numpy as np,torch,iree.runtime as rt
from study_cache import BranchAttention,inputs,reference
from block_study import BranchBlock,block_inputs
from winner_study import WinnerAttention,winner_inputs,CASE as WC
from probe_update_chain import ChainAttention,chain_inputs
from snapshot_read_study import SnapshotReadAttention,CASE as SC
from projection_study import ProjectedAttention,create_inputs
from rotary_projection_study import RotaryProjected,rotate
from native_projection_study import NativeProjected

def specifications(a):
 jobs=[]
 if a.family in ['all','cache']:
  for row in json.loads(Path('research2/results/cache-study.json').read_text())['rows']:
   case=row['shape'];jobs.append(('cache',row['case'],Path('research2/results/gpu')/a.arch/('attention-'+row['case']),lambda seed,c=case:(BranchAttention(c),inputs(c,seed))))
  for row in json.loads(Path('research2/results/block-study.json').read_text()):
   f,p=row['branches'],row['pool'];jobs.append(('block',row['case'],Path('research2/results/gpu')/a.arch/('block-'+row['case']),lambda seed,f=f,p=p:(BranchBlock(f,p),block_inputs(f,p,seed))))
 if a.family in ['all','winner']:
  jobs.append(('winner','winner',Path('research2/results/gpu')/a.arch/'winner-attention-derived-winner',lambda seed:(WinnerAttention(WC),winner_inputs(seed,-1 if seed%2 else 1))))
 if a.family in ['all','chain']:
  for t in [1,4,8,16]:jobs.append(('chain',str(t),Path('research2/results/gpu-extensions')/a.arch/f'chain-{t}',lambda seed,t=t:(ChainAttention(t),chain_inputs(t,seed))))
  jobs.append(('snapshot','snapshot',Path('research2/results/gpu-extensions')/a.arch/'snapshot-1',lambda seed:(SnapshotReadAttention(SC),inputs(SC,seed))))
 if a.family in ['all','projection']:
  for kind,file in [('decomposed','projection-study.json'),('native','native-projection-study.json'),('rotary','rotary-projection-study.json')]:
   for row in json.loads((Path('research/results')/file).read_text())['rows']:
    def make(seed,c=row['shape'],kind=kind):
     base=tuple(c) if kind!='rotary' else (*c[:6],True,False)
     x=create_inputs(base);torch.manual_seed(seed)
     # New data and weights, same valid shape/contract; no frontend re-export.
     x=tuple(torch.randn_like(z)*(z.std().item() if z.numel()>1 else 1) for z in x)
     if kind=='rotary':
      h,m,n,l,d,dv,r=c
      x=(*x,rotate(torch.randn(h,m,r),torch.arange(n-m,n)),rotate(torch.randn(1,n,r),torch.arange(n)))
      model=RotaryProjected()
     else:model=NativeProjected(c[6]) if kind=='native' else ProjectedAttention(c[6],c[7])
     return model,x
    jobs.append(('projection',row['case'],Path('research/results/gpu')/a.arch/(kind+'-'+row['case']),make))
 return jobs

def main():
 p=argparse.ArgumentParser();p.add_argument('--arch',required=True);p.add_argument('--family',default='all');p.add_argument('--seeds',type=int,default=10);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');a=p.parse_args();torch.set_num_threads(1)
 report={'arch':a.arch,'seeds':a.seeds,'device':a.device,'rows':[]};output=Path(a.output)
 for family,name,directory,make in specifications(a):
  binaries=sorted(directory.glob('*.vmfb'))
  stems={b.stem for b in binaries}
  base='strong' if family in ['cache','block','winner'] else 'baseline'
  opts={'optimized','decomposed-fallback'} if family=='projection' else {'conditional','masked'} if family in ['chain','snapshot'] else {'forwarded','patch-forwarded'}
  if base not in stems or not (stems & opts):report['rows'].append(dict(family=family,case=name,status='INCOMPLETE_BINARIES',available=sorted(stems)))
  for binary in binaries:
   record=dict(family=family,case=name,variant=binary.stem,sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),checks=[])
   try:
    cfg=rt.Config(a.device);ctx=rt.SystemContext(config=cfg);ctx.add_vm_module(rt.VmModule.copy_buffer(cfg.vm_instance,binary.read_bytes()))
    for seed in range(101,101+a.seeds):
     check={'seed':seed}
     try:
      model,x=make(seed)
      with torch.inference_mode():
       ref=model(*[z.double() if z.is_floating_point() else z for z in x]);exact=model(*x)
      refs=list(ref) if isinstance(ref,(list,tuple)) else [ref];exacts=list(exact) if isinstance(exact,(list,tuple)) else [exact]
      # Fresh device arrays every call: a previous invocation cannot hide input mutation.
      arrays=[rt.asdevicearray(cfg.device,z.numpy()) for z in x]
      stats=cfg.device.allocator.statistics if cfg.device.allocator.has_statistics else None
      got=ctx.modules.module.main(*arrays);ys=list(got) if isinstance(got,(list,tuple)) else [got]
      assert len(ys)==len(refs)
      error=0.;bits=True
      for i,(actual,r) in enumerate(zip(ys,refs)):
       z=actual.to_host();r=r.numpy()
       if family=='winner' and i>=2:bits &= z.tobytes()==exacts[i].numpy().tobytes()
       elif family=='snapshot' and i>0:bits &= z.tobytes()==x[i].numpy().tobytes()
       elif z.dtype==np.bool_:np.testing.assert_array_equal(z,r)
       else:
        np.testing.assert_allclose(z,r,atol=2e-4 if family=='block' else 5e-5,rtol=2e-3)
        error=max(error,float(np.abs(z-r).max()))
      assert bits,'Cache/decision/snapshot bit mismatch'
      check.update(ok=True,max_abs=error,exact_cache_checks=bits)
      if stats:check['invocation_alloc_bytes']=cfg.device.allocator.statistics['device_bytes_allocated']-stats['device_bytes_allocated']
     except Exception as e:check.update(ok=False,error=str(e))
     record['checks'].append(check)
    record['status']='PASS' if all(c['ok'] for c in record['checks']) else 'FAIL'
   except Exception as e:record.update(status='LOAD_FAIL',error=str(e))
   report['rows'].append(record)
   output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2));print(family,name,binary.stem,record['status'],flush=True)
 output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2))
 if not report['rows'] or any(r['status']!='PASS' for r in report['rows']):raise SystemExit(1)
if __name__=='__main__':main()
