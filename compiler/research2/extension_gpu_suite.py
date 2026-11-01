"""CUDA reproduction for update chains and observable snapshots; compile-only here."""
import sys,json,argparse,time,random,statistics
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from study_cache import *
from probe_update_chain import ChainAttention,chain_inputs
from snapshot_read_study import SnapshotReadAttention,CASE
from repair_snapshot_cow import compile_repaired
import iree.runtime as rt

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--compile-only',action='store_true');ap.add_argument('--arch',default='sm_80');ap.add_argument('--family',choices=['chain','snapshot']);ap.add_argument('--trials',type=int,default=21);ap.add_argument('--prepared-ir',action='store_true',help='Reuse bundled architecture-specific COW-repaired IR; no standalone SDK needed');ap.add_argument('--skip-root-cow',action='store_true',help='Run tensor-level forms without the phase-local COW baseline');a=ap.parse_args();torch.set_num_threads(1)
 if not a.compile_only:assert rt.get_driver('cuda').query_available_devices(),'No CUDA devices; use --compile-only on CPU hosts'
 cases=[('chain',t) for t in [1,4,8,16]]+[('snapshot',1)];rows=[]
 for family,t in cases:
  if a.family and a.family!=family:continue
  source=Path('research2/results/update-chain')/f't{t}' if family=='chain' else Path('research2/results/snapshot-read');model=ChainAttention(t) if family=='chain' else SnapshotReadAttention(CASE);x=chain_inputs(t) if family=='chain' else inputs(CASE);refs=reference(model,x);w=Path('research2/results/gpu-extensions')/a.arch/f'{family}-{t}';w.mkdir(parents=True,exist_ok=True)
  variants=[('baseline','baseline' if family=='chain' else 'protected'),('fused','normalized'),('conditional','coalesced-forwarded' if family=='chain' else 'forwarded'),('masked','coalesced-masked' if family=='chain' else 'masked')]+([('root-cow','baseline')] if family=='snapshot' and not a.skip_root_cow else [])
  row={'family':family,'tokens':t,'arch':a.arch,'compile_only':a.compile_only,'variants':{}};loaded={}
  for v,stem in variants:
   try:
    if v=='root-cow':
     if a.prepared_ir:
      prepared=w/(v+'.cow-artifacts')/'repaired.mlir';assert prepared.exists(),str(prepared)+' missing; use --skip-root-cow or build gqa-opt'
      cs=run(['iree-compile',prepared,'--compile-from=flow','--iree-hal-target-backends=cuda','--iree-cuda-target='+a.arch,*STRONG,'-o',w/(v+'.vmfb')],w/(v+'-prepared-compile.log'))
     else:count=compile_repaired(source/(stem+'.mlir'),w/(v+'.vmfb'),target='cuda',arch=a.arch);cs=None
    else:cs=run(['iree-compile',source/(stem+'.mlir'),'--iree-hal-target-backends=cuda','--iree-cuda-target='+a.arch,*STRONG,*(['--iree-dispatch-creation-enable-aggressive-fusion'] if v=='fused' else []),'-o',w/(v+'.vmfb')],w/(v+'-compile.log'))
    row['variants'][v]={'compiled':True,'compile_s':cs}
    if a.compile_only:continue
    cfg=rt.Config('cuda');ctx=rt.SystemContext(config=cfg);ctx.add_vm_module(rt.VmModule.copy_buffer(cfg.vm_instance,(w/(v+'.vmfb')).read_bytes()));dev=[rt.asdevicearray(cfg.device,z.numpy()) for z in x];ys=ctx.modules.module.main(*dev);assert len(ys)==len(refs);error=0
    for i,(actual,ref) in enumerate(zip(ys,refs)):
     arr=actual.to_host();np.testing.assert_allclose(arr,ref,atol=5e-5,rtol=2e-3);error=max(error,float(np.max(np.abs(arr-ref))))
     if family=='snapshot' and i>0:assert arr.tobytes()==x[i].numpy().tobytes()
    row['variants'][v].update(correct=True,max_abs=error);loaded[v]=(cfg,ctx,ctx.modules.module.main,dev)
   except Exception as e:row['variants'].setdefault(v,{}).update(error=str(e))
  if loaded:
   def invoke(fn,dev):
    for out in fn(*dev):out.to_host()
   samples={v:[] for v in loaded};rng=random.Random(652)
   for _,_,fn,dev in loaded.values():
    for _ in range(10):invoke(fn,dev)
   for _ in range(a.trials):
    order=list(loaded);rng.shuffle(order)
    for v in order:
     _,_,fn,dev=loaded[v];start=time.perf_counter_ns();invoke(fn,dev);samples[v].append((time.perf_counter_ns()-start)/1000)
   row['host_samples_us']=samples;row['host_median_us']={v:statistics.median(s) for v,s in samples.items()};row['timing_note']='Synchronized invocation + all output downloads. Resident input upload excluded. Not kernel-only latency.'
  rows.append(row);Path('research2/results/gpu-extensions-'+a.arch+('-compile-only' if a.compile_only else '-runtime')+('-'+a.family if a.family else '')+'.json').write_text(json.dumps(rows,indent=2));print(family,t,{v:r.get('correct',r.get('compiled',False)) for v,r in row['variants'].items()},flush=True)
 if not a.compile_only:assert all(r['variants'].get('baseline',{}).get('correct') and any(r['variants'].get(v,{}).get('correct') for v in ['masked','conditional']) for r in rows)
 print('Completed',len(rows),flush=True)
if __name__=='__main__':main()
