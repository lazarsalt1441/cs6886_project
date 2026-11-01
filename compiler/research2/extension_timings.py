"""Isolated calibration/holdout timing for coalescing and repaired snapshots."""
import sys,json,time,random,statistics,hashlib,gc,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from study_cache import *
sys.path.insert(0,str(Path(__file__).resolve().parent))
from final_timings import sample
from probe_update_chain import ChainAttention,chain_inputs
from snapshot_read_study import SnapshotReadAttention,CASE as SNAPSHOT_CASE

def main():
 torch.set_num_threads(1);os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});groups=[]
 for row in json.loads(Path('research2/results/update-chain-study.json').read_text()):
  t=row['tokens'];groups.append(('chain',t,ChainAttention(t),lambda seed,t=t:chain_inputs(t,seed),Path('research2/results/update-chain')/f't{t}',row['variants'],['strong','aggressive','fused'],'coalesced-masked','coalesced-forwarded'))
 row=json.loads(Path('research2/results/snapshot-read-study.json').read_text());groups.append(('snapshot',1,SnapshotReadAttention(SNAPSHOT_CASE),lambda seed:inputs(SNAPSHOT_CASE,seed),Path('research2/results/snapshot-read'),row['variants'],['protected','aggressive','fused','fused-aggressive','root-cow'],'masked','forwarded'))
 out={'cpu':min(os.sched_getaffinity(0)),'method':'No source builds or compilation during timing. Resident inputs, local-sync, no output downloads, torch1thread; randomized order,5warmups,7calibration(seed613),21holdout(seed811). Incorrect stock snapshot baseline excluded.','groups':[]}
 for family,label,model,make_inputs,w,metadata,baselines,patchkey,condkey in groups:
  variants=[v for v,r in metadata.items() if r.get('correct')];row={'family':family,'case':label,'checks':{}}
  for phase,seed,trials in [('calibration',613,7),('holdout',811,21)]:
   x=make_inputs(seed);refs=reference(model,x);loaded={};functions={}
   for v in variants:
    p=w/(v+'.vmfb');assert hashlib.sha256(p.read_bytes()).hexdigest()==metadata[v]['vmfb_sha256'];cfg,ctx,fn,dev=load(p,[a.clone() for a in x]);ys=fn(*dev);assert len(ys)==len(refs);errors=[]
    for a,b in zip(ys,refs):arr=a.to_host();np.testing.assert_allclose(arr,b,atol=4e-5,rtol=1e-3);errors.append(float(np.max(np.abs(arr-b))))
    if family=='snapshot':assert all(ys[i].to_host().tobytes()==x[i].numpy().tobytes() for i in [1,2])
    row['checks'].setdefault(v,{})[phase]={'max_abs':max(errors),'correct':True};loaded[v]=(cfg,ctx,fn,dev);functions[v]=lambda fn=fn,dev=dev:fn(*dev)
   eager_x=[a.clone() for a in x]
   def eager():
    with torch.inference_mode():return model(*eager_x)
   functions['torch-eager']=eager;row[phase]=sample(functions,trials)
   if phase=='calibration':
    options=[*baselines,patchkey,condkey];row['selected']=min(options,key=row[phase]['median_us'].get)
   else:
    med=row[phase]['median_us'];best=min(baselines,key=med.get);row['best_baseline']=best;row['patch_speedup']=med[best]/med[patchkey];row['selected_speedup']=med[best]/med[row['selected']];row['patch_key']=patchkey;row['conditional_key']=condkey
   del loaded,functions;gc.collect()
  out['groups'].append(row);Path('research2/results/extension-timings.json').write_text(json.dumps(out,indent=2));print(family,label,'best',row['best_baseline'],'patch',round(row['patch_speedup'],3),'choice',row['selected'],round(row['selected_speedup'],3),flush=True)
 print('Completed',len(groups),flush=True)
if __name__=='__main__':main()
