"""Compilation-free CPU timing with frozen calibration choices and holdout inputs.
Pause source builds first. Inputs resident, no output download in measured calls.
"""
import sys,time,statistics,random,json,hashlib,gc
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from study_cache import *
from block_study import BranchBlock,block_inputs
from winner_study import WinnerAttention,winner_inputs,CASE as WINNER_CASE

def sample(functions,trials,warmup=5):
 samples={v:[] for v in functions};rng=random.Random(898)
 for fn in functions.values():
  for _ in range(warmup):fn()
 for _ in range(trials):
  order=list(functions);rng.shuffle(order)
  for v in order:
   start=time.perf_counter_ns();functions[v]();samples[v].append((time.perf_counter_ns()-start)/1000)
 return {'samples_us':samples,'median_us':{v:statistics.median(s) for v,s in samples.items()}}
def make_inputs(family,shape,seed):
 if family=='winner':return winner_inputs(seed)
 if family!='block':return inputs(shape,seed)
 f,pool=shape;x=list(block_inputs(f,pool,seed));fixed=block_inputs(f,pool,515)
 x[4:]=fixed[4:];return tuple(x) # Keep model parameters fixed across inputs.
def main():
 torch.set_num_threads(1);os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});groups=[]
 for row in json.loads(Path('research2/results/cache-study.json').read_text())['rows']:
  groups.append(('attention',row['case'],row['shape'],BranchAttention(row['shape']),Path('research2/results/study')/row['case'],row['variants']))
 for row in json.loads(Path('research2/results/block-study.json').read_text()):
  shape=[row['branches'],row['pool']];groups.append(('block',row['case'],shape,BranchBlock(*shape),Path('research2/results/blocks')/row['case'],row['variants']))
 for row in json.loads(Path('research2/results/commit-study.json').read_text()):
  model=BranchAttention(row['shape'],returned='one' if row['case']=='one' else True);groups.append(('commit',row['case'],row['shape'],model,Path('research2/results/commit')/row['case'],row['variants']))
 if Path('research2/results/winner-study.json').exists():
  row=json.loads(Path('research2/results/winner-study.json').read_text());groups.append(('winner',row['case'],WINNER_CASE,WinnerAttention(WINNER_CASE),Path('research2/results/winner'),row['variants']))
 out={'cpu':min(os.sched_getaffinity(0)),'method':'Source compilation paused; local-sync, resident inputs, no output download, one invocation/sample, randomized order; 5warmups,7calibration and21holdout observations. Calibration seed613,holdout811; block weights fixed515.','groups':[]}
 for family,label,shape,model,w,metadata in groups:
  variants=[v for v in ['strong','aggressive','lifted-normalized','lifted-aggressive','forwarded','patch-forwarded','factored','factor-conditional','factor-masked'] if metadata.get(v,{}).get('correct')]
  row={'family':family,'case':label,'shape':shape,'checks':{}}
  selected=None
  for phase,seed,trials in [('calibration',613,7),('holdout',811,21)]:
   x=make_inputs(family,shape,seed)
   with torch.inference_mode():
    value=model(*[a.double() if a.is_floating_point() else a for a in x])
    refs=[a.numpy() for a in value] if isinstance(value,(tuple,list)) else [value.numpy()]
   loaded={};fns={};actuals={}
   for v in variants:
    path=w/(v+'.vmfb');sha=hashlib.sha256(path.read_bytes()).hexdigest();assert sha==metadata[v]['vmfb_sha256'],(label,v)
    cfg,ctx,fn,args=load(path,[a.clone() for a in x]);y=fn(*args);ys=[y] if hasattr(y,'to_host') else list(y);actual=[a.to_host() for a in ys];assert len(actual)==len(refs)
    error=0
    for a,b in zip(actual,refs):
     if a.dtype==np.bool_:np.testing.assert_array_equal(a,b)
     else:np.testing.assert_allclose(a,b,atol=1e-4 if family=='block' else 4e-5,rtol=2e-3);error=max(error,float(np.max(np.abs(a-b))))
    loaded[v]=(cfg,ctx,fn,args);actuals[v]=actual;fns[v]=lambda fn=fn,args=args:fn(*args)
    row['checks'].setdefault(v,{})[phase]={'max_abs':error,'sha256':sha}
   for v in variants:row['checks'][v][phase]['bitwise_equal_strong']=all(a.tobytes()==b.tobytes() for a,b in zip(actuals[v],actuals['strong']))
   eager_inputs=[a.clone() for a in x]
   def eager():
    with torch.inference_mode():return model(*eager_inputs)
   fns['torch-eager']=eager
   # Drop the first result objects before timing; closures retain loaded modules.
   del y,ys,actuals,actual
   t=time.perf_counter();measurement=sample(fns,trials);measurement['measurement_wall_s']=time.perf_counter()-t;row[phase]=measurement
   if phase=='calibration':
    options=[v for v in variants if v in ['strong','lifted-normalized','lifted-aggressive','forwarded','patch-forwarded','factored','factor-conditional','factor-masked']]
    selected=min(options,key=measurement['median_us'].get);row['selected']=selected
   else:
    med=measurement['median_us'];baselines=[v for v in ['strong','aggressive','lifted-normalized','lifted-aggressive'] if v in med];best=min(baselines,key=med.get);row['best_baseline']=best;row['selected_speedup']=med[best]/med[selected];row['patch_speedup']=med[best]/med['factor-masked' if family=='winner' else 'patch-forwarded'];row['conditional_speedup']=med[best]/med['factor-conditional' if family=='winner' else 'forwarded']
   del loaded,fns;gc.collect()
  out['groups'].append(row);Path('research2/results/final-timings.json').write_text(json.dumps(out,indent=2));print(family,label,'baseline',row['best_baseline'],'choice',selected,'selected=%.3f'%row['selected_speedup'],'patch=%.3f'%row['patch_speedup'],flush=True)
 print('Completed',len(groups),flush=True)
if __name__=='__main__':main()
