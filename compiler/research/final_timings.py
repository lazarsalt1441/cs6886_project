"""Timing-only confirmation after compilation jobs finish; retain all samples."""
import os,sys,json,time,statistics,random,gc,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from projection_study import ROOT,ProjectedAttention,create_inputs,STRONG
from rotary_projection_study import RotaryProjected,rotate
from portfolio_study import inputs as portfolio_inputs
from native_projection_study import NativeProjected
from native_attention_audit import Attention
from run_suite import load,run
import torch,numpy as np

def measure(functions,trials=21):
 samples={v:[] for v in functions};rng=random.Random(99)
 for fn in functions.values():
  for _ in range(5):fn()
 for _ in range(trials):
  order=list(functions);rng.shuffle(order)
  for v in order:
   start=time.perf_counter_ns();functions[v]();samples[v].append((time.perf_counter_ns()-start)/1000)
 return {'samples_us':samples,'median_us':{v:statistics.median(s) for v,s in samples.items()}}

def main():
 torch.set_num_threads(1);os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
 out={'cpu':min(os.sched_getaffinity(0)),'trials':21,'warmup':5,'groups':[],'method':'Timing-only sequential confirmation, no concurrent compilation in this process, preuploaded inputs, no output download, local-sync; one invocation per sample. Candidate choices frozen from calibration.'}
 groups=[]
 dense=json.loads(Path('research/results/projection-study.json').read_text())['rows']
 for r in dense:
  case=r['shape'];x=create_inputs(case);w=Path('research/results/projections')/r['case'];model=ProjectedAttention(case[6],case[7]);variants={'strong':w/'strong.vmfb','optimized':w/'optimized.vmfb'}
  for v,path in variants.items():
   if not path.exists() or path.stat().st_size==0:
    source=w/('baseline.mlir' if v=='strong' else 'optimized.mlir')
    run(['iree-compile',source,'--iree-hal-target-backends=llvm-cpu','--iree-llvmcpu-target-cpu=host',*STRONG,'-o',path],w/(v+'-final-repair.log'))
  groups.append(('decomposed',r['case'],x,model,variants,None))
 rotary=json.loads(Path('research/results/rotary-projection-study.json').read_text())['rows']
 for r in rotary:
  h,m,n,l,d,dv,rope=r['shape'];x=create_inputs((h,m,n,l,d,dv,True,False));x=(*x,rotate(torch.randn(h,m,rope),torch.arange(n-m,n)),rotate(torch.randn(1,n,rope),torch.arange(n)));w=Path('research/results/rotary-projections')/r['case']
  groups.append(('rotary',r['case'],x,RotaryProjected(),{v:w/(v+'.vmfb') for v in ['native','optimized']},None))
 portfolio=json.loads(Path('research/results/portfolio-study.json').read_text())['rows']
 for r in portfolio:
  x=portfolio_inputs(r['shape'],789);w=Path('research/results/portfolio')/r['case']
  variants={v:w/(v+'.vmfb') for v,data in r['variants'].items() if data.get('compiled') and all(c['correct'] for c in data.get('checks',[]))}
  groups.append(('portfolio',r['case'],x,NativeProjected(r['shape'][6]),variants,r['selected']))
 # All repairs finish before this timing-only section begins.
 for family,name,x,model,variants,selected in groups:
  with torch.inference_mode():ref=model(*[a.double() for a in x]).numpy()
  loaded={};functions={};checks={}
  for v,path in variants.items():
   assert path.stat().st_size>0,path
   cfg,ctx,fn,args=load(path,x);y=fn(*args).to_host();np.testing.assert_allclose(y,ref,atol=5e-5,rtol=2e-3)
   loaded[v]=(cfg,ctx,fn,args);functions[v]=lambda fn=fn,args=args:fn(*args)
   checks[v]={'max_abs':float(np.max(np.abs(y-ref))),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
  measurements=measure(functions);measurements.update(family=family,case=name,selected=selected,checks=checks)
  med=measurements['median_us'];baseline='strong' if family=='decomposed' else 'native' if family=='rotary' else 'original';chosen=selected if family=='portfolio' else 'optimized';measurements['speedup']=med[baseline]/med[chosen]
  if family=='portfolio':measurements['speedup_vs_materialized']=med['materialized']/med[chosen]
  out['groups'].append(measurements);Path('research/results/final-timings.json').write_text(json.dumps(out,indent=2));print(family,name,'speedup=%.3f'%measurements['speedup'],flush=True)
  del loaded;gc.collect()
 print('Completed',len(groups),flush=True)
if __name__=='__main__':main()
