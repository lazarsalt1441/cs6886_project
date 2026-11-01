"""Fresh randomized host invocation timing, run AFTER all compilation jobs finish."""
from pathlib import Path
import argparse,json,os,time,random
import numpy as np
import iree.runtime as rt
P=Path(__file__).resolve().parents[1];ap=argparse.ArgumentParser();ap.add_argument('--label',required=True);a=ap.parse_args();R=P/'results'/a.label
try:os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
except OSError:pass
cases={c['name']:c for c in json.loads((P/'reproducers/cases.json').read_text())};results=[]
groups=[(b,b+'_stock',b+('_actual' if b.startswith('gather_forks') else '_custom')) for b in ['forward_64','forward_4096','winner','chain','projection_32','projection_512','partial_rope','gather_forks_512','gather_forks_8192']]
groups += [('winner_cpp','winner_generic_stock','winner_actual'),('projection_cpp','original_projection_stock','original_projection_actual'),('partial_rope_cpp','original_rope_stock','original_rope_actual'),('coalesce_cpp','coalesce_stock','coalesce_actual')]
groups += [('concat_cpp','concat_stock','concat_actual')]
for base,stock_name,custom_name in groups:
 variants={};rng=np.random.default_rng(817)
 entries=[(s,k) for s,n in [('stock',stock_name),('custom',custom_name)] for k in ['default','O3','strong','aggressive','max_reassociation'] if (R/(n+'__'+k)/'result.json').exists()]
 for suffix,k in entries:
  name=stock_name if suffix=='stock' else custom_name;meta=cases[name];status=json.loads((R/(name+'__'+k)/'result.json').read_text())
  if not all(x['passed'] for x in status['validation']):raise ValueError('Incorrect benchmark candidate')
  conf=rt.Config('local-sync');ctx=rt.SystemContext(config=conf);vm=rt.VmModule.mmap(conf.vm_instance,str(R/(name+'__'+k)/'module.vmfb'));ctx.add_vm_module(vm)
  arrays=[rng.normal(0,.3,s).astype('float32') for s in meta['shapes']]
  # All variants receive IDENTICAL arrays, independently uploaded.
  rng=np.random.default_rng(817)
  if base.startswith('forward'):arrays[-1]=np.array([16,18],dtype='int64')
  if base=='winner':arrays[-1]=np.array([1],dtype='int32')
  if base=='chain':arrays[-1]=np.array([16,18,19],dtype='int64')
  if base.startswith('gather_forks') or base=='coalesce_cpp':arrays[-1]=np.array([0,127,128,129,131,132,256,meta['N']-1],dtype='int64')
  if base=='winner_cpp':arrays[-1]=np.array(True)
  if base=='concat_cpp':arrays[-1]=np.array(True)
  dev=[rt.asdevicearray(conf.device,x.copy()) for x in arrays];f=ctx.modules.module.main
  for _ in range(20):f(*dev)
  variants[suffix+'_'+k]=dict(conf=conf,ctx=ctx,vm=vm,dev=dev,f=f,times=[])
 order=list(variants);rr=random.Random(128)
 calibration={key:[] for key in variants}
 for _ in range(31):
  rr.shuffle(order)
  for key in order:
   v=variants[key];ts=time.perf_counter_ns();out=v['f'](*v['dev']);calibration[key].append((time.perf_counter_ns()-ts)/1000)
 selected=min((k for k in variants if k.startswith('stock_')),key=lambda k:np.median(calibration[k]))
 selected_custom=min((k for k in variants if k.startswith('custom_')),key=lambda k:np.median(calibration[k]))
 for _ in range(201):
  rr.shuffle(order)
  for key in order:
   v=variants[key];ts=time.perf_counter_ns();out=v['f'](*v['dev']);v['times'].append((time.perf_counter_ns()-ts)/1000)
 row={'case':base,'stock_program':stock_name,'custom_program':custom_name,'selected_stock':selected,'selected_custom':selected_custom,'calibration_us':calibration,'variants':{key:{'median_us':float(np.median(v['times'])),'p10_us':float(np.percentile(v['times'],10)),'p90_us':float(np.percentile(v['times'],90)),'samples_us':v['times']} for key,v in variants.items()}}
 stock=np.array(variants[selected]['times']);custom=np.array(variants[selected_custom]['times']);boots=[];br=np.random.default_rng(89)
 for _ in range(2000):
  idx=br.integers(0,len(stock),len(stock));boots.append(float(np.median(stock[idx])/np.median(custom[idx])))
 row['speedup_vs_selected']=float(np.median(stock)/np.median(custom));row['paired_bootstrap_95pct']=np.percentile(boots,[2.5,97.5]).tolist();results.append(row);print(base,selected,row['speedup_vs_selected'],flush=True)
 row['matched_strong_speedup']=float(np.median(variants['stock_strong']['times'])/np.median(variants['custom_strong']['times']))
(R/'paired_timings.json').write_text(json.dumps(results,indent=2))
