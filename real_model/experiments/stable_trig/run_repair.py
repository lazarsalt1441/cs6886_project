"""Correctness-gated long-context decode, with non-aborting failure diagnostics.
Reuse the original project/model/compiled modules. No tolerance changes.
"""
import argparse,copy,gc,hashlib,json,os,random,statistics,subprocess,sys,time,traceback
from pathlib import Path

def compare(reference,actual,offset):
 import numpy as np
 rows=[]
 if len(reference)!=len(actual):return dict(pass_=False,error='output count')
 for i,(ref,got) in enumerate(zip(reference,actual)):
  r=np.asarray(ref);g=np.asarray(got)
  row=dict(output=i,shape=list(r.shape))
  if r.shape!=g.shape:row.update(pass_=False,error='shape mismatch');rows.append(row);continue
  if r.dtype.kind!='f':row.update(pass_=bool(np.array_equal(g,r)));rows.append(row);continue
  def metrics(x,y):
   d=np.abs(x.astype('float64')-y.astype('float64'));allowed=5e-4+3e-3*np.abs(x.astype('float64'));bad=(d>allowed)|~np.isfinite(d)
   coord=np.unravel_index(np.argmax(np.where(np.isfinite(d),d,np.inf)),d.shape)
   return dict(max_abs=float(np.max(d)),rms=float(np.sqrt(np.mean(d*d))),bad_elements=int(np.count_nonzero(bad)),max_tolerance_ratio=float(np.max(d/allowed)),worst_coordinate=[int(k) for k in coord],reference_at_worst=float(x[coord]),actual_at_worst=float(y[coord]))
  row.update(metrics(r,g));row['pass_']=row['bad_elements']==0
  if i>=offset and r.ndim==4:
   row['cache_layer_zero_based']=(i-offset)//2;row['cache_kind']='K' if (i-offset)%2==0 else 'V'
   row['append_row']=metrics(r[:,:,-1:,:],g[:,:,-1:,:]);row['prefix_exact']=bool(np.array_equal(r[:,:,:-1,:],g[:,:,:-1,:]))
  rows.append(row)
 return dict(pass_=all(r['pass_'] for r in rows),outputs=rows)

def main():
 p=argparse.ArgumentParser();p.add_argument('--project',type=Path,default=Path('/content/attention_real_llm'));p.add_argument('--set',choices=['A','B'],required=True);p.add_argument('--contexts',nargs='+',type=int,default=[2048,3072,4096]);p.add_argument('--prompts',nargs='+',type=int,default=[0,1,2]);p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--trials',type=int,default=31);p.add_argument('--repetitions',type=int,default=3);p.add_argument('--warmup',type=int,default=5);p.add_argument('--torch-compile',action='store_true');p.add_argument('--source-results',type=Path);a=p.parse_args()
 if any(n<2 or n>4096 for n in a.contexts):p.error('Existing export supports past lengths 2..4096; no generation in this test')
 if min(a.trials,a.repetitions,a.warmup)<1:p.error('counts must be positive')
 root=a.project.resolve();sys.path.insert(0,str(root))
 import benchmark as b
 import numpy as np,torch,iree.runtime as rt
 from llm_workloads import Decode,Lookahead
 from transformers import AutoTokenizer
 torch.set_num_threads(1);torch.manual_seed(0)
 if a.device=='cuda':
  assert torch.cuda.is_available();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 arch='sm_%d%d'%torch.cuda.get_device_capability() if a.device=='cuda' else None
 out=root/('repair_results_v2_'+a.set);out.mkdir(exist_ok=True);prepared=root/'prepared'/a.set;audit=json.loads((prepared/'audit.json').read_text());offset=1 if a.set=='A' else 2
 b.write_json(out/'environment.json',dict(runner_version=2,config=vars(a),torch=torch.__version__,python=sys.version,arch=arch,gpu=torch.cuda.get_device_name() if a.device=='cuda' else None,audit=audit,tolerance=dict(atol=5e-4,rtol=3e-3),scope='Fixed input one-step timing only; no generation/TTFT/serving claim'))
 if not (root/'model/model.safetensors').exists():subprocess.run([sys.executable,str(root/'scripts/download_model.py')],check=True)
 model=b.load_model();wrapper=(Decode if a.set=='A' else Lookahead)(model).eval();tok=AutoTokenizer.from_pretrained(root/'model',local_files_only=True)
 weights=prepared/'weights.irpa'
 if not weights.exists():b.aot.save_module_parameters(weights,wrapper)
 runtimes={};compilation={};source=a.source_results or root/('results_'+a.set)
 repair=Path(__file__).resolve().parent/'prepared'
 specs=[('iree_baseline',root/'prepared'/a.set/'baseline.mlir',False)]
 if a.set=='B':specs.append(('iree_optimized',root/'prepared/B/optimized.mlir',False))
 specs.append(('iree_stable',repair/(a.set+'_stable.mlir'),False))
 if a.set=='B':specs.extend([('iree_stable_commit',repair/'B_stable_commit.mlir',False),('iree_stable_commit_aggressive',repair/'B_stable_commit.mlir',True)])
 flags=['--iree-input-demote-f64-to-f32=false','--iree-opt-numeric-precision-reduction=false']
 b.STRONG+=flags
 for name,src,aggressive in specs:
  folder=out/name;vm=folder/'module.vmfb'
  original=list(b.STRONG)
  if aggressive:b.STRONG+=['--iree-dispatch-creation-enable-aggressive-fusion']
  print('COMPILING',name,flush=True)
  try:compilation[name]=dict(seconds=b.compile_iree(src,vm,a.device,arch,folder),source=str(src),flags=list(b.STRONG))
  finally:b.STRONG=original
  runtimes[name]=b.load_iree(vm,weights,'cuda' if a.device=='cuda' else 'local-sync')
 b.write_json(out/'compilation.json',compilation)
 # Keep a CPU reference model; GPU eager and SDPA models are independent copies.
 eager=(Decode if a.set=='A' else Lookahead)(copy.deepcopy(model)).to(a.device).eval()
 sdpa_model=copy.deepcopy(model);sdpa_model.config._attn_implementation='sdpa';sdpa=(Decode if a.set=='A' else Lookahead)(sdpa_model).to(a.device).eval()
 compiled=torch.compile(eager,fullgraph=True,dynamic=False) if a.torch_compile else None
 all_cases=[];timings=[]
 def sync():
  if a.device=='cuda':torch.cuda.synchronize()
 def arrays(result):return tuple(b.cpu_array(z).copy() for z in result)
 for n in a.contexts:
  for prompt in a.prompts:
   case=dict(context=n,prompt=prompt,validation={},errors={});calls={};iree_inputs={};actual={}
   try:
    host,_=b.real_inputs(model,tok,n,prompt,a.set)
    with torch.inference_mode():gold=arrays(wrapper(*host))
    device_args=tuple(x.to(a.device).clone() for x in host)
    calls={'torch_eager':lambda:eager(*device_args),'torch_sdpa':lambda:sdpa(*device_args)}
    if compiled is not None:
     try:
      t=time.perf_counter()
      with torch.inference_mode():compiled(*device_args);sync()
      case['torch_compile_first_call_s']=time.perf_counter()-t;calls['torch_compile']=lambda:compiled(*device_args)
     except Exception:case['errors']['torch_compile']=traceback.format_exc()
    for name,r in runtimes.items():
     resident=tuple(rt.asdevicearray(r['config'].device,z.numpy().copy()) for z in host);iree_inputs[name]=resident;calls[name]=(lambda r=r,resident=resident:r['fn'](*resident))
    with torch.inference_mode():
     for name,fn in calls.items():
      try:actual[name]=arrays(fn());case['validation'][name]=compare(gold,actual[name],offset)
      except Exception:case['errors'][name]=traceback.format_exc()
     if 'iree_optimized' in actual and 'iree_baseline' in actual:
      case['optimized_vs_baseline']=compare(actual['iree_baseline'],actual['iree_optimized'],offset)
      case['optimized_baseline_array_equal']=all(np.array_equal(x,y) for x,y in zip(actual['iree_baseline'],actual['iree_optimized']))
     if 'iree_stable_commit' in actual and 'iree_stable' in actual:
      case['stable_commit_vs_stable']=compare(actual['iree_stable'],actual['iree_stable_commit'],offset)
     if 'torch_eager' in actual:
      case['vs_gpu_torch_eager']={name:compare(actual['torch_eager'],got,offset) for name,got in actual.items() if name.startswith('iree_')}
     # Both branch winners are exercised independently; no forced case is timed.
     if a.set=='B':
      case['forced_winners']={}
      for wanted in [0,1]:
       priors=torch.tensor([0.,-100.] if wanted==0 else [-100.,0.])
       forced=(host[0],priors,*host[2:]);fgold=arrays(wrapper(*forced));assert int(fgold[1])==wanted
       reports={}
       for name,rtv in runtimes.items():
        original=iree_inputs[name]
        fargs=(original[0],rt.asdevicearray(rtv['config'].device,priors.numpy()),*original[2:])
        try:reports[name]=compare(fgold,arrays(rtv['fn'](*fargs)),offset)
        except Exception:reports[name]=dict(pass_=False,error=traceback.format_exc())
       # Torch must have forced-winner reports too, otherwise the strict gate
       # excludes these baselines despite successful natural-case validation.
       torch_forced=(device_args[0],priors.to(a.device),*device_args[2:])
       torch_models={'torch_eager':eager,'torch_sdpa':sdpa}
       if 'torch_compile' in calls:torch_models['torch_compile']=compiled
       for name,fn in torch_models.items():
        try:reports[name]=compare(fgold,arrays(fn(*torch_forced)),offset)
        except Exception:reports[name]=dict(pass_=False,error=traceback.format_exc())
       case['forced_winners'][str(wanted)]=reports
     # Compact numerical evidence: logits + newly appended cache row per output.
     saved={}
     for name,values in [('cpu_reference',gold),*actual.items()]:
      for i,z in enumerate(values):saved[f'{name}_{i}']=z[:,:,-1:,:] if i>=offset and z.ndim==4 else z
     evidence=out/f'evidence_{n}_{prompt}.npz';tmp=evidence.with_suffix('.tmp')
     with tmp.open('wb') as f:np.savez_compressed(f,**saved)
     tmp.replace(evidence);saved.clear()
     all_cases.append(case);b.write_json(out/'validation.json',all_cases)
     eligible={k:fn for k,fn in calls.items() if case['validation'].get(k,{}).get('pass_',False) and all(v.get(k,{}).get('pass_',False) for v in case.get('forced_winners',{}).values())}
     print('VALIDATION',n,prompt,{k:v['pass_'] for k,v in case['validation'].items()},'opt_base_equal',case.get('optimized_baseline_array_equal'),flush=True)
     for name,fn in list(eligible.items()):
      try:
       for _ in range(a.warmup):temp=fn()
       sync()
      except Exception:case['errors'][name+'_warmup']=traceback.format_exc();del eligible[name]
     for rep in range(a.repetitions):
      samples={k:[] for k in eligible};rng=random.Random(n*100+prompt*10+rep)
      for _ in range(a.trials):
       order=list(eligible);rng.shuffle(order)
       for name in order:
        sync();t=time.perf_counter();temp=eligible[name]();sync();samples[name].append((time.perf_counter()-t)*1000)
      for name,values in samples.items():
       if values:timings.append(dict(context=n,prompt=prompt,repetition=rep,backend=name,median_ms=statistics.median(values),p95_ms=float(np.percentile(values,95)),samples_ms=values,correctness_passed_before=True))
      b.write_json(out/'latency.json',timings)
     for name,fn in eligible.items():
      try:case.setdefault('after_timing',{})[name]=compare(gold,arrays(fn()),offset)
      except Exception:case['errors'][name+'_after']=traceback.format_exc()
     for name,inputs in iree_inputs.items():case.setdefault('input_unchanged',{})[name]=all(np.array_equal(b.cpu_array(x),z.numpy()) for x,z in zip(inputs,host))
     case.setdefault('input_unchanged',{})['torch']=all(np.array_equal(b.cpu_array(x),z.numpy()) for x,z in zip(device_args,host))
     for row in timings:
      if row['context']==n and row['prompt']==prompt:
       row['valid']=bool(case.get('after_timing',{}).get(row['backend'],{}).get('pass_',False) and all(case['input_unchanged'].values()))
   except Exception:
    case['errors']['case']=traceback.format_exc()
    if not any(x is case for x in all_cases):all_cases.append(case)
    print('CASE ERROR',n,prompt,case['errors']['case'][-500:],flush=True)
   finally:
    b.write_json(out/'validation.json',all_cases);b.write_json(out/'latency.json',timings)
    calls.clear();iree_inputs.clear();actual.clear();gc.collect()
    if a.device=='cuda':torch.cuda.empty_cache()
 b.write_json(out/'completion.json',dict(completed=True,cases=len(all_cases),scope='Failures retained and skipped for timing; only valid=true rows may support performance claims'))
 print('TIMED BACKENDS',sorted(set(row['backend'] for row in timings)),flush=True)
 print('FINISHED',out,flush=True)
if __name__=='__main__':main()
