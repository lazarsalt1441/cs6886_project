"""Compile/run all cache-forwarding forms on NVIDIA; CPU reference first.
Compile-only results are never reported as GPU numerical/performance results.
"""
import sys,argparse,json,time,statistics,random
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from study_cache import *
from block_study import BranchBlock,block_inputs
from winner_study import WinnerAttention,winner_inputs,CASE as WINNER_CASE
import iree.runtime as rt

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--compile-only',action='store_true');ap.add_argument('--family',choices=['attention','block','winner']);ap.add_argument('--arch',default='sm_80');ap.add_argument('--limit',type=int);ap.add_argument('--trials',type=int,default=21);ap.add_argument('--torch-cuda',action='store_true',help='Also measure CUDA PyTorch with all output downloads');args=ap.parse_args();torch.set_num_threads(1)
 cases=[]
 for row in json.loads(Path('research2/results/cache-study.json').read_text())['rows']:cases.append(('attention',row['case'],row['shape']))
 if Path('research2/results/block-study.json').exists():
  for row in json.loads(Path('research2/results/block-study.json').read_text()):cases.append(('block',row['case'],[row['branches'],row['pool']]))
 if Path('research2/results/winner-study.json').exists():cases.append(('winner','attention-derived-winner',WINNER_CASE))
 if args.family:cases=[c for c in cases if c[0]==args.family]
 rows=[]
 for family,label,case in cases[:args.limit]:
  source=Path('research2/results/winner') if family=='winner' else Path('research2/results')/('study' if family=='attention' else 'blocks')/label;w=Path('research2/results/gpu')/args.arch/(family+'-'+label);w.mkdir(parents=True,exist_ok=True)
  if family=='attention':x=inputs(case);refs=reference(BranchAttention(case),x)
  elif family=='winner':x=winner_inputs();refs=reference(WinnerAttention(case),x)
  else:
   f,pool=case;x=block_inputs(f,pool)
   with torch.inference_mode():refs=[BranchBlock(f,pool)(*[a.double() if a.is_floating_point() else a for a in x]).numpy()]
  row={'family':family,'case':label,'arch':args.arch,'compile_only':args.compile_only,'variants':{}};loaded={}
  for v,path in [('strong','baseline'),('aggressive','baseline'),('lifted-aggressive','lifted-normalized'),('lifted-normalized','lifted-normalized'),('forwarded','factor-conditional' if family=='winner' else 'forwarded'),('patch-forwarded','factor-masked' if family=='winner' else 'patch-forwarded')]:
   try:
    cs=run(['iree-compile',source/(path+'.mlir'),'--iree-hal-target-backends=cuda','--iree-cuda-target='+args.arch,*STRONG,*(['--iree-dispatch-creation-enable-aggressive-fusion'] if v in ['aggressive','lifted-aggressive'] else []),'--iree-hal-dump-executable-files-to='+str(w/(v+'-codegen')),'-o',w/(v+'.vmfb')],w/(v+'-compile.log'));row['variants'][v]={'compiled':True,'compile_s':cs}
    if args.compile_only:continue
    cfg=rt.Config('cuda');ctx=rt.SystemContext(config=cfg);ctx.add_vm_module(rt.VmModule.copy_buffer(cfg.vm_instance,(w/(v+'.vmfb')).read_bytes()));dev=[rt.asdevicearray(cfg.device,a.numpy()) for a in x];fn=ctx.modules.module.main;y=fn(*dev);ys=[y] if hasattr(y,'to_host') else list(y);error=0
    assert len(ys)==len(refs),(len(ys),len(refs))
    for actual,ref in zip(ys,refs):
     arr=actual.to_host()
     if arr.dtype==np.bool_:np.testing.assert_array_equal(arr,ref)
     else:np.testing.assert_allclose(arr,ref,atol=2e-4 if family=='block' else 5e-5,rtol=2e-3);error=max(error,float(np.max(np.abs(arr-ref))))
    row['variants'][v].update(correct=True,max_abs=error);loaded[v]=(cfg,ctx,fn,dev)
   except Exception as e:row['variants'].setdefault(v,{}).update(error=str(e))
  row['compatible_fallback']=next((v for v in ['forwarded','strong','lifted-normalized','patch-forwarded'] if row['variants'].get(v,{}).get('compiled') and (args.compile_only or row['variants'][v].get('correct'))),None)
  row['fallback_note']='Compatibility order only; no performance selection. Runtime validation required before deployment.'
  if not args.compile_only and loaded:
   samples={v:[] for v in loaded};rng=random.Random(251)
   def invoke(fn,dev):
    res=fn(*dev)
    for out in ([res] if hasattr(res,'to_host') else res):out.to_host()
   for _,_,fn,dev in loaded.values():
    for _ in range(10):invoke(fn,dev)
   sys.path.insert(0,str(Path('research').resolve()))
   from gpu_selection import calibrate
   calibration=calibrate({v:(lambda fn=fn,dev=dev:invoke(fn,dev)) for v,(_,_,fn,dev) in loaded.items()},'strong')
   row['calibration']=calibration
   for _ in range(args.trials):
    order=list(loaded);rng.shuffle(order)
    for v in order:
     _,_,fn,dev=loaded[v];start=time.perf_counter_ns();invoke(fn,dev);samples[v].append((time.perf_counter_ns()-start)/1000)
   row['synchronized_host_samples_us']=samples;row['synchronized_host_median_us']={v:statistics.median(s) for v,s in samples.items()};row['timing_note']='Input upload excluded; invocation, execution and all output downloads included. Not kernel-only latency.'
   med=row['synchronized_host_median_us'];row['selected_holdout_speedup']=med['strong']/med[calibration['selected']]
  if args.torch_cuda and not args.compile_only:
   model=BranchAttention(case) if family=='attention' else WinnerAttention(case) if family=='winner' else BranchBlock(*case)
   cuda_inputs=[a.cuda() for a in x]
   def eager():
    with torch.inference_mode():
     outputs=model(*cuda_inputs)
     return [a.cpu() for a in outputs] if isinstance(outputs,(tuple,list)) else [outputs.cpu()]
   actual=eager();assert len(actual)==len(refs)
   for a,b in zip(actual,refs):np.testing.assert_allclose(a.numpy(),b,atol=2e-4 if family=='block' else 5e-5,rtol=2e-3)
   for _ in range(10):eager()
   timings=[]
   for _ in range(args.trials):
    torch.cuda.synchronize();start=time.perf_counter_ns();eager();torch.cuda.synchronize();timings.append((time.perf_counter_ns()-start)/1000)
   row['torch_cuda_synchronized_host_samples_us']=timings;row['torch_cuda_synchronized_host_median_us']=statistics.median(timings)
  rows.append(row);Path('research2/results/gpu-'+args.arch+('-compile-only' if args.compile_only else '-runtime')+('-'+args.family if args.family else '')+'.json').write_text(json.dumps(rows,indent=2));print(family,label,{v:r.get('correct',r.get('compiled',False)) for v,r in row['variants'].items()},flush=True)
 print('Completed',len(rows),flush=True)
 if not args.compile_only:assert all(r['variants'].get('strong',{}).get('correct') and r['compatible_fallback'] for r in rows),'Missing baseline or valid optimized implementation; inspect runtime JSON'
if __name__=='__main__':main()
