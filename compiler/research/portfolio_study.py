"""Calibration/holdout evaluation of correct CPU attention compilation choices."""
import sys,os,json,time,statistics,random,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from projection_study import ROOT,STRONG,annotate
from native_projection_study import NativeProjected
from cpu_compile import compile_guarded
from run_suite import run,load
from iree.compiler import ir
from iree.turbine import aot
import torch,numpy as np

def inputs(case,seed):
 h,m,n,l,d,dv,shared=case;torch.manual_seed(seed)
 return (torch.randn(h,m,d),torch.randn(1 if shared else h,n,l),torch.randn(h,l,d)/l**.5,torch.randn(h,l,dv)/l**.5)
def contract(text,project,force):
 with ir.Context():
  mod=ir.Module.parse(text)
  for op in mod.body.operations:
   if op.operation.name=='func.func':
    if project:
     op.attributes['attention.allow_reassociation']=ir.UnitAttr.get();op.attributes['attention.enable_native_projection']=ir.UnitAttr.get()
    if force:op.attributes['attention.force_safe_attention_lowering']=ir.UnitAttr.get()
  return mod.operation.get_asm(print_generic_op_form=True)
def elapsed(fn,args,inner):
 start=time.perf_counter_ns()
 for _ in range(inner):fn(*args)
 return (time.perf_counter_ns()-start)/inner/1000

def main():
 torch.set_num_threads(1);os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});rows=[]
 cases=[(4,1,128,64,32,32,False),(8,1,512,128,64,64,True),(8,16,512,128,64,64,True),(4,128,512,128,32,32,True),(8,1,2048,512,128,128,True),(2,7,257,96,40,24,True),(4,32,512,32,128,64,False)]
 for case in cases:
  h,m,n,l,d,dv,shared=case;name=f'h{h}_m{m}_n{n}_l{l}_d{d}_v{dv}_s{int(shared)}';source=Path('research/results/native-projections')/name
  w=Path('research/results/portfolio')/name;w.mkdir(parents=True,exist_ok=True)
  model=NativeProjected(shared);cal=inputs(case,123);hold=inputs(case,789)
  with torch.inference_mode():references=[model(*[x.double() for x in data]).numpy() for data in [cal,hold]]
  aot.export(torch.export.export(model,cal)).save_mlir(str(w/'torch.mlir'))
  run(['iree-opt',w/'torch.mlir','--torch-function-to-torch-backend-pipeline','--torch-backend-to-linalg-on-tensors-backend-pipeline','--canonicalize','-o',w/'baseline.mlir'],w/'lower.log')
  source=w
  row={'case':name,'shape':case,'calibration_seed':123,'holdout_seed':789,'variants':{}};loaded={}
  for v,project,force in [('original',False,False),('projected-online',True,False),('materialized',False,True),('projected-materialized',True,True)]:
   p=w/(v+'-input.mlir');p.write_text(contract((source/'baseline.mlir').read_text(),project,force))
   lowered=w/(v+'.mlir');passes=['--attention-project-reassociate'] if project else []
   if force:passes+=['--attention-cpu-safe-lowering']
   run(['build/gqa-opt',p,'--allow-unregistered-dialect',*passes,'--canonicalize','-o',lowered],w/(v+'-pass.log'))
   try:
    if force:
     compile_s=run(['iree-compile',lowered,'--iree-hal-target-backends=llvm-cpu','--iree-llvmcpu-target-cpu=host',*STRONG,'-o',w/(v+'.vmfb')],w/(v+'-compile.log'));guarded=0
    else:
     start=time.perf_counter();guarded=compile_guarded(lowered,w/(v+'.vmfb'),w,v);compile_s=time.perf_counter()-start
    checks=[];devices=[]
    for data,reference in zip([cal,hold],references):
     cfg,ctx,fn,args=load(w/(v+'.vmfb'),data);before=cfg.device.allocator.statistics;y=fn(*args).to_host();after=cfg.device.allocator.statistics
     correct=bool(np.allclose(y,reference,atol=5e-5,rtol=2e-3));checks.append(dict(correct=correct,max_abs=float(np.max(np.abs(y-reference))),relative_l2=float(np.linalg.norm(y-reference)/np.linalg.norm(reference)),allocated_bytes=after['device_bytes_allocated']-before['device_bytes_allocated']));devices.append((cfg,ctx,fn,args))
    record=dict(compiled=True,compile_s=compile_s,guarded_ops=guarded,checks=checks,ir_sha256=hashlib.sha256(lowered.read_bytes()).hexdigest());row['variants'][v]=record
    if all(c['correct'] for c in checks):loaded[v]=devices
   except Exception as e:row['variants'][v]=dict(compiled=False,error=str(e))
  assert 'original' in loaded,(name,row)
  # Candidate selection uses calibration input only; holdout seed is never timed
  # until after the choice is frozen. All candidate checks are retained.
  calibration={v:[] for v in loaded};rng=random.Random(32)
  for devices in loaded.values():
   _,_,fn,args=devices[0]
   for _ in range(3):fn(*args)
  for _ in range(7):
   order=list(loaded);rng.shuffle(order)
   for v in order:
    _,_,fn,args=loaded[v][0];calibration[v].append(elapsed(fn,args,1))
  selected=min(calibration,key=lambda v:statistics.median(calibration[v]));row['selected']=selected;row['calibration_samples_us']=calibration
  holdout={v:[] for v in loaded}
  for devices in loaded.values():
   _,_,fn,args=devices[1]
   for _ in range(5):fn(*args)
  for _ in range(21):
   order=list(loaded);rng.shuffle(order)
   for v in order:
    _,_,fn,args=loaded[v][1];holdout[v].append(elapsed(fn,args,1))
  row['holdout_samples_us']=holdout;row['holdout_median_us']={v:statistics.median(s) for v,s in holdout.items()};row['speedup']=row['holdout_median_us']['original']/row['holdout_median_us'][selected];row['holdout_oracle']=min(row['holdout_median_us'],key=row['holdout_median_us'].get)
  rows.append(row);Path('research/results/portfolio-study.json').write_text(json.dumps({'cpu':min(os.sched_getaffinity(0)),'rows':rows,'method':'Two deterministic seeds, selection based on calibration timing; correctness and independent holdout timing retained. Not a universal equivalence proof.'},indent=2));print(name,'selected',selected,'speedup=%.3f'%row['speedup'],flush=True)
 print('Completed',len(rows),flush=True)
if __name__=='__main__':main()
