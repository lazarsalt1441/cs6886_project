"""Independent key/value decisions, forced absorption, sharing and range guards."""
import sys,os,json,time,statistics,random
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from projection_study import ROOT,create_inputs,ProjectedAttention,annotate,STRONG
from rotary_projection_study import RotaryProjected,rotate
from range_safety_study import add_bounds
from run_suite import run,load
from iree.compiler import ir
import torch,numpy as np

def flags(text,names):
 with ir.Context():
  m=ir.Module.parse(text)
  for op in m.body.operations:
   if op.operation.name=='func.func':
    for name in names:op.attributes['attention.'+name]=ir.UnitAttr.get()
  return str(m)
def main():
 torch.set_num_threads(1);os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});out=[]
 cases=[('dense',(8,1,512,128,64,64,True,False)),('dense',(4,64,512,128,32,32,True,False)),('dense',(4,128,512,128,32,32,True,False)),('dense',(4,512,512,128,32,32,True,False)),('rotary',(4,1,128,64,32,32,16)),('rotary',(16,1,1024,512,128,128,64))]
 for kind,case in cases:
  if kind=='dense':
   h,m,n,l,d,dv,shared,masked=case;name=f'h{h}_m{m}_n{n}_l{l}_d{d}_v{dv}_s{int(shared)}_mask{int(masked)}';source=Path('research/results/projections')/name;x=create_inputs(case);model=ProjectedAttention(shared,masked)
  else:
   h,m,n,l,d,dv,r=case;name=f'h{h}_m{m}_n{n}_l{l}_d{d}_v{dv}_r{r}';source=Path('research/results/rotary-projections')/name;x=create_inputs((h,m,n,l,d,dv,True,False));x=(*x,rotate(torch.randn(h,m,r),torch.arange(n-m,n)),rotate(torch.randn(1,n,r),torch.arange(n)));model=RotaryProjected()
  w=Path('research/results/ablations')/(kind+'-'+name);w.mkdir(parents=True,exist_ok=True);baseline=(source/'baseline.mlir').read_text();contract=annotate(baseline)
  with torch.inference_mode():ref=model(*[a.double() for a in x]).numpy()
  variants=[('baseline',None),('default',[]),('key-only',['disable_value_reassociation']),('value-only',['disable_key_reassociation']),('always',['force_reassociation']),('no-sharing',['disable_shared_concat']),('range-safe',[])]
  row=dict(case=kind+'-'+name,variants={});loaded={}
  for v,attributes in variants:
   p=w/(v+'.mlir')
   if attributes is None:p.write_text(baseline)
   else:
    annotated=flags(contract,attributes)
    if v=='range-safe':annotated=add_bounds(annotated,[float(a.abs().max())*1.01 for a in x])
    inputfile=w/(v+'-input.mlir');inputfile.write_text(annotated)
    run(['build/gqa-opt',inputfile,'--attention-project-reassociate','--canonicalize','-o',p],w/(v+'-pass.log'))
   run(['iree-compile',p,'--iree-hal-target-backends=llvm-cpu','--iree-llvmcpu-target-cpu=host',*STRONG,'-o',w/(v+'.vmfb')],w/(v+'-compile.log'))
   cfg,ctx,fn,args=load(w/(v+'.vmfb'),x);before=cfg.device.allocator.statistics;y=fn(*args).to_host();after=cfg.device.allocator.statistics
   np.testing.assert_allclose(y,ref,atol=5e-5,rtol=2e-3)
   row['variants'][v]=dict(max_abs=float(np.max(np.abs(y-ref))),allocated_bytes=after['device_bytes_allocated']-before['device_bytes_allocated'],placements=p.read_text().count('attention.projection_placement'));loaded[v]=(cfg,ctx,fn,args)
  samples={v:[] for v in loaded};rng=random.Random(7);inner=1 if m>=128 else 5
  for _,_,fn,args in loaded.values():
   for _ in range(3):fn(*args)
  for _ in range(15):
   order=list(loaded);rng.shuffle(order)
   for v in order:
    _,_,fn,args=loaded[v];start=time.perf_counter_ns()
    for _ in range(inner):fn(*args)
    samples[v].append((time.perf_counter_ns()-start)/inner/1000)
  row['samples_us']=samples;row['median_us']={v:statistics.median(s) for v,s in samples.items()};out.append(row);Path('research/results/placement-ablations.json').write_text(json.dumps(out,indent=2));print(row['case'],row['median_us'],flush=True)
 print('Completed',len(out),flush=True)
if __name__=='__main__':main()
