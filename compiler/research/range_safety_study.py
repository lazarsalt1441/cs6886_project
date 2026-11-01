"""Contrast permissive reassociation and bound-conditioned range-safe mode."""
import sys,json,torch,numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from projection_study import ROOT,ProjectedAttention,create_inputs,annotate,STRONG
from run_suite import run,load
from iree.turbine import aot
from iree.compiler import ir

def add_bounds(text,bounds):
 with ir.Context(), ir.Location.unknown():
  m=ir.Module.parse(text)
  for op in m.body.operations:
   if op.operation.name=='func.func':
    op.attributes['attention.require_range_safety']=ir.UnitAttr.get()
    op.attributes['attention.input_abs_bounds']=ir.ArrayAttr.get([ir.FloatAttr.get(ir.F64Type.get(),float(b)) for b in bounds])
  return str(m)
def main():
 torch.set_num_threads(1);out=[]
 nominal=create_inputs((4,1,128,64,32,32,False,False))
 # Finite inputs do not imply finite intermediate products after reassociation.
 q=torch.full((1,1,2),1e20);c=torch.ones(1,128,2)
 wk=torch.tensor([[[1e20,1e20],[-1e20,-1e20]]]);wv=torch.eye(2).unsqueeze(0)
 overflow=(q,c,wk,wv)
 for kind,x in [('nominal',nominal),('overflow-counterexample',overflow)]:
  w=Path('research/results/range-safety')/kind;w.mkdir(parents=True,exist_ok=True);model=ProjectedAttention()
  aot.export(torch.export.export(model,x)).save_mlir(str(w/'torch.mlir'))
  run(['iree-opt',w/'torch.mlir','--torch-function-to-torch-backend-pipeline','--torch-backend-to-linalg-on-tensors-backend-pipeline','--canonicalize','-o',w/'baseline.mlir'],w/'lower.log')
  baseline=(w/'baseline.mlir').read_text();contract=annotate(baseline)
  (w/'permissive-input.mlir').write_text(contract)
  bounds=[float(a.abs().max())*1.01 for a in x]
  (w/'safe-input.mlir').write_text(add_bounds(contract,bounds))
  with torch.inference_mode():ref=model(*x).numpy()
  for mode in ['permissive','safe']:
   run(['build/gqa-opt',w/(mode+'-input.mlir'),'--attention-project-reassociate','--canonicalize','-o',w/(mode+'.mlir')],w/(mode+'-pass.log'))
   text=(w/(mode+'.mlir')).read_text();placements=text.count('attention.projection_placement')
   run(['iree-compile',w/(mode+'.mlir'),'--iree-hal-target-backends=llvm-cpu','--iree-llvmcpu-target-cpu=host',*STRONG,'-o',w/(mode+'.vmfb')],w/(mode+'-compile.log'))
   cfg,ctx,fn,args=load(w/(mode+'.vmfb'),x);y=fn(*args).to_host();finite=bool(np.all(np.isfinite(y)));correct=bool(np.allclose(y,ref,atol=4e-5,rtol=1e-3))
   row=dict(case=kind,mode=mode,bounds=bounds,placements=placements,finite=finite,correct=correct,max_abs=float(np.max(np.abs(y-ref))) if finite else None);out.append(row);print(row,flush=True)
   if mode=='safe':assert finite and correct
   if kind=='overflow-counterexample' and mode=='safe':assert placements==0
   if kind=='overflow-counterexample' and mode=='permissive':assert not finite
 Path('research/results/range-safety-study.json').write_text(json.dumps(out,indent=2))
if __name__=='__main__':main()
