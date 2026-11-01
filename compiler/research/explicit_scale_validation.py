"""Execute explicit zero/negative/positive native scales through projection and fallback."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from projection_study import ROOT,create_inputs,annotate
from run_suite import run,load
from cpu_compile import compile_guarded
from iree.turbine import aot
from iree.compiler import ir
import torch,numpy as np
class Attention(torch.nn.Module):
 def __init__(self,scale):super().__init__();self.scale=scale
 def forward(self,q,c,wk,wv):
  c=c.expand(q.shape[0],-1,-1)
  return torch.nn.functional.scaled_dot_product_attention(q.unsqueeze(0),torch.bmm(c,wk).unsqueeze(0),torch.bmm(c,wv).unsqueeze(0),enable_gqa=True,scale=self.scale).squeeze(0)
def main():
 torch.set_num_threads(1);rows=[]
 for scale in [.125,-.25,0.0]:
  w=Path('research/results/explicit-scale')/str(scale);w.mkdir(parents=True,exist_ok=True)
  x=create_inputs((4,2,128,64,32,32,True,False));model=Attention(scale)
  with torch.inference_mode():ref=model(*[a.double() for a in x]).numpy()
  aot.export(torch.export.export(model,x)).save_mlir(str(w/'torch.mlir'))
  run(['iree-opt',w/'torch.mlir','--torch-function-to-torch-backend-pipeline','--torch-backend-to-linalg-on-tensors-backend-pipeline','--canonicalize','-o',w/'baseline.mlir'],w/'export.log')
  with ir.Context():
   module=ir.Module.parse(annotate((w/'baseline.mlir').read_text()).replace('attention.allow_reassociation','attention.enable_native_projection, attention.allow_reassociation'))
   (w/'contract.mlir').write_text(module.operation.get_asm(print_generic_op_form=True))
  run(['build/gqa-opt',w/'contract.mlir','--allow-unregistered-dialect','--attention-project-reassociate','--canonicalize','-o',w/'projected.mlir'],w/'projection.log')
  assert 'native-both' in (w/'projected.mlir').read_text()
  text=(w/'projected.mlir').read_text().replace('attention.allow_reassociation','attention.force_safe_attention_lowering, attention.allow_reassociation');(w/'fallback-input.mlir').write_text(text)
  run(['build/gqa-opt',w/'fallback-input.mlir','--allow-unregistered-dialect','--attention-cpu-safe-lowering','--canonicalize','-o',w/'fallback.mlir'],w/'fallback-pass.log')
  row={'scale':scale,'variants':{}}
  for variant in ['baseline','projected','fallback']:
   compile_guarded(w/(variant+'.mlir'),w/(variant+'.vmfb'),w,variant)
   cfg,ctx,fn,dev=load(w/(variant+'.vmfb'),x);y=fn(*dev).to_host();err=float(np.max(np.abs(y-ref)));np.testing.assert_allclose(y,ref,atol=4e-5,rtol=1e-3);row['variants'][variant]={'max_abs':err,'correct':True}
  rows.append(row);print(row,flush=True)
 Path('research/results/explicit-scale-validation.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
