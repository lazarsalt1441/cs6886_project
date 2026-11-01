"""Runtime cache write positions in native GQA: hand-generalized Torch export."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from study_cache import *
from add_patch_variant import patch

def main():
 torch.set_num_threads(1);case=CASES[0];w=Path('research2/results/dynamic-attention');w.mkdir(exist_ok=True);base=Path('research2/results/study')/name(case)/'baseline.mlir';text=base.read_text()
 text=text.replace('%arg5: tensor<128xi64>)','%arg5: tensor<128xi64>, %position: tensor<i64>)',1)
 text=text.replace('    %cst =','    %pos = tensor.extract %position[] : tensor<i64>\n    %slot = arith.index_cast %pos : i64 to index\n    %cst =',1).replace('[37, 0, 0]','[%slot, 0, 0]');(w/'baseline.mlir').write_text(text)
 forward(w/'baseline.mlir',w/'forwarded.mlir',w/'forward.log');lift(w/'baseline.mlir',w/'lifted.mlir');normalize(w/'lifted.mlir',w/'lifted-normalized.mlir',w/'normalize.log');patch(w/'lifted-normalized.mlir',w/'patch-forwarded.mlir',w/'patch.log')
 rows=[]
 for variant,source in [('strong','baseline'),('lifted-normalized','lifted-normalized'),('forwarded','forwarded'),('patch-forwarded','patch-forwarded')]:
  run(['iree-compile',w/(source+'.mlir'),'--iree-hal-target-backends=llvm-cpu','--iree-llvmcpu-target-cpu=host',*STRONG,'-o',w/(variant+'.vmfb')],w/(variant+'-compile.log'))
  for pos in [0,37,4094]:
   c=list(case);c[7]=pos;x=inputs(c,seed=879);refs=reference(BranchAttention(c),x);cfg,ctx,fn,dev=load(w/(variant+'.vmfb'),[*[a.clone() for a in x],torch.tensor(pos)])
   result=fn(*dev);error=0
   for a,b in zip(result,refs):arr=a.to_host();np.testing.assert_allclose(arr,b,atol=4e-5,rtol=1e-3);error=max(error,float(np.max(np.abs(arr-b))))
   row={'variant':variant,'position':pos,'max_abs':error,'correct':True};rows.append(row);print(row,flush=True)
 Path('research2/results/dynamic-attention.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
