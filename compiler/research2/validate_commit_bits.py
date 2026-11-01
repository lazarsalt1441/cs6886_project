"""Bitwise winner-commit identity, runtime offset, strides, NaN payloads."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_commit_factoring import fixture
from winner_study import factor
from study_cache import run,STRONG
import numpy as np,iree.runtime as rt

def main():
 rows=[];rng=np.random.default_rng(314)
 for dtype,npdtype in [('f32',np.float32),('f16',np.float16),('i32',np.int32)]:
  for stride in [1,2]:
   w=Path('research2/results/commit-bits')/f'{dtype}-s{stride}';w.mkdir(parents=True,exist_ok=True);text=fixture(dtype=dtype,dynamic=True,strided=stride==2).replace(',%p:index',',%position:tensor<i64>').replace(' %wa=', ' %pv=tensor.extract %position[]:tensor<i64>\n %p=arith.index_cast %pv:i64 to index\n %wa=',1);(w/'baseline.mlir').write_text(text);factor(w/'baseline.mlir',w/'factored.mlir',w/'factor.log')
   for variant in ['baseline','factored']:
    run(['iree-compile',w/(variant+'.mlir'),'--iree-hal-target-backends=llvm-cpu','--iree-llvmcpu-target-cpu=host',*STRONG,'-o',w/(variant+'.vmfb')],w/(variant+'-compile.log'));cfg=rt.Config('local-sync');ctx=rt.SystemContext(config=cfg);ctx.add_vm_module(rt.VmModule.copy_buffer(cfg.vm_instance,(w/(variant+'.vmfb')).read_bytes()))
    for pos in [0,17,61]:
     for condition in [False,True]:
      base=rng.integers(-99,99,(64,2,2)).astype(npdtype);a=rng.integers(-99,99,(2,2,2)).astype(npdtype);b=rng.integers(-99,99,(2,2,2)).astype(npdtype)
      if dtype in ['f32','f16']:
       uint=np.uint32 if dtype=='f32' else np.uint16;values=[0x7fc00123,0x80000000,0x7f800000] if dtype=='f32' else [0x7e01,0x8000,0x7c00]
       a.view(uint).flat[:3]=values;b.view(uint).flat[:3]=[values[0]+7,values[1],values[2]]
      ref=base.copy();ref[pos:pos+stride+1:stride]=a if condition else b
      args=[rt.asdevicearray(cfg.device,x.copy()) for x in [base,base.copy(),a,b,np.array(condition,dtype=np.bool_),np.array(pos,dtype=np.int64)]];actual=ctx.modules.module.main(*args).to_host();assert actual.tobytes()==ref.tobytes(),(dtype,stride,variant,pos,condition);rows.append({'dtype':dtype,'stride':stride,'variant':variant,'position':pos,'condition':condition,'bitwise_equal':True})
 Path('research2/results/commit-bits.json').write_text(json.dumps(rows,indent=2));print('Completed',len(rows),flush=True)
if __name__=='__main__':main()
