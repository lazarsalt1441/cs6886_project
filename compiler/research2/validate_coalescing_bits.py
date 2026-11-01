"""Exact coalescing identity over ranks1–5, append/prepend, four dtypes."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_update_coalescing import fixture
from coalesce_study import coalesce
from study_cache import run,STRONG
import numpy as np,iree.runtime as rt,ml_dtypes

def main():
 rng=np.random.default_rng(862);rows=[]
 for rank in range(1,6):
  for dtype,npdtype in [('f32',np.float32),('f16',np.float16),('bf16',ml_dtypes.bfloat16),('i32',np.int32)]:
   for second in [9,5]:
    w=Path('research2/results/coalesce-bits')/f'fork-r{rank}-{dtype}-o{second}';w.mkdir(parents=True,exist_ok=True);text=fixture(rank=rank,dtype=dtype,second=second);large='tensor<'+'x'.join(map(str,[64]+[4]*(rank-1)))+'x'+dtype+'>';small='tensor<'+'x'.join(map(str,[2]*rank))+'x'+dtype+'>';branch=f'%branch=tensor.insert_slice %a into %base[30'+',0'*(rank-1)+f'] {[2]*rank} {[1]*rank} : {small} into {large}\n return %outer,%branch';text=text.replace('return %outer,%base',branch);(w/'baseline.mlir').write_text(text);coalesce(w/'baseline.mlir',w/'coalesced.mlir',w/'pass.log');assert (w/'coalesced.mlir').read_text().count('cache_updates_coalesced')==1
    base=rng.integers(-99,99,[64]+[4]*(rank-1)).astype(npdtype);a=rng.integers(-99,99,[2]*rank).astype(npdtype);b=rng.integers(-99,99,[2]*rank).astype(npdtype)
    if dtype!='i32':
     uint=np.uint32 if dtype=='f32' else np.uint16;bits={'f32':[0x7fc00123,0x80000000,0x7f800000],'f16':[0x7e01,0x8000,0x7c00],'bf16':[0x7fc1,0x8000,0x7f80]}[dtype]
     for arr,delta in [(a,0),(b,7)]:
      n=min(arr.size,3);arr.view(uint).flat[:n]=[bits[0]+delta,*bits[1:]][:n]
    expected=base.copy();expected[(slice(7,9),)+tuple(slice(0,2) for _ in range(rank-1))]=a;expected[(slice(second,second+2),)+tuple(slice(0,2) for _ in range(rank-1))]=b
    other=base.copy();other[(slice(30,32),)+tuple(slice(0,2) for _ in range(rank-1))]=a
    for v in ['baseline','coalesced']:
     run(['iree-compile',w/(v+'.mlir'),'--iree-hal-target-backends=llvm-cpu','--iree-llvmcpu-target-cpu=host',*STRONG,'-o',w/(v+'.vmfb')],w/(v+'-compile.log'));cfg=rt.Config('local-sync');ctx=rt.SystemContext(config=cfg);ctx.add_vm_module(rt.VmModule.copy_buffer(cfg.vm_instance,(w/(v+'.vmfb')).read_bytes()));ys=ctx.modules.module.main(*[rt.asdevicearray(cfg.device,x.copy()) for x in [base,a,b]]);assert len(ys)==2
     assert ys[0].to_host().tobytes()==expected.tobytes(),(rank,dtype,second,v,'updated');assert ys[1].to_host().tobytes()==other.tobytes(),(rank,dtype,second,v,'other_fork');rows.append({'rank':rank,'dtype':dtype,'second_offset':second,'variant':v,'bitwise_equal':True})
    Path('research2/results/coalesce-bits.json').write_text(json.dumps(rows,indent=2));print(rank,dtype,second,'pass',flush=True)
 print('Completed',len(rows),flush=True)
if __name__=='__main__':main()
