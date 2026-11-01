"""Original-cache snapshots: stock diagnostic vs two compiled C++ repairs."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from snapshot_study import protect
from repair_snapshot_cow import compile_repaired
from study_cache import run,STRONG
import numpy as np,iree.runtime as rt,ml_dtypes

def main():
 rng=np.random.default_rng(862);rows=[]
 for rank in range(1,6):
  for dtype,npdtype in [('f32',np.float32),('f16',np.float16),('bf16',ml_dtypes.bfloat16),('i32',np.int32)]:
   for second in [9,5]:
    w=Path('research2/results/snapshot-bits')/f'r{rank}-{dtype}-o{second}';w.mkdir(parents=True,exist_ok=True);shape=[64]+[4]*(rank-1);small=[2]+[4]*(rank-1);ty=lambda s:'tensor<'+'x'.join(map(str,s))+'x'+dtype+'>';large,update=ty(shape),ty(small)
    text=f'''module {{func.func @main(%base:{large},%a:{update},%b:{update})->({large},{large}) {{
 %inner=tensor.insert_slice %a into %base[7{',0'*(rank-1)}] {small} {[1]*rank}:{update} into {large}
 %outer=tensor.insert_slice %b into %inner[{second}{',0'*(rank-1)}] {small} {[1]*rank}:{update} into {large}
 return %outer,%base:{large},{large}
 }} }}''';(w/'baseline.mlir').write_text(text);protect(w/'baseline.mlir',w/'protected.mlir',w/'protect.log')
    base=rng.integers(-99,99,shape).astype(npdtype);a=rng.integers(-99,99,small).astype(npdtype);b=rng.integers(-99,99,small).astype(npdtype)
    if dtype!='i32':
     uint=np.uint32 if dtype=='f32' else np.uint16;bits={'f32':[0x7fc00123,0x80000000,0x7f800000],'f16':[0x7e01,0x8000,0x7c00],'bf16':[0x7fc1,0x8000,0x7f80]}[dtype]
     for arr,delta in [(a,0),(b,7)]:
      n=min(arr.size,3);arr.view(uint).flat[:n]=[bits[0]+delta,*bits[1:]][:n]
    expected=base.copy();expected[7:9]=a;expected[second:second+2]=b
    for v in ['baseline','protected','root-cow']:
     if v=='root-cow':count=compile_repaired(w/'baseline.mlir',w/(v+'.vmfb'));assert count==1
     else:run(['iree-compile',w/(v+'.mlir'),'--iree-hal-target-backends=llvm-cpu','--iree-llvmcpu-target-cpu=host',*STRONG,'-o',w/(v+'.vmfb')],w/(v+'-compile.log'))
     cfg=rt.Config('local-sync');ctx=rt.SystemContext(config=cfg);ctx.add_vm_module(rt.VmModule.copy_buffer(cfg.vm_instance,(w/(v+'.vmfb')).read_bytes()));dev=[rt.asdevicearray(cfg.device,x.copy()) for x in [base,a,b]];ys=ctx.modules.module.main(*dev);assert len(ys)==2
     updated=ys[0].to_host().tobytes()==expected.tobytes();snapshot=ys[1].to_host().tobytes()==base.tobytes();assert updated,(rank,dtype,second,v,'updated')
     if v!='baseline':assert snapshot,(rank,dtype,second,v,'snapshot')
     rows.append({'rank':rank,'dtype':dtype,'second_offset':second,'variant':v,'updated_bitwise_equal':updated,'snapshot_bitwise_equal':snapshot})
    Path('research2/results/snapshot-bits.json').write_text(json.dumps(rows,indent=2));print(rank,dtype,second,'pass repairs',flush=True)
 print('Completed',len(rows),flush=True);assert any(not r['snapshot_bitwise_equal'] for r in rows if r['variant']=='baseline')
if __name__=='__main__':main()
