"""Rank1–5 row and strided-region forwarding, dynamic offsets, bitwise checks."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from study_cache import forward,run,STRONG
from add_patch_variant import patch
import numpy as np,iree.runtime as rt

def fixture(rank,dtype,strided,native):
 shape=[65]+[3+(i%2) for i in range(rank-1)];small=[3]+([1]*(rank-1) if strided else shape[1:]);out=[7]+shape[1:];typ=lambda s:'tensor<'+'x'.join(map(str,s))+'x'+dtype+'>';base,update,result=map(typ,[shape,small,out]);dims=','.join('d'+str(i) for i in range(rank));maps=f'#i=affine_map<({dims})->(d0)>\n#o=affine_map<({dims})->({dims})>\n';integer='i32' if rank%2==0 else 'i64';idtype=f'tensor<7x{integer}>';offs=['%slot']+(['1']*(rank-1) if strided else ['0']*(rank-1));steps=[2]*rank if strided else [1]*rank
 s=maps+f'module {{ func.func @main(%base:{base},%a:{update},%b:{update},%ids:{idtype},%position:tensor<i64>) -> ({result},{result}) {{\n %pos=tensor.extract %position[] : tensor<i64>\n %slot=arith.index_cast %pos : i64 to index\n'
 for name in ['a','b']:s+=f' %w{name}=tensor.insert_slice %{name} into %base[{",".join(offs)}] {small} {steps} : {update} into {base}\n'
 s+=f' %e=tensor.empty() : {result}\n'
 for name in ['a','b']:
  if native:s+=f' %r{name}=iree_linalg_ext.gather dimension_map=[0] ins(%w{name},%ids:{base},{idtype}) outs(%e:{result}) -> {result}\n'
  else:
   s+=f' %r{name}=linalg.generic {{indexing_maps=[#i,#o],iterator_types=[{",".join(chr(34)+"parallel"+chr(34) for _ in range(rank))}]}} ins(%ids:{idtype}) outs(%e:{result}) {{\n ^bb0(%id:{integer},%out:{dtype}):\n %p=arith.index_cast %id : {integer} to index\n'
   for j in range(1,rank):s+=f' %d{j}=linalg.index {j} : index\n'
   s+=f' %v=tensor.extract %w{name}[%p'+''.join(',%d'+str(j) for j in range(1,rank))+f'] : {base}\n linalg.yield %v:{dtype}\n }} -> {result}\n'
 s+=f' return %ra,%rb : {result},{result}\n }} }}';return s,shape,small,integer

def main():
 rows=[];rng=np.random.default_rng(1762)
 for rank in range(1,6):
  for dtype,npdtype in [('f32',np.float32),('f16',np.float16),('i32',np.int32)]:
   for strided in [False,True]:
    w=Path('research2/results/ranks')/f'r{rank}-{dtype}-{strided}';w.mkdir(parents=True,exist_ok=True)
    text,shape,small,integer=fixture(rank,dtype,strided,False);(w/'baseline.mlir').write_text(text);native,*_=fixture(rank,dtype,strided,True);(w/'native.mlir').write_text(native)
    forward(w/'baseline.mlir',w/'forwarded.mlir',w/'forward.log');patch(w/'native.mlir',w/'masked.mlir',w/'masked.log',ratio=4)
    assert (w/'forwarded.mlir').read_text().count('cache_read_forwarded')==2
    assert (w/'masked.mlir').read_text().count('cache_gather_forwarded')==(0 if strided else 2)
    loaded={}
    for variant in ['baseline','forwarded','masked']:
     run(['iree-compile',w/(variant+'.mlir'),'--iree-hal-target-backends=llvm-cpu','--iree-llvmcpu-target-cpu=host',*STRONG,'-o',w/(variant+'.vmfb')],w/(variant+'-compile.log'))
     cfg=rt.Config('local-sync');ctx=rt.SystemContext(config=cfg);ctx.add_vm_module(rt.VmModule.copy_buffer(cfg.vm_instance,(w/(variant+'.vmfb')).read_bytes()));loaded[variant]=(cfg,ctx,ctx.modules.module.main)
    for pos in ([0,31,60] if strided else [0,31,62]):
     arrays=[rng.integers(-99,99,shape).astype(npdtype),rng.integers(-99,99,small).astype(npdtype),rng.integers(-99,99,small).astype(npdtype)]
     if dtype=='f32':
      for a in arrays:
       a.view(np.uint32).flat[:min(a.size,3)]=[0x7fc00123,0x80000000,0x7f800000][:min(a.size,3)]
     ids=np.array([0,pos,pos+1,pos+2,64,31,60],dtype=np.int32 if integer=='i32' else np.int64)
     slices=tuple([slice(pos,pos+5,2)]+[slice(1,2,2)]*(rank-1)) if strided else tuple([slice(pos,pos+3)]+[slice(None)]*(rank-1))
     refs=[]
     for a in arrays[1:]:v=arrays[0].copy();v[slices]=a;refs.append(v[ids])
     for variant,(cfg,ctx,fn) in loaded.items():
      dev=[rt.asdevicearray(cfg.device,a.copy()) for a in [*arrays,ids,np.array(pos,dtype=np.int64)]];ys=fn(*dev);assert len(ys)==len(refs)
      assert all(y.to_host().tobytes()==ref.tobytes() for y,ref in zip(ys,refs)),(rank,dtype,strided,pos,variant)
      rows.append({'rank':rank,'dtype':dtype,'strided':strided,'position':pos,'variant':variant,'bitwise_equal':True})
    Path('research2/results/rank-semantics.json').write_text(json.dumps(rows,indent=2));print(rank,dtype,strided,'passed',flush=True)
 print('Completed',len(rows),flush=True)
if __name__=='__main__':main()
