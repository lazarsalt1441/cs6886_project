"""Bit-preserving cache observations, dynamic offsets, strides, overlapping versions.
Hand-written standard MLIR fixtures isolate data-movement semantics from softmax.
"""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from study_cache import ROOT,run,forward,normalize,STRONG
from lift_gather_baseline import lift
from add_patch_variant import patch
import numpy as np,iree.runtime as rt

def fixture(dtype='f32',dynamic=False,strided=False,nested=False):
 pool=257;n=32;h=2;d=4;t=3
 small='3x1x2' if strided else '3x2x4';large='257x2x4';output='32x2x4'
 maps='#i = affine_map<(d0,d1,d2)->(d0)>\n#o = affine_map<(d0,d1,d2)->(d0,d1,d2)>\n'
 pos='%slot' if dynamic else '37';args=f'%base: tensor<{large}x{dtype}>, %a: tensor<{small}x{dtype}>, %b: tensor<{small}x{dtype}>, %ids: tensor<32xi64>'
 if dynamic:args+=', %position: tensor<i64>'
 outputs=3 if nested else 2
 text=maps+f'module {{ func.func @main({args}) -> ('+', '.join([f'tensor<{output}x{dtype}>']*outputs)+') {\n'
 if dynamic:text+=' %pv = tensor.extract %position[] : tensor<i64>\n %slot = arith.index_cast %pv : i64 to index\n'
 off=f'[{pos}, 1, 1]' if strided else f'[{pos}, 0, 0]';size='[3, 1, 2]' if strided else '[3, 2, 4]';steps='[2, 1, 1]' if strided else '[1, 1, 1]'
 text+=f' %wa = tensor.insert_slice %a into %base{off} {size} {steps} : tensor<{small}x{dtype}> into tensor<{large}x{dtype}>\n'
 target='%wa' if nested else '%base';offb='[38, 0, 0]' if nested else off
 text+=f' %wb = tensor.insert_slice %b into {target}{offb} {size} {steps} : tensor<{small}x{dtype}> into tensor<{large}x{dtype}>\n'
 text+=f' %empty = tensor.empty() : tensor<{output}x{dtype}>\n'
 for i,tensor in enumerate(['%wa','%wb','%base'][:outputs]):
  text+=f''' %r{i} = linalg.generic {{indexing_maps=[#i,#o], iterator_types=["parallel","parallel","parallel"]}} ins(%ids: tensor<32xi64>) outs(%empty: tensor<{output}x{dtype}>) {{
  ^bb0(%in: i64,%out: {dtype}):
   %p = arith.index_cast %in : i64 to index
   %h = linalg.index 1 : index
   %d = linalg.index 2 : index
   %v = tensor.extract {tensor}[%p,%h,%d] : tensor<{large}x{dtype}>
   linalg.yield %v : {dtype}
 }} -> tensor<{output}x{dtype}>\n'''
 text+=' return '+', '.join('%r'+str(i) for i in range(outputs))+' : '+', '.join([f'tensor<{output}x{dtype}>']*outputs)+'\n } }\n';return text

def main():
 rows=[];rng=np.random.default_rng(71)
 for dtype in ['f32','f16','i32']:
  for mode in ['normal','dynamic','strided','nested']:
   dyn=mode=='dynamic';strided=mode=='strided';nested=mode=='nested';npdtype={'f32':np.float32,'f16':np.float16,'i32':np.int32}[dtype]
   w=Path('research2/results/semantics')/(dtype+'-'+mode);w.mkdir(parents=True,exist_ok=True);(w/'baseline.mlir').write_text(fixture(dtype,dyn,strided,nested));forward(w/'baseline.mlir',w/'forwarded.mlir',w/'pass.log')
   lift(w/'baseline.mlir',w/'lifted.mlir');normalize(w/'lifted.mlir',w/'normalized.mlir',w/'normalize.log');patch(w/'normalized.mlir',w/'patch-forwarded.mlir',w/'patch.log')
   binaries={}
   for variant in ['baseline','forwarded','patch-forwarded']:
    try:
     run(['iree-compile',w/(variant+'.mlir'),'--iree-hal-target-backends=llvm-cpu','--iree-llvmcpu-target-cpu=host',*STRONG,'-o',w/(variant+'.vmfb')],w/(variant+'-compile.log'));binaries[variant]=w/(variant+'.vmfb')
    except Exception as e:rows.append({'dtype':dtype,'mode':mode,'variant':variant,'compile_error':str(e)})
   assert len(binaries)==3,('Missing compiled semantic variant',dtype,mode)
   for pos in ([0,37,254] if dyn else [37]):
    base=np.arange(257*2*4,dtype=npdtype).reshape(257,2,4)
    small=(3,1,2) if strided else (3,2,4);a=rng.integers(-99,99,small).astype(npdtype);b=rng.integers(-199,-100,small).astype(npdtype)
    if dtype=='f32':
     # Include NaNs with payloads, signed zero, and infinities. No FP operations
     # are needed by this transform, so all selected bits must be preserved.
     a.view(np.uint32).flat[:5]=[0x7fc00123,0x80000000,0x7f800000,0xff800000,0x00000000]
     b.view(np.uint32).flat[:5]=[0x7fc00456,0x80000000,0x00000000,0x7f800000,0xff800000]
    ids=rng.integers(0,257,32,dtype=np.int64);ids[:9]=[pos,pos+1,pos+2,min(pos+3,256),max(pos-1,0),0,256,38,40]
    aa=base.copy();ss=(slice(pos,pos+5,2),slice(1,2),slice(1,3)) if strided else (slice(pos,pos+3),slice(None),slice(None));aa[ss]=a
    bb=aa.copy() if nested else base.copy();bs=(slice(38,41),slice(None),slice(None)) if nested else ss;bb[bs]=b
    refs=[aa[ids],bb[ids]]+([base[ids]] if nested else [])
    for variant,path in binaries.items():
     cfg=rt.Config(device=rt.get_driver('local-sync').create_default_device());vm=rt.VmModule.copy_buffer(cfg.vm_instance,path.read_bytes());ctx=rt.SystemContext(config=cfg);ctx.add_vm_module(vm)
     arrays=[base.copy(),a.copy(),b.copy(),ids.copy()]+([np.array(pos,dtype=np.int64)] if dyn else []);args=[rt.asdevicearray(cfg.device,x) for x in arrays]
     result=ctx.modules.module.main(*args);ys=[result] if hasattr(result,'to_host') else list(result)
     bits_equal=all(y.to_host().tobytes()==ref.tobytes() for y,ref in zip(ys,refs));row={'dtype':dtype,'mode':mode,'position':pos,'variant':variant,'bitwise_equal':bits_equal};rows.append(row);print(row,flush=True);assert bits_equal,row
 Path('research2/results/cache-semantics.json').write_text(json.dumps(rows,indent=2));print('Completed',len(rows),flush=True)
if __name__=='__main__':main()
