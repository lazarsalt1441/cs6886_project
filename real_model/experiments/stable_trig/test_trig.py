import os,sys,subprocess,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent/'trig_test';OUT.mkdir(exist_ok=True)
import numpy as np,iree.runtime as rt
source='''module {func.func @main(%x:tensor<64xf32>) -> (tensor<64xf32>,tensor<64xf32>) {
 %e=tensor.empty():tensor<64xf32>
 %a=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]} ins(%x:tensor<64xf32>) outs(%e:tensor<64xf32>){^bb0(%v:f32,%o:f32):
 %s=math.sin %v:f32
 linalg.yield %s:f32
 } -> tensor<64xf32>
 %b=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]} ins(%x:tensor<64xf32>) outs(%e:tensor<64xf32>){^bb0(%v:f32,%o:f32):
 %s=math.cos %v:f32
 linalg.yield %s:f32
 } -> tensor<64xf32>
 return %a,%b:tensor<64xf32>,tensor<64xf32>
}}'''
(OUT/'base.mlir').write_text(source);sdk=ROOT/'compiler/toolchain/usr/lib';env=os.environ.copy();env['LD_LIBRARY_PATH']=str(sdk/'llvm-19/lib')+':'+str(sdk/'x86_64-linux-gnu')
cmd=[str(ROOT/'compiler/build/gqa-opt'),'--attention-stable-trig','--canonicalize'];r=subprocess.run(cmd,input=source,text=True,capture_output=True,env=env,check=True);(OUT/'fixed.mlir').write_text(r.stdout)
assert r.stdout.count('attention.stable_trig')==2
assert subprocess.run(cmd,input=r.stdout,text=True,capture_output=True,env=env,check=True).stdout==r.stdout
# Lowering-config protected kernels are rejected.
neg=source.replace('indexing_maps=[','lowering_config={}, indexing_maps=[')
assert 'attention.stable_trig' not in subprocess.run(cmd,input=neg,text=True,capture_output=True,env=env,check=True).stdout
for neg in [source.replace('f32','f64'),source.replace('64xf32','8192xf32')]:
 assert 'attention.stable_trig' not in subprocess.run(cmd,input=neg,text=True,capture_output=True,env=env,check=True).stdout
funcs={}
for variant in ['base','fixed']:
 with (OUT/'compile.log').open('a') as log:subprocess.run(['iree-compile',str(OUT/(variant+'.mlir')),'--iree-hal-target-backends=llvm-cpu','--iree-llvmcpu-target-cpu=host','--iree-input-demote-f64-to-f32=false','--iree-opt-numeric-precision-reduction=false','-o',str(OUT/(variant+'.vmfb'))],check=True,stdout=log,stderr=subprocess.STDOUT)
 cfg=rt.Config('local-sync');ctx=rt.SystemContext(config=cfg);ctx.add_vm_module(rt.VmModule.mmap(cfg.vm_instance,str(OUT/(variant+'.vmfb'))));funcs[variant]=(ctx,ctx.modules.module.main)
rows=[]
for seed in range(20):
 x=np.random.default_rng(seed).uniform(-8192,8192,64).astype('float32');ref=[np.sin(x.astype('float64')),np.cos(x.astype('float64'))]
 vals={v:[z.to_host() for z in fn(x)] for v,(_,fn) in funcs.items()}
 error={v:max(float(np.max(abs(a-b))) for a,b in zip(out,ref)) for v,out in vals.items()}
 assert error['fixed']<2e-6,error
 rows.append(dict(seed=seed,**error))
# Fallback is unchanged for small/out-of-repair-domain finite inputs.
x=np.array([0.,-0.,1.,-1.,32.,-32.,1e5,-1e5]*8,'float32');a=funcs['base'][1](x);b=funcs['fixed'][1](x)
for u,v in zip(a,b):np.testing.assert_array_equal(u.to_host().view('u4'),v.to_host().view('u4'))
(OUT/'results.json').write_text(json.dumps(rows,indent=2));print('PASS real C++ rewrite, idempotence, lowering-config negative, 20 compiled numerical seeds and exact fallback')
