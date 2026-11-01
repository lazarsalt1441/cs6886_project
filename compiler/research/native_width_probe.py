"""Isolate native attention feature widths; numpy reference, no PyTorch required.
Runs original native, transformed native, and decomposed fallback on identical
fresh inputs. This is diagnosis and fallback evaluation, not a root-cause fix.
"""
import argparse, hashlib, json, math, random, statistics, time
from pathlib import Path
import numpy as np
import iree.compiler as compiler
import iree.runtime as rt

ROOT=Path(__file__).resolve().parents[1]
FLAGS=['--iree-opt-level=O3','--iree-opt-generalize-matmul',
       '--iree-dispatch-creation-enable-aggressive-reshape-movement']

def attention_ir(d,dv):
    return f'''module {{
 func.func @main(%q: tensor<1x1x{d}xf32>, %k: tensor<1x32x{d}xf32>, %v: tensor<1x32x{dv}xf32>) -> tensor<1x1x{dv}xf32> {{
 %z = arith.constant 0.0 : f32
 %e = tensor.empty() : tensor<1x1x{dv}xf32>
 %i = linalg.fill ins(%z : f32) outs(%e : tensor<1x1x{dv}xf32>) -> tensor<1x1x{dv}xf32>
 %r = tm_tensor.attention ins(%q, %k, %v : tensor<1x1x{d}xf32>, tensor<1x32x{d}xf32>, tensor<1x32x{dv}xf32>) outs(%i : tensor<1x1x{dv}xf32>) -> tensor<1x1x{dv}xf32>
 return %r : tensor<1x1x{dv}xf32>
 }}
}}'''

def ref_attention(q,k,v):
    s=q.astype('float64')@k.astype('float64').swapaxes(-1,-2)/math.sqrt(q.shape[-1])
    p=np.exp(s-s.max(axis=-1,keepdims=True));p/=p.sum(axis=-1,keepdims=True)
    return p@v.astype('float64')

def compile_load(source,stem,a):
    stem.parent.mkdir(parents=True,exist_ok=True)
    flags=FLAGS+(['--iree-cuda-target='+a.arch] if a.device=='cuda' else ['--iree-llvmcpu-target-cpu=host'])
    blob=compiler.compile_str(source,target_backends=['cuda' if a.device=='cuda' else 'llvm-cpu'],extra_args=flags)
    stem.with_suffix('.vmfb').write_bytes(blob)
    cfg=rt.Config(a.device);ctx=rt.SystemContext(config=cfg)
    ctx.add_vm_module(rt.VmModule.copy_buffer(cfg.vm_instance,blob))
    return cfg,ctx,ctx.modules.module.main,hashlib.sha256(blob).hexdigest()

def check(fn,cfg,x,ref):
    got=fn(*[rt.asdevicearray(cfg.device,z) for z in x]).to_host()
    return dict(ok=bool(np.allclose(got,ref,atol=5e-5,rtol=2e-3)),max_abs=float(np.max(np.abs(got-ref))))

def main():
    p=argparse.ArgumentParser();p.add_argument('--arch',default='sm_75');p.add_argument('--device',default='cuda');p.add_argument('--seeds',type=int,default=10);p.add_argument('--trials',type=int,default=101);p.add_argument('--output',default='followup_results');a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);rows=[]
    def save(row):
        rows.append(row);(out/'probe.json').write_text(json.dumps(rows,indent=2));print(row['case'],row.get('status'),flush=True)
    for d,dv in [(64,64),(128,128),(256,128),(512,128),(128,512),(512,512)]:
        name=f'd{d}-v{dv}';source=attention_ir(d,dv);(out/(name+'.mlir')).write_text(source)
        row=dict(case=name,device=a.device,arch=a.arch,checks=[])
        try:
            cfg,ctx,fn,sha=compile_load(source,out/name,a);row['sha256']=sha
            for seed in range(101,101+a.seeds):
                rng=np.random.default_rng(seed)
                for mode in ['random','tail-only']:
                    q=rng.normal(size=(1,1,d)).astype('float32');k=rng.normal(size=(1,32,d)).astype('float32');v=rng.normal(size=(1,32,dv)).astype('float32')
                    if mode=='tail-only':q[...,:-1]=0;k[...,:-1]=0;q[...,-1]=math.sqrt(d)
                    row['checks'].append(dict(seed=seed,mode=mode,**check(fn,cfg,[q,k,v],ref_attention(q,k,v))))
            row['status']='PASS' if all(c['ok'] for c in row['checks']) else 'FAIL'
        except Exception as e:row.update(status='ERROR',error=str(e))
        save(row)
    name='h8_m1_n2048_l512_d128_v128_s1'
    sources={v:ROOT/'research/results/native-projections'/name/(v+'.mlir') for v in ['baseline','optimized']}
    sources['decomposed-fallback']=ROOT/'research/results/projections'/(name+'_mask0')/'optimized.mlir'
    loaded={}
    for variant,path in sources.items():
        row=dict(case=name,variant=variant,checks=[])
        try:
            cfg,ctx,fn,sha=compile_load(path.read_text(),out/('projection-'+variant),a);row['sha256']=sha
            for seed in range(101,101+a.seeds):
                rng=np.random.default_rng(seed)
                x=[rng.normal(size=s).astype('float32') for s in [(8,1,128),(1,2048,512),(8,512,128),(8,512,128)]]
                x[2]/=math.sqrt(512);x[3]/=math.sqrt(512)
                q,c,wk,wv=x;ref=ref_attention(q,c@wk,c@wv)
                row['checks'].append(dict(seed=seed,**check(fn,cfg,x,ref)))
            row['status']='PASS' if all(c['ok'] for c in row['checks']) else 'FAIL'
            if row['status']=='PASS':loaded[variant]=(cfg,ctx,fn,[rt.asdevicearray(cfg.device,z) for z in x])
        except Exception as e:row.update(status='ERROR',error=str(e))
        save(row)
    def invoke(item):
        _,_,fn,dev=item;fn(*dev).to_host()
    from gpu_selection import calibrate
    if 'baseline' in loaded:
        for item in loaded.values():
            for _ in range(10):invoke(item)
        selection=calibrate({v:lambda item=item:invoke(item) for v,item in loaded.items()},'baseline')
        samples={v:[] for v in loaded};rng=random.Random(813)
        for _ in range(a.trials):
            order=list(loaded);rng.shuffle(order)
            for v in order:
                start=time.perf_counter_ns();invoke(loaded[v]);samples[v].append((time.perf_counter_ns()-start)/1000)
        med={v:statistics.median(s) for v,s in samples.items()}
        (out/'timing.json').write_text(json.dumps(dict(calibration=selection,samples_us=samples,medians_us=med,selected_holdout_speedup=med['baseline']/med[selection['selected']],note='Invocation + execution + output download; inputs resident; compilation excluded.'),indent=2))
    # Retain the native failure as a finding; fail only if baseline/fallback fails.
    required=[r for r in rows if r.get('variant') in ['baseline','decomposed-fallback']]
    if len(required)!=2 or any(r['status']!='PASS' for r in required):raise SystemExit(1)

if __name__=='__main__':main()
