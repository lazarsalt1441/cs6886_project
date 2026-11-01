"""Export, optimize, compile, check and benchmark each real PyTorch GQA graph."""
import argparse, csv, gc, json, os, re, statistics, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
import iree.runtime as rt
from iree.turbine import aot
from export_gqa import GQA, inputs

ROOT=Path(__file__).resolve().parents[1]; os.chdir(ROOT)
os.environ['PATH']=str(Path(sys.executable).parent)+':'+str(Path.home()/'.local/bin')+':'+os.environ['PATH']
os.environ['LD_LIBRARY_PATH']=str(ROOT/'toolchain/usr/lib/llvm-19/lib')+':'+str(ROOT/'toolchain/usr/lib/x86_64-linux-gnu')+':'+os.environ.get('LD_LIBRARY_PATH','')
# b,hq,hkv,query_len,kv_len,head_dim. Ratio-one is an unchanged control.
CASES=[(1,4,4,1,32,16),(1,4,2,1,64,16),(1,8,2,1,128,32),
 (2,8,2,1,128,32),(1,8,1,1,256,64),(1,16,2,1,512,64),
 (1,16,4,1,1024,64),(1,8,2,16,128,32),(2,8,2,16,256,64),
 (1,16,4,32,512,64),(1,8,2,1,4096,64),(1,16,2,1,8192,128),
 (1,6,2,3,37,24),(2,12,4,7,65,40),(1,8,2,64,256,64)]

def run(cmd,log):
    # Publish compiler artifacts atomically. Partial/empty artifacts were seen
    # during long runs; do not let a later stage consume them silently.
    import uuid
    cmd=[str(x) for x in cmd]
    output_index=cmd.index('-o')+1 if '-o' in cmd else -1
    for i,x in enumerate(cmd):
        if i!=output_index and x.endswith('.mlir') and Path(x).exists() and Path(x).stat().st_size==0:
            raise RuntimeError('Empty input IR; regenerate from its Torch export: '+x)
    output=None;temporary=None
    if '-o' in cmd:
        index=cmd.index('-o')+1;output=Path(cmd[index])
        temporary=output.with_name('.'+output.name+'.'+uuid.uuid4().hex+'.tmp')
        cmd[index]=str(temporary)
    start=time.perf_counter()
    with open(log,'w') as f: subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
    if output is not None:
        if not temporary.exists() or temporary.stat().st_size==0:
            raise RuntimeError('Compiler produced an empty artifact: '+str(output))
        os.replace(temporary,output)
    return time.perf_counter()-start

def load(path,x):
    cfg=rt.Config(device=rt.get_driver('local-sync').create_default_device()); vm=rt.VmModule.copy_buffer(cfg.vm_instance,Path(path).read_bytes())
    ctx=rt.SystemContext(config=cfg);ctx.add_vm_module(vm)
    args=[rt.asdevicearray(cfg.device,t.numpy()) for t in x]
    return cfg,ctx,ctx.modules.module.main,args

def allocations(text):
    constants={k:int(v) for k,v in re.findall(r'(%[\w]+) = arith.constant (\d+) : index',text)}
    # SSA constant names repeated across functions have identical values in this compiler output.
    refs=re.findall(r'hal.device.queue.alloca[^\n]*!hal.buffer\{(%[\w]+)\}',text)
    return [constants[r] for r in refs]

def bench(fn,repeats,inner):
    for _ in range(5):fn()
    values=[]
    for _ in range(repeats):
        start=time.perf_counter_ns()
        for _ in range(inner):fn()
        values.append((time.perf_counter_ns()-start)/inner/1000)
    return values

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int,default=len(CASES));ap.add_argument('--repeats',type=int,default=9);ap.add_argument('--inner',type=int,default=30)
    a=ap.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    rows=[]; detail=[]; model=GQA().eval()
    for case in CASES[:a.limit]:
        name='b%d_q%d_kv%d_m%d_n%d_d%d'%case
        work=Path('results/cases')/name;work.mkdir(parents=True,exist_ok=True)
        x=inputs(*case)
        with torch.inference_mode():
            expected=model(*x).numpy()
            # Independent grouped-head implementation, no expanded K/V.
            b,hq,hkv,m,n,d=case;r=hq//hkv
            q=x[0].reshape(b,hkv,r,m,d)
            score=torch.einsum('bhrmd,bhnd->bhrmn',q,x[1])*(d**-0.5)
            independent=torch.einsum('bhrmn,bhnd->bhrmd',score.softmax(-1),x[2]).reshape(b,hq,m,d).numpy()
            np.testing.assert_allclose(independent,expected,atol=2e-5,rtol=2e-4)
        aot.export(torch.export.export(model,x)).save_mlir(str(work/'torch.mlir'))
        run(['iree-opt',work/'torch.mlir','--torch-function-to-torch-backend-pipeline',
             '--torch-backend-to-linalg-on-tensors-backend-pipeline','--canonicalize','-o',work/'baseline.mlir'],work/'lower.log')
        pass_s=run(['build/gqa-opt',work/'baseline.mlir','--gqa-head-reuse','--canonicalize','-o',work/'optimized.mlir'],work/'pass.log')
        btext=(work/'baseline.mlir').read_text();otext=(work/'optimized.mlir').read_text()
        removed=btext.count('linalg.batch_matmul')-otext.count('linalg.batch_matmul')
        assert removed==(0 if r==1 else 2),(name,removed)
        row=dict(case=name,b=b,hq=hq,hkv=hkv,ratio=r,m=m,n=n,d=d,contractions_rewritten=removed,pass_ms=pass_s*1000)
        loaded={};checks={}
        for variant in ['baseline','optimized']:
            flags=['--iree-hal-target-backends=llvm-cpu','--iree-llvmcpu-target-cpu=host']
            row[variant+'_compile_s']=run(['iree-compile',work/(variant+'.mlir'),*flags,'-o',work/(variant+'.vmfb')],work/(variant+'-compile.log'))
            run(['iree-compile',work/(variant+'.mlir'),*flags,'--compile-to=hal','-o',work/(variant+'-hal.mlir')],work/(variant+'-hal.log'))
            sizes=allocations((work/(variant+'-hal.mlir')).read_text())
            row[variant+'_hal_alloc_sites']=len(sizes);row[variant+'_hal_alloc_bytes']=sum(sizes)
            cfg,ctx,fn,args=load(work/(variant+'.vmfb'),x)
            before=cfg.device.allocator.statistics
            y=fn(*args).to_host()
            after=cfg.device.allocator.statistics
            stats_available=cfg.device.allocator.has_statistics
            delta=y-expected
            row[variant+'_max_abs']=float(np.max(np.abs(delta)))
            row[variant+'_relative_l2']=float(np.linalg.norm(delta)/max(np.linalg.norm(expected),1e-30))
            row[variant+'_runtime_peak_bytes']=after['device_bytes_peak'] if stats_available else None
            row[variant+'_first_call_allocated_bytes']=after['device_bytes_allocated']-before['device_bytes_allocated'] if stats_available else None
            np.testing.assert_allclose(y,expected,atol=2e-5,rtol=2e-4)
            loaded[variant]=(cfg,ctx,fn,args)
            checks[variant]=after
        samples={v:[] for v in ['baseline','optimized']}
        # Alternate order each trial to limit drift; input upload and output copies excluded.
        for j in range(a.repeats):
            for v in (['baseline','optimized'] if j%2==0 else ['optimized','baseline']):
                _,_,fn,args=loaded[v]
                samples[v]+=bench(lambda:fn(*args),1,a.inner)
        with torch.inference_mode():samples['torch_eager']=bench(lambda:model(*x),a.repeats,a.inner)
        for v,s in samples.items():
            row[v+'_median_us']=statistics.median(s)
            row[v+'_min_us']=min(s);row[v+'_max_us']=max(s)
        row['speedup']=row['baseline_median_us']/row['optimized_median_us']
        rows.append(row);detail.append(dict(case=name,samples_us=samples,allocator=checks))
        Path('results/benchmark.json').write_text(json.dumps(dict(methodology=dict(driver='local-sync',torch_threads=1,repeats=a.repeats,inner=a.inner,warmup=5),rows=rows,details=detail),indent=2))
        with open('results/benchmark.csv','w') as f:
            w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerows(rows)
        print(name,'PASS','speedup=%.3f'%row['speedup'],'alloc',row['baseline_hal_alloc_bytes'],'->',row['optimized_hal_alloc_bytes'],flush=True)
        del loaded;gc.collect()
    print('Completed',len(rows),'cases',flush=True)
if __name__=='__main__':main()
