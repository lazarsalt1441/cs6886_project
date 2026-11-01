from pathlib import Path
import argparse,subprocess,os,json,time,random
import numpy as np
import iree.runtime as rt
ap=argparse.ArgumentParser();ap.add_argument('--project',required=True);ap.add_argument('--compiler',required=True);ap.add_argument('--label',required=True);a=ap.parse_args();P=Path(__file__).resolve().parents[1];proj=Path(a.project).resolve();out=P/'results'/a.label/'snapshots';out.mkdir(exist_ok=True)
env=os.environ.copy();env['LD_LIBRARY_PATH']=str(proj/'compiler/toolchain/usr/lib/llvm-19/lib')+':'+str(proj/'compiler/toolchain/usr/lib/x86_64-linux-gnu');tool=proj/'compiler/build/gqa-opt'
s=(P/'reproducers/winner_generic_stock.mlir').read_text().replace('-> tensor<4096xf32> {','-> (tensor<4096xf32>,tensor<4096xf32>) {',1).replace('return %r : tensor<4096xf32>','return %r,%base : tensor<4096xf32>,tensor<4096xf32>')
src=out/'input.mlir';src.write_text(s);rows=[];loaded={}
try:os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
except OSError:pass
for variant,flags in [('stock_unguarded',[]),('stock_guarded',['--attention-protect-cache-snapshots']),('custom_guarded',['--attention-factor-cache-commit','--attention-protect-cache-snapshots'])]:
 dest=out/(variant+'.mlir');cmd=[str(tool),str(src),'--allow-unregistered-dialect',*flags,'--canonicalize','-o',str(dest)];p=subprocess.run(cmd,env=env,capture_output=True,text=True);p.check_returncode();cmd2=[a.compiler,str(dest),'--iree-hal-target-backends=llvm-cpu','--iree-llvmcpu-target-cpu=host','--iree-opt-level=O3','--iree-opt-generalize-matmul','--iree-dispatch-creation-enable-aggressive-reshape-movement','-o',str(dest.with_suffix('.vmfb'))];p=subprocess.run(cmd2,capture_output=True,text=True);(out/(variant+'.log')).write_text(p.stderr);p.check_returncode()
 cfg=rt.Config('local-sync');ctx=rt.SystemContext(config=cfg);ctx.add_vm_module(rt.VmModule.mmap(cfg.vm_instance,str(dest.with_suffix('.vmfb'))));fn=ctx.modules.module.main;checks=[]
 for seed in range(10):
  rng=np.random.default_rng(seed);base=rng.normal(size=4096).astype('float32');left=rng.normal(size=16).astype('float32');right=rng.normal(size=16).astype('float32');ref=base.copy();ref[128:144]=left if seed%2 else right;dev=[rt.asdevicearray(cfg.device,v.copy()) for v in [base,left,right,np.array(bool(seed%2))]];ret=fn(*dev);vals=[x.to_host().copy() for x in ret];checks.append({'seed':seed,'output_correct':bool(np.array_equal(vals[0],ref)),'snapshot_correct':bool(np.array_equal(vals[1],base)),'input_preserved':bool(np.array_equal(dev[0].to_host(),base))})
 row=dict(variant=variant,commands=[cmd,cmd2],checks=checks);rows.append(row);loaded[variant]=(cfg,ctx,fn,dev,row)
# Time only fully correct programs. Equal output and immutability contracts.
valid={k:v for k,v in loaded.items() if all(c['output_correct'] and c['snapshot_correct'] and c['input_preserved'] for c in v[-1]['checks'])};rr=random.Random(872);ts={k:[] for k in valid}
for v in valid.values():
 for _ in range(20):v[2](*v[3])
for _ in range(201):
 order=list(valid);rr.shuffle(order)
 for key in order:
  v=valid[key];st=time.perf_counter_ns();r=v[2](*v[3]);ts[key].append((time.perf_counter_ns()-st)/1000)
for row in rows:
 if row['variant'] in ts:
  x=ts[row['variant']];row['timing']={'median_us':float(np.median(x)),'p10_us':float(np.percentile(x,10)),'p90_us':float(np.percentile(x,90)),'samples_us':x}
(out/'summary.json').write_text(json.dumps(rows,indent=2));print([(r['variant'],all(c['snapshot_correct'] for c in r['checks']),r.get('timing',{}).get('median_us')) for r in rows])
