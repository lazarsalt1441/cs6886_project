"""Compare merged-query-loop and IREE fusion variants without replacing the baseline."""
import csv, json, statistics, time
from pathlib import Path
import numpy as np
import torch
from run_suite import ROOT,CASES,inputs,GQA,run,load,bench,allocations

def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 output=[]
 # Includes regression, small decode, large decode and prefill.
 for case in [CASES[2],CASES[5],CASES[9],CASES[11],CASES[12],CASES[14]]:
  name='b%d_q%d_kv%d_m%d_n%d_d%d'%case;w=Path('results/cases')/name
  x=inputs(*case)
  with torch.inference_mode():ref=GQA()(*x).numpy()
  run(['build/gqa-opt',w/'baseline.mlir','--gqa-compact-head-reuse','--canonicalize','-o',w/'compact.mlir'],w/'compact-pass.log')
  for variant,source,extra in [
    ('compact','compact',[]),
    ('aggressive-fusion','optimized',['--iree-dispatch-creation-enable-aggressive-fusion']),
    ('compact-fusion','compact',['--iree-dispatch-creation-enable-aggressive-fusion'])]:
   flags=['--iree-hal-target-backends=llvm-cpu','--iree-llvmcpu-target-cpu=host',*extra]
   row=dict(case=name,variant=variant)
   try:
    row['compile_s']=run(['iree-compile',w/(source+'.mlir'),*flags,'-o',w/(variant+'.vmfb')],w/(variant+'-compile.log'))
    run(['iree-compile',w/(source+'.mlir'),*flags,'--compile-to=hal','-o',w/(variant+'-hal.mlir')],w/(variant+'-hal.log'))
    cfg,ctx,fn,args=load(w/(variant+'.vmfb'),x)
    result=fn(*args).to_host();row['max_abs']=float(np.max(np.abs(result-ref)))
    np.testing.assert_allclose(result,ref,atol=2e-5,rtol=2e-4)
    row['hal_alloc_bytes']=sum(allocations((w/(variant+'-hal.mlir')).read_text()))
    s=bench(lambda:fn(*args),11,40);row['samples_us']=s;row['median_us']=statistics.median(s)
    row['status']='PASS'
   except Exception as e:row.update(status='FAIL',error=str(e))
   output.append(row);Path('results/extensions.json').write_text(json.dumps(output,indent=2))
   print(name,variant,row['status'],row.get('median_us'),flush=True)
if __name__=='__main__':main()
