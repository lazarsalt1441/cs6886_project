import sys,json,time,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import benchmark as b
from llm_workloads import Decode
from transformers import AutoTokenizer
import numpy as np
r=Path(__file__).resolve().parent;torch.set_num_threads(1);model=b.load_model();tok=AutoTokenizer.from_pretrained(ROOT/'model',local_files_only=True);wrapper=Decode(model)
fs={label:b.load_iree(path,ROOT/'prepared/A/weights.irpa','local-sync') for label,path in [('base',ROOT/'local_checks/final_A/iree_baseline/module.vmfb'),('stable',r/'prepared/A_stable_cpu.vmfb')]};rows=[]
for n,p in [(2048,1),(3072,1),(4096,0),(4096,1)]:
 args,_=b.real_inputs(model,tok,n,p,'A')
 with torch.inference_mode():gold=tuple(z.detach().numpy() for z in wrapper(*args))
 row=dict(context=n,prompt=p,variants={})
 for label,rt in fs.items():
  out=tuple(z.to_host() for z in rt['fn'](*(z.numpy() for z in args)));bad=[]
  for i,(x,y) in enumerate(zip(gold,out)):
   delta=abs(x.astype(float)-y.astype(float));mask=delta>5e-4+3e-3*abs(x.astype(float));bad.append(dict(output=i,max_abs=float(delta.max()),failures=int(mask.sum())))
  row['variants'][label]=dict(pass_=not any(k['failures'] for k in bad),max_abs=max(k['max_abs'] for k in bad),outputs=bad)
 rows.append(row);b.write_json(r/'model_cpu_results.json',rows);print(n,p,{k:{kk:vv for kk,vv in v.items() if kk!='outputs'} for k,v in row['variants'].items()},flush=True)
