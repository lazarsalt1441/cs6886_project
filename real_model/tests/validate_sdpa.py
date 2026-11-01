import copy,sys,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from benchmark import load_model,real_inputs,check_outputs,write_json
from llm_workloads import Decode,Lookahead
from transformers import AutoTokenizer
torch.set_num_threads(1);m=load_model();sdpa=copy.deepcopy(m);sdpa.config._attn_implementation='sdpa';tok=AutoTokenizer.from_pretrained(ROOT/'model',local_files_only=True);rows=[]
for which in 'AB':
 a=(Decode if which=='A' else Lookahead)(m);b=(Decode if which=='A' else Lookahead)(sdpa)
 for n in [8,128]:
  args,_=real_inputs(m,tok,n,0,which)
  with torch.inference_mode():checks=check_outputs(a(*args),b(*args),'sdpa')
  rows.append(dict(set=which,past=n,checks=checks))
write_json(ROOT/'local_checks/sdpa_cpu.json',rows);print('PASS natural Torch SDPA vs eager-attention outputs, both full pretrained workloads')
