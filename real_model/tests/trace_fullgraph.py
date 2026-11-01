import sys,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from benchmark import load_model,real_inputs,check_outputs,write_json
from llm_workloads import Decode,Lookahead
from transformers import AutoTokenizer
torch.set_num_threads(1);m=load_model();tok=AutoTokenizer.from_pretrained(ROOT/'model',local_files_only=True);rows=[]
for which in 'AB':
 w=(Decode if which=='A' else Lookahead)(m);args,_=real_inputs(m,tok,8,0,which)
 with torch.inference_mode():expected=w(*args);actual=torch.compile(w,backend='eager',fullgraph=True,dynamic=False)(*args)
 rows.append(dict(set=which,backend='eager',fullgraph=True,checks=check_outputs(expected,actual,which)))
write_json(ROOT/'local_checks/fullgraph_capture.json',rows);print('PASS fullgraph Dynamo capture, both full pretrained workloads; not Inductor GPU validation')
