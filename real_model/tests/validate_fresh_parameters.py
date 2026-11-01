"""Exercise Colab's missing-IRPA path: save parameters without re-export."""
import sys,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from benchmark import load_model,real_inputs,load_iree,check_outputs,write_json
from llm_workloads import Decode,Lookahead
from iree.turbine import aot
from transformers import AutoTokenizer
torch.set_num_threads(1);m=load_model();tok=AutoTokenizer.from_pretrained(ROOT/'model',local_files_only=True);rows=[]
for which in 'AB':
 wrapper=(Decode if which=='A' else Lookahead)(m);fresh=ROOT/'local_checks'/('fresh_parameters_'+which+'.irpa');aot.save_module_parameters(fresh,wrapper)
 args,_=real_inputs(m,tok,8,0,which)
 with torch.inference_mode():gold=wrapper(*args)
 for v in (['baseline'] if which=='A' else ['baseline','optimized']):
  r=load_iree(ROOT/'local_checks'/('final_'+which)/('iree_'+v)/'module.vmfb',fresh,'local-sync');out=r['fn'](*(z.numpy() for z in args));rows.append(dict(set=which,variant=v,checks=check_outputs(gold,out,which+v)))
write_json(ROOT/'local_checks/fresh_parameters.json',rows);print('PASS fresh Colab-style parameter archives, all three compiled model variants')
