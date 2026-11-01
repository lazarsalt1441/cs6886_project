import sys,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from benchmark import load_model,load_iree,real_inputs,check_outputs,write_json
from llm_workloads import Decode,Lookahead
from transformers import AutoTokenizer
def main():
 torch.set_num_threads(1);m=load_model();tok=AutoTokenizer.from_pretrained(ROOT/'model',local_files_only=True);rows=[]
 for which in 'AB':
  wrapper=(Decode if which=='A' else Lookahead)(m);folder=ROOT/'local_checks'/('final_'+which)
  runtimes={v:load_iree(folder/('iree_'+v)/'module.vmfb',ROOT/'prepared'/which/'weights.irpa','local-sync') for v in (['baseline'] if which=='A' else ['baseline','optimized'])}
  for n in [32,128,512]:
   for prompt in [0,1]:
    args,_=real_inputs(m,tok,n,prompt,which);choices=[('natural',args)]
    if which=='B':choices.extend((f'forced_{j}',(args[0],torch.tensor([0.,-100.]) if j==0 else torch.tensor([-100.,0.]),*args[2:])) for j in [0,1])
    for case,actual in choices:
     with torch.inference_mode():gold=wrapper(*actual)
     for v,r in runtimes.items():rows.append(dict(set=which,past=n,prompt=prompt,case=case,variant=v,checks=check_outputs(gold,r['fn'](*(z.numpy() for z in actual)),which+v)))
     write_json(ROOT/'local_checks/pretrained_extended.json',rows)
 print('PASS',len(rows),'pretrained cases; all logits/caches/winners')
if __name__=='__main__':main()
