import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import text_only,torch
from transformers import LlamaForCausalLM,LlamaConfig
from iree.turbine import aot
from iree.compiler import ir
from llm_workloads import Decode,Lookahead,MODEL_ID,REVISION
from shape_legalize import legalize
from apply_passes import apply
def main():
 p=argparse.ArgumentParser();p.add_argument('--set',choices=['A','B'],required=True);p.add_argument('--tiny',action='store_true');a=p.parse_args()
 torch.set_num_threads(1);torch.manual_seed(0);out=ROOT/'prepared'/(a.set+'_tiny' if a.tiny else a.set);out.mkdir(parents=True,exist_ok=True)
 if a.tiny:
  cfg=LlamaConfig(vocab_size=128,hidden_size=64,intermediate_size=128,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,head_dim=16);model=LlamaForCausalLM(cfg).eval();model.config._attn_implementation='eager'
 else:model=LlamaForCausalLM.from_pretrained(ROOT/'model',dtype=torch.float32,attn_implementation='eager').eval();cfg=model.config
 wrapper=(Decode if a.set=='A' else Lookahead)(model);n=8 if a.tiny else 128
 flat=tuple(torch.randn(1,cfg.num_key_value_heads,n,cfg.hidden_size//cfg.num_attention_heads) for _ in range(2*cfg.num_hidden_layers))
 mask=torch.ones(1,n+1,dtype=torch.int64);pos=torch.tensor([[n]])
 args=(torch.ones(1,1,dtype=torch.int64),mask,pos,*flat) if a.set=='A' else (torch.tensor([[1],[2]]),torch.tensor([-.3,-1.3]),mask,pos,*flat)
 shape=torch.export.Dim('past',min=2,max=4096)
 ds=(None,{1:shape+1},None,tuple({2:shape} for _ in flat)) if a.set=='A' else (None,None,{1:shape+1},None,tuple({2:shape} for _ in flat))
 with torch.inference_mode():
  aot.externalize_module_parameters(wrapper);aot.save_module_parameters(out/'weights.irpa',wrapper)
  ep=torch.export.export(wrapper,args,dynamic_shapes=ds,prefer_deferred_runtime_asserts_over_guards=True).run_decompositions();proofs=legalize(ep)
  (out/'fx.txt').write_text(str(ep.graph));aot.export(ep).save_mlir(str(out/'torch.mlir'))
 with (out/'lower.log').open('w') as log:subprocess.run(['iree-compile',str(out/'torch.mlir'),'--iree-input-type=torch','--compile-to=input','--iree-hal-target-backends=llvm-cpu','--iree-llvmcpu-target-cpu=host','-o',str(out/'input.mlir')],stdout=log,stderr=subprocess.STDOUT,check=True)
 with ir.Context():m=ir.Module.parse((out/'input.mlir').read_text());(out/'generic.mlir').write_text(m.operation.get_asm(print_generic_op_form=True))
 result=apply(out);result.update(model=MODEL_ID,revision=REVISION,set=a.set,tiny=a.tiny,dynamic=True,symbolic_min_range_proofs=proofs,attention_implementation='eager',layers=cfg.num_hidden_layers,scope='Tiny random protocol test' if a.tiny else 'Full pretrained model');result['any_rewrite']=any(v>0 for v in result['markers'].values())
 (out/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
