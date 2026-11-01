"""Thin tensor-ABI adapters around unchanged Transformers model forwards."""
import torch
from transformers import DynamicCache
MODEL_ID='HuggingFaceTB/SmolLM2-135M-Instruct'
REVISION='12fd25f77366fa6b3b4b768ec3050bf629380bac'
def flatten_cache(cache):return tuple(z for pair in cache.to_legacy_cache() for z in pair)
class Decode(torch.nn.Module):
 def __init__(self,model):super().__init__();self.model=model
 def forward(self,tokens,mask,position,*flat):
  cache=DynamicCache.from_legacy_cache(tuple(zip(flat[::2],flat[1::2])))
  out=self.model(input_ids=tokens,attention_mask=mask,position_ids=position,past_key_values=cache,use_cache=True,logits_to_keep=1)
  return out.logits,*flatten_cache(out.past_key_values)
class Lookahead(torch.nn.Module):
 """Custom 2-candidate one-step lookahead; not HF beam/speculative decoding."""
 def __init__(self,model):super().__init__();self.decode=Decode(model)
 def forward(self,tokens,priors,mask,position,*flat):
  a=self.decode(tokens[0:1],mask,position,*flat);b=self.decode(tokens[1:2],mask,position,*flat)
  choose_a=priors[0]+a[0].log_softmax(-1).amax()>=priors[1]+b[0].log_softmax(-1).amax()
  chosen=tuple(torch.where(choose_a,x,y) for x,y in zip(a,b))
  index=torch.where(choose_a,torch.zeros((),dtype=torch.int64,device=tokens.device),torch.ones((),dtype=torch.int64,device=tokens.device))
  return chosen[0],index,*chosen[1:]
