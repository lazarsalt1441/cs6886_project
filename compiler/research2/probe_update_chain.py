"""Probe per-token functional KV update chains before extending the optimizer."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from study_cache import *
from add_patch_variant import patch
class ChainAttention(torch.nn.Module):
 def __init__(self,tokens,pool=4096):super().__init__();self.tokens=tokens;self.pool=pool
 def forward(self,qs,k,v,nks,nvs,ids):
  outs=[]
  for b in range(2):
   kb,vb=k,v
   for t in range(self.tokens):
    kb=torch.slice_scatter(kb,nks[b,t:t+1],dim=0,start=37+t,end=38+t)
    vb=torch.slice_scatter(vb,nvs[b,t:t+1],dim=0,start=37+t,end=38+t)
   kk=kb.index_select(0,ids).permute(1,0,2).unsqueeze(0);vv=vb.index_select(0,ids).permute(1,0,2).unsqueeze(0);outs.append(torch.nn.functional.scaled_dot_product_attention(qs[b],kk,vv,enable_gqa=True))
  return tuple(outs)
def chain_inputs(tokens,seed=171,pool=4096):return inputs((pool,128,2,4,32,1,2,37,tokens),seed)
def main():
 torch.set_num_threads(1);rows=[]
 for t in [1,4,8,16]:
  w=Path('research2/results/update-chain')/f't{t}';w.mkdir(parents=True,exist_ok=True);x=chain_inputs(t);model=ChainAttention(t);refs=reference(model,x)
  aot.export(torch.export.export(model,x)).save_mlir(str(w/'torch.mlir'));run(['iree-opt',w/'torch.mlir','--torch-function-to-torch-backend-pipeline','--torch-backend-to-linalg-on-tensors-backend-pipeline','--canonicalize','-o',w/'baseline.mlir'],w/'lower.log');forward(w/'baseline.mlir',w/'forwarded.mlir',w/'forward.log');lift(w/'baseline.mlir',w/'lifted.mlir');normalize(w/'lifted.mlir',w/'normalized.mlir',w/'normalize.log');patch(w/'normalized.mlir',w/'masked.mlir',w/'masked.log',ratio=4)
  row={'tokens':t,'variants':{}}
  for variant,stem in [('strong','baseline'),('aggressive','baseline'),('fused','normalized'),('forwarded','forwarded'),('masked','masked')]:
   flags=['--iree-hal-target-backends=llvm-cpu','--iree-llvmcpu-target-cpu=host',*STRONG]+(['--iree-dispatch-creation-enable-aggressive-fusion'] if variant in ['aggressive','fused'] else []);run(['iree-compile',w/(stem+'.mlir'),*flags,'-o',w/(variant+'.vmfb')],w/(variant+'-compile.log'));run(['iree-compile',w/(stem+'.mlir'),*flags,'--compile-to=hal','-o',w/(variant+'-hal.mlir')],w/(variant+'-hal.log'));cfg,ctx,fn,args=load(w/(variant+'.vmfb'),[a.clone() for a in x]);before=cfg.device.allocator.statistics;ys=fn(*args);after=cfg.device.allocator.statistics
   assert len(ys)==len(refs)
   for a,b in zip(ys,refs):np.testing.assert_allclose(a.to_host(),b,atol=4e-5,rtol=1e-3)
   hal=(w/(variant+'-hal.mlir')).read_text();row['variants'][variant]={'correct':True,'allocated_bytes':after['device_bytes_allocated']-before['device_bytes_allocated'],'copies':copy_stats(hal),'read_forwards':(w/(stem+'.mlir')).read_text().count('cache_read_forwarded'),'masked_forwards':(w/(stem+'.mlir')).read_text().count('cache_gather_forwarded'),'vmfb_sha256':hashlib.sha256((w/(variant+'.vmfb')).read_bytes()).hexdigest()}
  rows.append(row);Path('research2/results/update-chain-probe.json').write_text(json.dumps(rows,indent=2));print(t,{v:(r['allocated_bytes'],r['read_forwards'],r['masked_forwards']) for v,r in row['variants'].items()},flush=True)
if __name__=='__main__':main()
