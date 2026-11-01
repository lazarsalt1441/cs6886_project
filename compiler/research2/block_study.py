"""A full synthetic transformer block with divergent cache versions.
Includes RMSNorm, Q/K/V projections, RoPE, attention, output projection and MLP.
This is a complete block microbenchmark, not pretrained end-to-end generation.
"""
import sys,copy,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from study_cache import *
from add_patch_variant import patch
class BranchBlock(torch.nn.Module):
 def __init__(self,f,pool,p=37,h=2,g=2,d=32):
  super().__init__();self.f=f;self.pool=pool;self.p=p;self.h=h;self.g=g;self.d=d
 def rms(self,x):return x*torch.rsqrt(x.square().mean(dim=-1,keepdim=True)+1e-5)
 def rope(self,x,cos,sin):return x*cos+torch.cat([-x[...,self.d//2:],x[...,:self.d//2]],dim=-1)*sin
 def forward(self,x,kcache,vcache,ids,wq,wk,wv,wo,w1,w2,cos,sin):
  linear=torch.nn.functional.linear
  norm=self.rms(x);qq=self.rope(linear(norm,wq).reshape(self.f,self.h*self.g,self.d),cos,sin);kk=self.rope(linear(norm,wk).reshape(self.f,self.h,self.d),cos,sin);vv=linear(norm,wv).reshape(self.f,self.h,self.d);outputs=[]
  for b in range(self.f):
   k=torch.slice_scatter(kcache,kk[b:b+1],dim=0,start=self.p,end=self.p+1);v=torch.slice_scatter(vcache,vv[b:b+1],dim=0,start=self.p,end=self.p+1)
   k=k.index_select(0,ids).permute(1,0,2).unsqueeze(0);v=v.index_select(0,ids).permute(1,0,2).unsqueeze(0)
   a=torch.nn.functional.scaled_dot_product_attention(qq[b].reshape(1,self.h*self.g,1,self.d),k,v,enable_gqa=True).reshape(-1)
   y=x[b]+linear(a,wo);y=y+linear(torch.nn.functional.gelu(linear(self.rms(y),w1),approximate='tanh'),w2);outputs.append(y)
  return torch.stack(outputs)
def block_inputs(f,pool,seed=515):
 torch.manual_seed(seed);ids=torch.randint(pool,(128,));ids[:3]=torch.tensor([37,0,pool-1]);x=(torch.randn(f,128),torch.randn(pool,2,32),torch.randn(pool,2,32),ids)
 weights=[torch.randn(o,i)*(i**-.5) for o,i in [(128,128),(64,128),(64,128),(128,128),(256,128),(128,256)]]
 theta=37/(10000**(torch.arange(16)/16.0));return (*x,*weights,torch.cat([theta.cos(),theta.cos()]),torch.cat([theta.sin(),theta.sin()]))
def main():
 torch.set_num_threads(1);rows=[]
 for f,pool in [(2,4096),(4,16384)]:
  torch.manual_seed(15);model=BranchBlock(f,pool).eval();x=block_inputs(f,pool);w=Path('research2/results/blocks')/f'f{f}-p{pool}';w.mkdir(parents=True,exist_ok=True)
  with torch.inference_mode():ref=copy.deepcopy(model).double()(*[a.double() if a.is_floating_point() else a for a in x]).numpy()
  aot.export(torch.export.export(model,x)).save_mlir(str(w/'torch.mlir'))
  run(['iree-opt',w/'torch.mlir','--torch-function-to-torch-backend-pipeline','--torch-backend-to-linalg-on-tensors-backend-pipeline','--canonicalize','-o',w/'baseline.mlir'],w/'lower.log')
  lc=lift(w/'baseline.mlir',w/'lifted.mlir');normalize(w/'lifted.mlir',w/'lifted-normalized.mlir',w/'normalize.log');forward(w/'baseline.mlir',w/'forwarded.mlir',w/'forward.log');patch(w/'lifted-normalized.mlir',w/'patch-forwarded.mlir',w/'patch.log')
  row={'case':w.name,'branches':f,'pool':pool,'seed':15,'lifted_gathers':lc,'variants':{}}
  for v,source in [('strong','baseline'),('lifted-normalized','lifted-normalized'),('forwarded','forwarded'),('patch-forwarded','patch-forwarded')]:
   flags=['--iree-hal-target-backends=llvm-cpu','--iree-llvmcpu-target-cpu=host',*STRONG]
   cs=run(['iree-compile',w/(source+'.mlir'),*flags,'-o',w/(v+'.vmfb')],w/(v+'-compile.log'));run(['iree-compile',w/(source+'.mlir'),*flags,'--compile-to=hal','-o',w/(v+'-hal.mlir')],w/(v+'-hal.log'))
   cfg,ctx,fn,args=load(w/(v+'.vmfb'),[a.clone() for a in x]);before=cfg.device.allocator.statistics;y=fn(*args).to_host();after=cfg.device.allocator.statistics;np.testing.assert_allclose(y,ref,atol=1e-4,rtol=2e-3)
   text=(w/(v+'-hal.mlir')).read_text();row['variants'][v]={'compiled':True,'correct':True,'compile_s':cs,'max_abs':float(np.max(np.abs(y-ref))),'allocated_bytes':after['device_bytes_allocated']-before['device_bytes_allocated'],'dispatches':text.count('hal.command_buffer.dispatch'),'copy_command_bytes':copy_stats(text),'vmfb_sha256':hashlib.sha256((w/(v+'.vmfb')).read_bytes()).hexdigest()}
  rows.append(row);Path('research2/results/block-study.json').write_text(json.dumps(rows,indent=2));print(row['case'],row['lifted_gathers'],{v:(r['max_abs'],r['allocated_bytes']) for v,r in row['variants'].items()},flush=True)
if __name__=='__main__':main()
