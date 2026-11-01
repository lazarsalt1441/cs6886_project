"""Forked functional KV-cache study. Strong, aggressive and lifted-gather baselines.
Compile first; final timing is a separate script. No hardware traffic claim.
"""
import sys,os,json,re,hashlib,argparse
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1]/'scripts')]
from probe_cache import ROOT,run,load,allocations,STRONG
from lift_gather_baseline import lift
from iree.turbine import aot
from iree.compiler import ir
import torch,numpy as np
CASES=[
 # pool,read,hkv,group,dim,query,branches,position,update
 (4096,128,2,4,32,1,2,37,2),
 (4096,128,1,8,32,1,4,0,1),
 (8192,256,2,2,64,1,8,8190,2),
 (65536,128,2,4,32,1,2,37,2),
 (1024,512,2,4,32,1,2,37,2),
 (4096,128,4,1,64,16,2,37,2),
 (1009,65,3,2,24,3,2,37,2),
 (4096,128,2,4,32,1,1,37,2), # unchanged single-version control
 (256,256,2,2,32,1,2,37,2), # unchanged dense-demand control
]
class BranchAttention(torch.nn.Module):
 def __init__(self,case,returned=False):super().__init__();self.case=case;self.returned=returned
 def forward(self,qs,k,v,nks,nvs,ids):
  pool,n,h,g,d,m,f,p,t=self.case;outs=[];caches=[]
  for b in range(f):
   kb=torch.slice_scatter(k,nks[b],dim=0,start=p,end=p+t)
   vb=torch.slice_scatter(v,nvs[b],dim=0,start=p,end=p+t)
   kk=kb.index_select(0,ids).permute(1,0,2).unsqueeze(0)
   vv=vb.index_select(0,ids).permute(1,0,2).unsqueeze(0)
   outs.append(torch.nn.functional.scaled_dot_product_attention(qs[b],kk,vv,enable_gqa=True));caches.extend([kb,vb])
  return tuple(outs+(caches[:2] if self.returned=='one' else caches)) if self.returned else tuple(outs)
def name(case):return 'p%d_n%d_h%d_g%d_d%d_m%d_f%d_o%d_t%d'%tuple(case)
def inputs(case,seed=171):
 pool,n,h,g,d,m,f,p,t=case;gen=torch.Generator().manual_seed(seed)
 x=[torch.randn(f,1,h*g,m,d,generator=gen),torch.randn(pool,h,d,generator=gen),torch.randn(pool,h,d,generator=gen),torch.randn(f,t,h,d,generator=gen),torch.randn(f,t,h,d,generator=gen)]
 ids=torch.randint(pool,(n,),generator=gen);ids[:min(t,n)]=torch.arange(p,p+min(t,n));ids[-1]=pool-1
 return (*x,ids)
def reference(model,x):
 with torch.inference_mode():return [y.numpy() for y in model(*[a.double() if a.is_floating_point() else a for a in x])]
def forward(source,dest,log,force=False):
 with ir.Context():
  mod=ir.Module.parse(Path(source).read_text())
  if force:
   for op in mod.body.operations:
    if op.operation.name=='func.func':op.attributes['attention.force_cache_forwarding']=ir.UnitAttr.get()
  generic=Path(dest).with_name(Path(dest).stem+'-input.mlir');generic.write_text(mod.operation.get_asm(print_generic_op_form=True))
 return run(['build/gqa-opt',generic,'--allow-unregistered-dialect','--attention-forward-cache-reads','--canonicalize','-o',dest],log)
def normalize(source,dest,log):
 with ir.Context():
  mod=ir.Module.parse(Path(source).read_text());generic=Path(dest).with_name(Path(dest).stem+'-input.mlir');generic.write_text(mod.operation.get_asm(print_generic_op_form=True))
 return run(['build/gqa-opt',generic,'--allow-unregistered-dialect','--attention-canonicalize-gather-indices','--canonicalize','-o',dest],log)
def copy_stats(hal):
 const={k:int(v) for k,v in re.findall(r'(%[\w]+) = arith.constant (\d+) : index',hal)}
 sizes=[]
 for line in hal.splitlines():
  if 'hal.command_buffer.copy_buffer' not in line:continue
  m=re.search(r'length\((%[\w]+)\)',line)
  if m and m[1] in const:sizes.append(const[m[1]])
 return sizes

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int);args=ap.parse_args();torch.set_num_threads(1);rows=[]
 for case in CASES[:args.limit]:
  w=Path('research2/results/study')/name(case);w.mkdir(parents=True,exist_ok=True);x=inputs(case);model=BranchAttention(case);refs=reference(model,x)
  aot.export(torch.export.export(model,x)).save_mlir(str(w/'torch.mlir'))
  run(['iree-opt',w/'torch.mlir','--torch-function-to-torch-backend-pipeline','--torch-backend-to-linalg-on-tensors-backend-pipeline','--canonicalize','-o',w/'baseline.mlir'],w/'lower.log')
  ps=forward(w/'baseline.mlir',w/'forwarded.mlir',w/'forward-pass.log');count=(w/'forwarded.mlir').read_text().count('attention.cache_read_forwarded')
  lc=lift(w/'baseline.mlir',w/'lifted.mlir')
  normalize(w/'lifted.mlir',w/'lifted-normalized.mlir',w/'normalize.log')
  row={'case':name(case),'shape':case,'forwarded_reads':count,'lifted_gathers':lc,'pass_s':ps,'variants':{}}
  variants=[('strong','baseline',STRONG),('aggressive','baseline',STRONG+['--iree-dispatch-creation-enable-aggressive-fusion']),('lifted-normalized','lifted-normalized',STRONG+['--iree-dispatch-creation-enable-aggressive-fusion']),('forwarded','forwarded',STRONG)]
  for variant,source,extra in variants:
   flags=['--iree-hal-target-backends=llvm-cpu','--iree-llvmcpu-target-cpu=host',*extra]
   try:
    cs=run(['iree-compile',w/(source+'.mlir'),*flags,'-o',w/(variant+'.vmfb')],w/(variant+'-compile.log'))
    run(['iree-compile',w/(source+'.mlir'),*flags,'--compile-to=hal','-o',w/(variant+'-hal.mlir')],w/(variant+'-hal.log'))
    cfg,ctx,fn,dev=load(w/(variant+'.vmfb'),[t.clone() for t in x]);before=cfg.device.allocator.statistics;y=fn(*dev);after=cfg.device.allocator.statistics
    ys=[y] if hasattr(y,'to_host') else list(y);assert len(ys)==len(refs)
    errors=[]
    for actual,ref in zip(ys,refs):arr=actual.to_host();np.testing.assert_allclose(arr,ref,atol=4e-5,rtol=1e-3);errors.append(float(np.max(np.abs(arr-ref))))
    text=(w/(variant+'-hal.mlir')).read_text()
    row['variants'][variant]={'compiled':True,'correct':True,'compile_s':cs,'max_abs':max(errors),'allocated_bytes':after['device_bytes_allocated']-before['device_bytes_allocated'],'hal_alloc_bytes':sum(allocations(text)),'dispatches':text.count('hal.command_buffer.dispatch'),'copy_command_bytes':copy_stats(text),'vmfb_sha256':hashlib.sha256((w/(variant+'.vmfb')).read_bytes()).hexdigest()}
   except Exception as e:row['variants'][variant]={'error':str(e)};print(name(case),variant,'ERROR',str(e)[:150],flush=True)
  assert row['variants']['forwarded'].get('correct'),row
  rows.append(row);Path('research2/results/cache-study.json').write_text(json.dumps({'rows':rows,'compiler':'89ece87b89852100c0a4ac3d515e14842d090967','source_matches_compiler':True},indent=2));print(name(case),'reads',count,{v:r.get('allocated_bytes',r.get('error')) for v,r in row['variants'].items()},flush=True)
 print('Completed',len(rows),flush=True)
if __name__=='__main__':main()
