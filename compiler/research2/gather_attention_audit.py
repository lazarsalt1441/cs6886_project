"""Independent correctness audit of explicit-index-component gather fusion."""
import sys,json,argparse,subprocess,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from study_cache import *
class ReadAttention(torch.nn.Module):
 def forward(self,q,k,v,ids):
  b,n=ids.shape;h=k.shape[1];d=k.shape[2]
  kk=k.index_select(0,ids.flatten()).reshape(b,n,h,d).permute(0,2,1,3)
  vv=v.index_select(0,ids.flatten()).reshape(b,n,h,d).permute(0,2,1,3)
  return torch.nn.functional.scaled_dot_product_attention(q,kk,vv,enable_gqa=True)
CASES=[(1,2,1,2,8),(1,2,1,8,32),(1,2,1,17,8),(1,2,4,128,32),(1,3,2,65,24),(1,1,8,128,32),(1,2,1,257,48),(2,2,4,65,24),(2,2,1,128,32)]
def make_case(i):
 case=CASES[i];b,h,g,n,d=case;pool=max(257,2*n+1);gen=torch.Generator().manual_seed(455+i);dtype=torch.float16 if i==6 else torch.float32;idtype=torch.int32 if i in [4,7] else torch.int64
 q=torch.zeros(b,h*g,1,d,dtype=dtype);k=torch.zeros(pool,h,d,dtype=dtype)
 v=(torch.arange(pool*h,dtype=torch.float32).reshape(pool,h,1).expand(pool,h,d)/(pool*h)).contiguous().to(dtype);ids=torch.randint(pool,(b,n),generator=gen,dtype=idtype)
 return case,(q,k,v,ids),dtype,idtype

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--worker-index',type=int);ap.add_argument('--worker-variant');ap.add_argument('--compiler',default='iree-compile');ap.add_argument('--tag');args=ap.parse_args();
 if args.compiler!='iree-compile':assert args.tag,'Use --tag to preserve wheel audit'
 audit_root=Path('research2/results/gather-audit'+('-'+args.tag if args.tag else ''));result_path=Path('research2/results/gather-attention-audit'+('-'+args.tag if args.tag else '')+'.json');torch.set_num_threads(1)
 if args.worker_index is not None:
  i=args.worker_index;case,x,dtype,idtype=make_case(i);w=audit_root/('b%d_h%d_g%d_n%d_d%d'%case)
  with torch.inference_mode():ref=ReadAttention()(*[a.double() if a.is_floating_point() else a for a in x]).numpy()
  cfg,ctx,fn,dev=load(w/(args.worker_variant+'.vmfb'),x);y=fn(*dev).to_host();finite=bool(np.isfinite(y).all());err=float(np.max(np.abs(y-ref))) if finite else None
  print(json.dumps({'correct':bool(np.allclose(y,ref,atol=1e-3 if dtype==torch.float16 else 2e-5,rtol=2e-3 if dtype==torch.float16 else 1e-4)),'finite':finite,'max_abs':err}),flush=True);return
 rows=[]
 for i,case in enumerate(CASES):
  case,x,dtype,idtype=make_case(i);model=ReadAttention();w=audit_root/('b%d_h%d_g%d_n%d_d%d'%case);w.mkdir(parents=True,exist_ok=True)
  aot.export(torch.export.export(model,x)).save_mlir(str(w/'torch.mlir'));run(['iree-opt',w/'torch.mlir','--torch-function-to-torch-backend-pipeline','--torch-backend-to-linalg-on-tensors-backend-pipeline','--canonicalize','-o',w/'baseline.mlir'],w/'lower.log');lc=lift(w/'baseline.mlir',w/'lifted.mlir');assert lc==2,(case,lc)
  normalize(w/'lifted.mlir',w/'normalized.mlir',w/'normalize.log');row={'case':case,'dtype':str(dtype),'index_dtype':str(idtype),'variants':{}}
  for variant,source in [('original','baseline'),('raw-gather','lifted'),('normalized','normalized')]:
   try:
    source_file=w/(source+'.mlir')
    if args.compiler!='iree-compile':
     source_file=w/(source+'-native.mlir');run(['iree-opt',w/(source+'.mlir'),'--torch-iree-tm-tensor-to-linalg-ext','-o',source_file],w/(source+'-native.log'))
    run([args.compiler,source_file,'--iree-hal-target-backends=llvm-cpu','--iree-llvmcpu-target-cpu=host',*STRONG,'-o',w/(variant+'.vmfb')],w/(variant+'-compile.log'))
    # Bad compiled indices can crash the process. Isolate each native invocation.
    child=subprocess.run([sys.executable,__file__,'--worker-index',str(i),'--worker-variant',variant,*(['--tag',args.tag] if args.tag else [])],capture_output=True,text=True,timeout=60)
    (w/(variant+'-runtime.log')).write_text(child.stdout+child.stderr)
    row['variants'][variant]=json.loads(child.stdout.strip()) if child.returncode==0 else {'correct':False,'worker_returncode':child.returncode,'crashed':child.returncode<0}
    row['variants'][variant]['vmfb_sha256']=hashlib.sha256((w/(variant+'.vmfb')).read_bytes()).hexdigest()
   except Exception as e:row['variants'][variant]={'error':str(e),'correct':False}
  rows.append(row);result_path.write_text(json.dumps(rows,indent=2));print(case,{v:d.get('correct') for v,d in row['variants'].items()},flush=True);assert row['variants']['normalized'].get('correct'),row
 print('Completed',len(rows),flush=True)
if __name__=='__main__':main()
