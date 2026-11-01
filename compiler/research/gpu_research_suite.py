"""Tomorrow's reproducible GPU experiment. Compile-only mode is CPU runnable.
No GPU measurements are claimed by --compile-only. Numerical failures are
recorded and excluded from latency speedups, rather than silently tolerated.
"""
import os,sys,json,argparse,time,statistics,random
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from projection_study import ROOT,ProjectedAttention,create_inputs,STRONG
from rotary_projection_study import RotaryProjected,rotate
from run_suite import run
import torch,numpy as np
import iree.runtime as rt

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--arch',default='sm_80');ap.add_argument('--compile-only',action='store_true');ap.add_argument('--trials',type=int,default=21);ap.add_argument('--device',default='cuda');ap.add_argument('--full',action='store_true');ap.add_argument('--native-only',action='store_true');ap.add_argument('--case',help='Exact case name filter');ap.add_argument('--torch-cuda',action='store_true',help='Also benchmark eager PyTorch if this environment has CUDA-enabled torch');args=ap.parse_args()
 torch.set_num_threads(1)
 cases=[('decomposed','h8_m1_n512_l128_d64_v64_s1_mask0',(8,1,512,128,64,64,True,False)),('decomposed','h8_m1_n2048_l512_d128_v128_s1_mask0',(8,1,2048,512,128,128,True,False)),('rotary','h16_m1_n1024_l512_d128_v128_r64',(16,1,1024,512,128,128,64))]
 if args.full:
  cases=[]
  for family,file in [('decomposed','projection-study.json'),('native','native-projection-study.json'),('rotary','rotary-projection-study.json')]:
   for row in json.loads((Path('research/results')/file).read_text())['rows']:cases.append((family,row['case'],row['shape']))
 if args.native_only:cases=[c for c in cases if c[0]=='native']
 if args.case:cases=[c for c in cases if c[1]==args.case]
 if not cases:raise ValueError('No matching cases; use --full with --native-only or --case')
 output=[]
 for kind,name,case in cases:
  group='projections' if kind=='decomposed' else 'native-projections' if kind=='native' else 'rotary-projections';source=Path('research/results')/group/name
  w=Path('research/results/gpu')/args.arch/(kind+'-'+name);w.mkdir(parents=True,exist_ok=True)
  if kind in ['decomposed','native']:
   x=create_inputs(case)
   if kind=='native':
    from native_projection_study import NativeProjected
    model=NativeProjected(case[6])
   else:model=ProjectedAttention(case[6],case[7])
  else:
   h,m,n,l,d,dv,r=case;x=create_inputs((h,m,n,l,d,dv,True,False));x=(*x,rotate(torch.randn(h,m,r),torch.arange(n-m,n)),rotate(torch.randn(1,n,r),torch.arange(n)));model=RotaryProjected()
  with torch.inference_mode():reference=model(*[a.double() for a in x]).numpy()
  row=dict(case=name,kind=kind,arch=args.arch,compile_only=args.compile_only,variants={});loaded={}
  sources=[('baseline',source/'baseline.mlir'),('optimized',source/'optimized.mlir')]
  if kind=='native':
   fallback=Path('research/results/projections')/(name+'_mask0')/'optimized.mlir'
   if fallback.exists():sources.append(('decomposed-fallback',fallback))
  for variant,source_file in sources:
   try:
    row['variants'][variant]={'compile_s':run(['iree-compile',source_file,'--iree-hal-target-backends=cuda','--iree-cuda-target='+args.arch,*STRONG,'--iree-hal-dump-executable-files-to='+str(w/(variant+'-codegen')),'-o',w/(variant+'.vmfb')],w/(variant+'-compile.log'))}
    row['variants'][variant]['compiled']=True
    if args.compile_only:continue
    cfg=rt.Config(args.device);vm=rt.VmModule.copy_buffer(cfg.vm_instance,(w/(variant+'.vmfb')).read_bytes());ctx=rt.SystemContext(config=cfg);ctx.add_vm_module(vm)
    dev=[rt.asdevicearray(cfg.device,a.numpy()) for a in x];fn=ctx.modules.module.main
    result=fn(*dev).to_host();difference=result-reference
    correct=bool(np.allclose(result,reference,atol=5e-5,rtol=2e-3))
    row['variants'][variant].update(correct=correct,max_abs=float(np.max(np.abs(difference))),relative_l2=float(np.linalg.norm(difference)/np.linalg.norm(reference)))
    if correct:loaded[variant]=(cfg,ctx,fn,dev)
   except Exception as e:row['variants'].setdefault(variant,{}).update(error=str(e))
  if args.torch_cuda and not args.compile_only:
   if not torch.cuda.is_available():row['variants']['torch-eager']={'error':'CUDA-enabled PyTorch/device unavailable; install an appropriate CUDA torch wheel'}
   else:
    try:
     cuda_inputs=[a.to('cuda') for a in x]
     class HostResult:
      def __init__(self,tensor):self.tensor=tensor
      def to_host(self):return self.tensor.detach().cpu().numpy()
     def eager(*dev):
      with torch.inference_mode():return HostResult(model(*dev))
     result=eager(*cuda_inputs).to_host();correct=bool(np.allclose(result,reference,atol=5e-5,rtol=2e-3))
     row['variants']['torch-eager']={'correct':correct,'max_abs':float(np.max(np.abs(result-reference))),'torch_version':torch.__version__}
     if correct:loaded['torch-eager']=(None,None,eager,cuda_inputs)
    except Exception as e:row['variants']['torch-eager']={'error':str(e)}
  if not args.compile_only and 'baseline' in loaded and len(loaded)>1:
   # to_host() synchronizes CUDA completion. This measures invocation + execution
   # + output download, not kernel-only latency; use saved VMFB with the IREE
   # benchmark CLI and Nsight for the separate device-side measurement.
   samples={v:[] for v in loaded};rng=random.Random(42)
   for _,_,fn,dev in loaded.values():
    for _ in range(10):fn(*dev).to_host()
   from gpu_selection import calibrate
   calibration=calibrate({v:(lambda fn=fn,dev=dev:fn(*dev).to_host()) for v,(_,_,fn,dev) in loaded.items()},'baseline')
   row['calibration']=calibration
   for _ in range(args.trials):
    order=list(loaded);rng.shuffle(order)
    for variant in order:
     _,_,fn,dev=loaded[variant];start=time.perf_counter_ns();fn(*dev).to_host();samples[variant].append((time.perf_counter_ns()-start)/1000)
   row['synchronized_host_samples_us']=samples;row['synchronized_host_median_us']={v:statistics.median(s) for v,s in samples.items()}
   med=row['synchronized_host_median_us']
   if 'optimized' in med:row['speedup']=med['baseline']/med['optimized']
   row['selected_holdout_speedup']=med['baseline']/med[calibration['selected']]
   row['selection_note']='Calibration frozen before holdout timing; deployment still requires independent multi-seed validation.'
  output.append(row);Path('research/results/gpu-'+args.arch+('-compile-only' if args.compile_only else '-runtime')+('-native' if args.native_only else '')+('-'+args.case if args.case else '')+'.json').write_text(json.dumps(output,indent=2));print(name,{v:r.get('correct',r.get('compiled',False)) for v,r in row['variants'].items()},flush=True)
 print('Completed',len(output),flush=True)
if __name__=='__main__':main()
