"""Independent CPU audit. IR counts are diagnostics, not DRAM traffic estimates."""
from pathlib import Path
import argparse, concurrent.futures as cf, subprocess, json, os, re, time
import numpy as np
import iree.runtime as rt
ROOT=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--compiler',default='/root/.local/bin/iree-compile');ap.add_argument('--label',default='pinned');ap.add_argument('--jobs',type=int,default=4);ap.add_argument('--only',default='');ap.add_argument('--profiles',default='');ap.add_argument('--backend',choices=['cpu','cuda'],default='cpu');ap.add_argument('--target',default='sm_75');a=ap.parse_args()
OUT=ROOT/'results'/a.label;OUT.mkdir(parents=True,exist_ok=True)
cases=json.loads((ROOT/'reproducers/cases.json').read_text())
if a.only:cases=[c for c in cases if any(c['name'].startswith(p) for p in a.only.split(','))]
configs={'default':[],'O3':['--iree-opt-level=O3'],'strong':['--iree-opt-level=O3','--iree-opt-generalize-matmul','--iree-dispatch-creation-enable-aggressive-reshape-movement'],'generalize':['--iree-opt-generalize-matmul'],'reshape':['--iree-dispatch-creation-enable-aggressive-reshape-movement'],'O0strong':['--iree-opt-level=O0','--iree-opt-generalize-matmul','--iree-dispatch-creation-enable-aggressive-reshape-movement']}
configs['early_decompose']=['--iree-opt-level=O3','--iree-preprocessing-pass-pipeline=builtin.module(util.func(iree-linalg-ext-convert-attention-to-online-attention,iree-linalg-ext-decompose-attention))']
configs['max_reassociation']=configs['strong']+['--iree-global-opt-enable-quantized-matmul-reassociation','--iree-llvmcpu-reassociate-fp-reductions','--acc-reassoc']
configs['aggressive']=configs['strong']+['--iree-dispatch-creation-enable-aggressive-fusion','--iree-dispatch-creation-fuse-multi-use','--iree-dispatch-creation-enable-fuse-horizontal-contractions','--iree-dispatch-creation-block-matmul-producer-fusion=false','--iree-dispatch-creation-element-wise-fuse-multi-reduction','--iree-llvmcpu-experimental-vectorize-to-transfer-gather']
version=subprocess.check_output([a.compiler,'--version'],text=True);(OUT/'version.txt').write_text(version)
def compile_one(c,k):
 d=OUT/(c['name']+'__'+k);d.mkdir(exist_ok=True)
 targetflags=['--iree-hal-target-backends=llvm-cpu','--iree-llvmcpu-target-cpu=host'] if a.backend=='cpu' else ['--iree-hal-target-backends=cuda','--iree-cuda-target='+a.target]
 cmd=[a.compiler,str(ROOT/'reproducers'/(c['name']+'.mlir'))]+targetflags+configs[k]
 results=[]
 for ph in ['global-optimization','flow','stream','end']:
  dest=d/('module.vmfb' if ph=='end' else ph+'.mlir');argv=cmd+([] if ph=='end' else ['--compile-to='+ph])+['-o',str(dest)]
  p=subprocess.run(argv,capture_output=True,text=True,timeout=180);(d/(ph+'.stderr')).write_text(p.stderr);results.append(dict(phase=ph,returncode=p.returncode,command=argv))
  if p.returncode:break
 rec=dict(case=c['name'],config=k,compiles=results,success=all(x['returncode']==0 for x in results))
 if rec['success']:
  patterns={'insert_slice':r'\btensor.insert_slice\b','matmul':r'\blinalg.(?:batch_)?matmul\b','flow_dispatch':r' = flow.dispatch @','stream_copy':r'\bstream.cmd.copy\b','stream_update':r'\bstream.async.update\b','online_attention':r'\biree_linalg_ext.online_attention\b'}
  rec['ir_counts']={ph:{k:len(re.findall(pat,(d/(ph+'.mlir')).read_text())) for k,pat in patterns.items()} for ph in ['global-optimization','flow','stream']}
 (d/'compile.json').write_text(json.dumps(rec,indent=2));return rec
jobs=[(c,k) for c in cases for k in (a.profiles.split(',') if a.profiles else (['default','O3','strong','generalize','reshape','O0strong'] if c['kind']=='gqa' else ['default','O3','strong']))]
with cf.ThreadPoolExecutor(a.jobs) as ex:records=list(ex.map(lambda z:compile_one(*z),jobs))
print('COMPILED',sum(r['success'] for r in records),'/',len(records),flush=True)
try:os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
except (AttributeError,OSError):pass
def sm(s):
 p=np.exp(s-s.max(axis=-1,keepdims=True));return p/p.sum(axis=-1,keepdims=True)
def samples(c,seed):
 rng=np.random.default_rng(seed);x=[rng.normal(0,.3,s).astype('float32') for s in c['shapes']];k=c['kind']
 if k=='forward':
  n,u=c['N'],c['U'];s=(seed*7)%(n-u+1);i=[max(0,s-1),s,s+u-1,min(n-1,s+u),(seed*17)%n][seed%5];x[-1]=np.array([s,i],dtype='int64');v=x[0].copy();v[s:s+u]=x[1];ref=v[i:i+1]
 elif k=='chain':
  s=(seed*7)%61;t=(s+[-4,-2,0,2,4][seed%5])%61;i=[s,t,min(63,s+3),min(63,t+3),63][seed%5];x[-1]=np.array([s,t,i],dtype='int64');v=x[0].copy();v[s:s+4]=x[1];v[t:t+4]=x[2];ref=v[i:i+1]
 elif k in ['gather_forks','coalesce']:
  x[-1]=np.array([0,127,128,129,131,132,256,c['N']-1],dtype='int64');u=x[0].copy();v=x[0].copy();u[128:132]=x[1];v[256:260]=x[2];ref=np.concatenate([u[x[-1]],v[x[-1]]],axis=0)
  if k=='coalesce':u[132:136]=x[1];ref=np.concatenate([u[x[-1]],v[x[-1]]],axis=0)
 elif k=='winner':x[-1]=np.array([seed%2],dtype='int32');ref=x[0].copy();ref[128:144]=x[1] if seed%2 else x[2]
 elif k=='winner_bool':x[-1]=np.array(bool(seed%2));ref=x[0].copy();ref[128:144]=x[1] if seed%2 else x[2]
 elif k=='concat':x[-1]=np.array(bool(seed%2));ref=np.concatenate([x[0],x[2] if seed%2 else x[3]],axis=2)
 elif k in ['same_slice','disjoint_chain']:ref=x[1]
 elif k=='overwrite':ref=x[0].copy();ref[8:12]=x[2]
 elif k=='projection':
  q,l,wk,wv=[v.astype('float64') for v in x];ref=sm(q@(l@wk).T/np.sqrt(c['D']))@(l@wv)
 elif k=='original_projection':
  q,l,w=[v.astype('float64') for v in x[:3]];key=l@w
  if c['rope']:key=np.concatenate([key,x[3].astype('float64')],axis=-1)
  ref=q@key.transpose(0,2,1)
 elif k=='partial_rope':
  q,l,wk,qr,kr,wv=[v.astype('float64') for v in x];ref=sm((q@(l@wk).T+qr@kr.T)/np.sqrt(20))@(l@wv)
 elif k=='gqa':ref=x[0].astype('float64')@np.repeat(x[1].astype('float64'),c['G'],axis=0).transpose(0,2,1)
 elif k=='native':
  if seed>=10:x[0].fill(0);x[1].fill(0);x[0][...,-1]=1;x[1][...,-1]=np.linspace(-4,4,32)
  q,key,v=[z.astype('float64') for z in x];ref=sm(q@key.transpose(0,2,1)/np.sqrt(c['D']))@v
 return x,np.asarray(ref,dtype='float32')
lookup={x['name']:x for x in cases}
for rec in records:
 if not rec['success']:continue
 c=lookup[rec['case']];d=OUT/(rec['case']+'__'+rec['config'])
 try:
  config=rt.Config('local-sync' if a.backend=='cpu' else 'cuda');ctx=rt.SystemContext(config=config);vm=rt.VmModule.mmap(config.vm_instance,str(d/'module.vmfb'));ctx.add_vm_module(vm);f=ctx.modules.module.main;checks=[]
  for seed in range(20 if c['kind']=='native' else 10):
   x,ref=samples(c,seed);dev=[rt.asdevicearray(config.device,v.copy()) for v in x];out=f(*dev).to_host().copy();diff=float(np.max(np.abs(out-ref)));ok=np.array_equal(out,ref) if c['kind'] in ['forward','chain','gather_forks','coalesce','concat','same_slice','overwrite','disjoint_chain','winner','winner_bool'] else np.allclose(out,ref,rtol=2e-3,atol=5e-5);checks.append(dict(seed=seed,passed=bool(ok),max_abs=diff))
  rec['validation']=checks
  if all(z['passed'] for z in checks):
   x,ref=samples(c,7);dev=[rt.asdevicearray(config.device,v.copy()) for v in x]
   for _ in range(10):
    out=f(*dev)
    if a.backend=='cuda':out.to_host()
   before=dict(config.device.allocator.statistics);ts=[]
   for _ in range(101):
    start=time.perf_counter_ns();out=f(*dev)
    if a.backend=='cuda':out.to_host()
    ts.append((time.perf_counter_ns()-start)/1000)
   after=dict(config.device.allocator.statistics);rec['timing']={'median_us':float(np.median(ts)),'p10_us':float(np.percentile(ts,10)),'p90_us':float(np.percentile(ts,90)),'samples_us':ts};rec['allocator']={'before':before,'after':after,'delta':{k:after[k]-before.get(k,0) for k in after if isinstance(after[k],(int,float))}}
 except Exception as e:rec['runtime_error']=repr(e)
 (d/'result.json').write_text(json.dumps(rec,indent=2));print(rec['case'],rec['config'],'PASS' if all(z['passed'] for z in rec.get('validation',[{'passed':False}])) else 'FAIL',flush=True)
if (OUT/'summary.json').exists():
 old=json.loads((OUT/'summary.json').read_text());keys={(r['case'],r['config']) for r in records};records=[r for r in old if (r['case'],r['config']) not in keys]+records
(OUT/'summary.json').write_text(json.dumps(records,indent=2));(OUT/'environment.json').write_text(json.dumps({'version':version,'uname':list(os.uname()),'affinity':list(os.sched_getaffinity(0)),'configs':configs},indent=2))
