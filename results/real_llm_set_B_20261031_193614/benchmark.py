"""Pinned pretrained full-model decode; genuine C++ before/after IR.
Timed inputs come from CPU HF prefill; weights, prefill and compilation excluded.
"""
import argparse,copy,hashlib,json,os,platform,random,statistics,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch,text_only
from transformers import AutoTokenizer,LlamaForCausalLM,LlamaConfig
from iree.turbine import aot
import iree.runtime as rt
from llm_workloads import Decode,Lookahead,flatten_cache,MODEL_ID,REVISION
ROOT=Path(__file__).resolve().parent
STRONG=['--iree-opt-level=O3','--iree-opt-generalize-matmul','--iree-dispatch-creation-enable-aggressive-reshape-movement']
CUDA_SAFE=['--iree-dispatch-creation-enable-split-reduction','--iree-dispatch-creation-split-argmax-threshold=1024','--iree-dispatch-creation-split-argmax-tile-size=256']
def write_json(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,default=str));tmp.replace(path)
def cpu_array(x):return x.detach().cpu().numpy() if isinstance(x,torch.Tensor) else np.asarray(x.to_host())
def load_model(tiny=False):
 if tiny:
  torch.manual_seed(0);c=LlamaConfig(vocab_size=128,hidden_size=64,intermediate_size=128,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,head_dim=16);m=LlamaForCausalLM(c).eval();m.config._attn_implementation='eager';return m
 return LlamaForCausalLM.from_pretrained(ROOT/'model',dtype=torch.float32,attn_implementation='eager',local_files_only=True).eval()
def compile_iree(src,target,device,arch,folder):
 folder.mkdir(parents=True,exist_ok=True)
 flags=['--iree-hal-target-backends=cuda','--iree-cuda-target='+arch,*CUDA_SAFE] if device=='cuda' else ['--iree-hal-target-backends=llvm-cpu','--iree-llvmcpu-target-cpu=host']
 cmd=['iree-compile',str(src),*flags,*STRONG,'--iree-hal-dump-executable-files-to='+str(folder/'codegen'),'-o',str(target)]
 write_json(folder/'compile_command.json',cmd);t=time.perf_counter()
 with (folder/'compile.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 return time.perf_counter()-t

def load_iree(vmfb,weights,driver):
 cfg=rt.Config(driver);index=rt.ParameterIndex();index.load(str(weights));provider=index.create_provider(scope='model')
 io=rt.create_io_parameters_module(cfg.vm_instance,provider);vm=rt.VmModule.mmap(cfg.vm_instance,str(vmfb));ctx=rt.SystemContext(config=cfg);ctx.add_vm_modules([io,vm])
 return dict(config=cfg,index=index,provider=provider,context=ctx,fn=ctx.modules.module.main)

def real_inputs(model,tokenizer,past,prompt_id,which,tiny=False):
 if tiny:ids=torch.arange(past).remainder(model.config.vocab_size).unsqueeze(0)
 else:
  prompts=['Explain why the sky looks blue in simple language. ', 'Write a short Python function to sort a list of numbers. ', 'A scientist measures the motion of a pendulum. ']
  seed=tokenizer(prompts[prompt_id%len(prompts)],add_special_tokens=True,return_tensors='pt').input_ids
  ids=seed.repeat(1,(past+seed.shape[1]-1)//seed.shape[1])[:,:past]
 with torch.inference_mode():o=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=True,logits_to_keep=1)
 flat=flatten_cache(o.past_key_values);logp=o.logits[0,-1].log_softmax(-1);values,tokens=logp.topk(2)
 mask=torch.ones(1,past+1,dtype=torch.int64);position=torch.tensor([[past]])
 args=(tokens[:1].reshape(1,1),mask,position,*flat) if which=='A' else (tokens.reshape(2,1),values,mask,position,*flat)
 return args,ids

def check_outputs(ref,got,name):
 if len(ref)!=len(got):raise AssertionError('Output count mismatch')
 rows=[]
 for i,(r,g) in enumerate(zip(ref,got)):
  r=cpu_array(r);g=cpu_array(g)
  if r.shape!=g.shape:raise AssertionError(f'{name} output {i} shape mismatch')
  if r.dtype.kind=='f':
   delta=np.abs(r.astype('float64')-g.astype('float64'));ok=bool(np.allclose(g,r,rtol=3e-3,atol=5e-4,equal_nan=False));row=dict(index=i,max_abs=float(delta.max()),rms=float(np.sqrt(np.mean(delta**2))),pass_=ok)
  else:ok=bool(np.array_equal(g,r));row=dict(index=i,exact=ok,pass_=ok)
  rows.append(row)
  if not ok:raise AssertionError(f'{name} numerical failure: {row}')
 return rows

def main():
 p=argparse.ArgumentParser();p.add_argument('--set',choices=['A','B'],required=True);p.add_argument('--device',choices=['cuda','cpu'],default='cuda');p.add_argument('--contexts',type=int,nargs='+',default=[128,512,2048]);p.add_argument('--prompts',type=int,default=2);p.add_argument('--trials',type=int,default=31);p.add_argument('--repetitions',type=int,default=3);p.add_argument('--warmup',type=int,default=5);p.add_argument('--steps',type=int,default=16);p.add_argument('--torch-compile',action='store_true');p.add_argument('--tiny',action='store_true');p.add_argument('--output',default='results');a=p.parse_args()
 if any(n<2 or n+a.steps>4096 for n in a.contexts):p.error('contexts >=2 and context+steps <=4096 required')
 if min(a.trials,a.repetitions,a.warmup,a.prompts,a.steps)<1:p.error('counts must be positive')
 out=ROOT/a.output;out.mkdir(parents=True,exist_ok=True);prepared=ROOT/'prepared'/(a.set+'_tiny' if a.tiny else a.set);audit=json.loads((prepared/'audit.json').read_text())
 for v in ['baseline','optimized']:
  if hashlib.sha256((prepared/(v+'.mlir')).read_bytes()).hexdigest()!=audit[v+'_sha256']:raise RuntimeError('Prepared IR hash mismatch')
  if 'hal.fence.await' not in (prepared/(v+'.mlir')).read_text():raise RuntimeError('Expected synchronous main ABI fence await')
 if not audit['dynamic']:raise RuntimeError('Dynamic cache export required')
 torch.set_num_threads(1);torch.manual_seed(0)
 if a.device=='cuda':
  if not torch.cuda.is_available():raise RuntimeError('Choose GPU runtime')
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 arch='sm_%d%d'%torch.cuda.get_device_capability() if a.device=='cuda' else None
 env=dict(python=sys.version,platform=platform.platform(),torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name() if a.device=='cuda' else None,arch=arch,config=vars(a),audit=audit)
 for name,cmd in [('iree_version',['iree-compile','--version']),('nvidia_smi',['nvidia-smi'])]:
  try:env[name]=subprocess.check_output(cmd,text=True,stderr=subprocess.STDOUT)
  except Exception as e:env[name]=str(e)
 write_json(out/'environment.json',env)
 model=load_model(a.tiny);tokenizer=None if a.tiny else AutoTokenizer.from_pretrained(ROOT/'model',local_files_only=True);wrapper=(Decode if a.set=='A' else Lookahead)(model).eval()
 weights=prepared/'weights.irpa'
 if not weights.exists():print('Creating parameter archive',flush=True);aot.save_module_parameters(weights,wrapper)
 variants=['baseline']+(['optimized'] if audit['any_rewrite'] and not audit['identical_ir'] else [])
 runtimes={};compile_rows={}
 for v in variants:
  folder=out/('iree_'+v);vmfb=folder/'module.vmfb';print('Compiling full model:',v,arch,flush=True)
  compile_rows[v]=compile_iree(prepared/(v+'.mlir'),vmfb,a.device,arch,folder);runtimes['iree_'+v]=load_iree(vmfb,weights,'cuda' if a.device=='cuda' else 'local-sync')
 write_json(out/'compilation.json',compile_rows)
 inputs=[]
 for n in a.contexts:
  for prompt in range(a.prompts):
   args,ids=real_inputs(model,tokenizer,n,prompt,a.set,a.tiny)
   with torch.inference_mode():gold=tuple(z.detach().clone() for z in wrapper(*args))
   inputs.append((n,prompt,args,ids,gold))
 sdpa_model=copy.deepcopy(model);sdpa_model.config._attn_implementation='sdpa'
 sdpa_wrapper=(Decode if a.set=='A' else Lookahead)(sdpa_model).eval().to(a.device)
 wrapper.to(a.device)
 def sync():
  if a.device=='cuda':torch.cuda.synchronize()
 compiled=None
 if a.torch_compile:
  try:compiled=torch.compile(wrapper,fullgraph=True,dynamic=False)
  except Exception:(out/'torch_compile_error.log').write_text(traceback.format_exc())
 rows=[];correctness=[];torch_first=[]
 for n,prompt,host_args,ids,gold in inputs:
  torch_args=tuple(z.to(a.device).clone() for z in host_args);calls={'torch_eager':lambda:wrapper(*torch_args),'torch_sdpa':lambda:sdpa_wrapper(*torch_args)};resident_inputs={}
  for name,r in runtimes.items():
   resident=tuple(rt.asdevicearray(r['config'].device,z.numpy().copy()) for z in host_args);resident_inputs[name]=resident;calls[name]=(lambda r=r,resident=resident:r['fn'](*resident))
  if compiled is not None:
   try:
    t=time.perf_counter()
    with torch.inference_mode():tmp=compiled(*torch_args);sync()
    torch_first.append(dict(context=n,prompt=prompt,first_call_s=time.perf_counter()-t,includes_execution=True));calls['torch_compile']=lambda:compiled(*torch_args)
   except Exception:
    with (out/'torch_compile_error.log').open('a') as f:f.write(traceback.format_exc())
    compiled=None
  with torch.inference_mode():
   outputs={name:fn() for name,fn in calls.items()};sync()
   for name,result in outputs.items():correctness.append(dict(context=n,prompt=prompt,backend=name,phase='before',checks=check_outputs(gold,result,name)))
   if 'iree_optimized' in outputs:correctness.append(dict(context=n,prompt=prompt,backend='optimized_vs_baseline',checks=check_outputs(outputs['iree_baseline'],outputs['iree_optimized'],'optimized_vs_baseline')))
   write_json(out/'correctness.json',correctness);del outputs
   for fn in calls.values():
    for _ in range(a.warmup):tmp=fn()
    sync()
   for mode in ['resident_outputs','logits_to_host']:
    for rep in range(a.repetitions):
     samples={k:[] for k in calls};rng=random.Random(10000*n+100*prompt+rep)
     for _ in range(a.trials):
      order=list(calls);rng.shuffle(order)
      for name in order:
       sync();t=time.perf_counter();tmp=calls[name]()
       if mode=='logits_to_host':
        cpu_array(tmp[0])
        if a.set=='B':cpu_array(tmp[1])
       sync();samples[name].append((time.perf_counter()-t)*1e3)
     for name,values in samples.items():rows.append(dict(context=n,prompt=prompt,mode=mode,repetition=rep,backend=name,median_ms=statistics.median(values),p10_ms=float(np.percentile(values,10)),p90_ms=float(np.percentile(values,90)),samples_ms=values))
     write_json(out/'latency.json',rows);print('TIMED',a.set,n,prompt,mode,rep,{k:round(statistics.median(v),3) for k,v in samples.items()},flush=True)
   for name,fn in calls.items():correctness.append(dict(context=n,prompt=prompt,backend=name,phase='after',checks=check_outputs(gold,fn(),name)))
   for values in [torch_args,*resident_inputs.values()]:
    for original,value in zip(host_args,values):np.testing.assert_array_equal(cpu_array(value),original.numpy())
   write_json(out/'correctness.json',correctness)
  del calls,torch_args,resident_inputs
  if a.device=='cuda':torch.cuda.empty_cache()
 write_json(out/'torch_compilation.json',torch_first)
 generation=[]
 for n,prompt,host_args,ids,_ in inputs[::a.prompts]:
  for name in ['torch_eager','torch_sdpa',*runtimes]:
   torch_call=name.startswith('torch_');active_wrapper=sdpa_wrapper if name=='torch_sdpa' else wrapper
   args=tuple(z.to(a.device) for z in host_args) if torch_call else tuple(rt.asdevicearray(runtimes[name]['config'].device,z.numpy()) for z in host_args)
   chosen=[];t=time.perf_counter()
   with torch.inference_mode():
    for step in range(a.steps):
     result=active_wrapper(*args) if torch_call else runtimes[name]['fn'](*args)
     logits=cpu_array(result[0])[0,-1].astype('float64');logp=logits-np.logaddexp.reduce(logits)
     winner=0 if a.set=='A' else int(cpu_array(result[1]));chosen.append(int(cpu_array(args[0])[winner,0]));flat=result[1 if a.set=='A' else 2:]
     next_ids=np.argsort(logp)[-2:][::-1].copy();length=n+step+1;mask=np.ones((1,length+1),np.int64);position=np.array([[length]],np.int64)
     small=(next_ids[:1].reshape(1,1),mask,position) if a.set=='A' else (next_ids.reshape(2,1),logp[next_ids].astype(np.float32),mask,position)
     args=tuple(torch.from_numpy(z).to(a.device) for z in small)+tuple(flat) if torch_call else tuple(rt.asdevicearray(runtimes[name]['config'].device,z) for z in small)+tuple(flat)
    sync()
   seconds=time.perf_counter()-t;generation.append(dict(context=n,prompt=prompt,backend=name,steps=a.steps,seconds=seconds,tokens_per_second=a.steps/seconds,token_ids=chosen,text=None if a.tiny else tokenizer.decode(chosen),scope='Single host-driven diagnostic: includes logits/scalar transfers, CPU top2 and small uploads. Not repeated steady-state timing.'))
   write_json(out/'generation.json',generation)
 agreements=[]
 for n,prompt,*_ in inputs[::a.prompts]:
  group=[g for g in generation if g['context']==n and g['prompt']==prompt];gold=group[0]['token_ids'];agreements.extend(dict(context=n,prompt=prompt,backend=g['backend'],exact_tokens=g['token_ids']==gold) for g in group)
 write_json(out/'generation_agreement.json',agreements)
 write_json(out/'summary.json',dict(status='PASS',set=a.set,optimization_matches=audit['markers'],no_rewrite_control=not audit['any_rewrite'],workload='Ordinary greedy decode' if a.set=='A' else 'Custom two-candidate lookahead; not HF beam/speculative decoding',timing_scope='Full model step, GPU-resident KV, f32 eager-attention exported IREE; Torch eager and automatic SDPA references. SDPA kernel selection is not profiled. Prefill/compilation excluded. IREE main ABI fence-await synchronous; Torch CUDA synchronized.',correctness_cases=len(correctness),generation_agreement=agreements,compile_s=compile_rows,memory_note='No allocation/DRAM claim; PTX or logical IR change is insufficient evidence.'))
 print('COMPLETE',out,flush=True)
if __name__=='__main__':main()
