"""Immutable input fixtures and resident runners. Setup never enters the profile range."""
from pathlib import Path
import copy, ctypes, hashlib, json, os, shutil, subprocess, sys, time
import numpy as np
from common import ROOT, STRONG, PRECISION, CUDA_SAFE, source_for, sha, write_json

def synthetic_inputs(label, seed=171):
    shape=json.loads((ROOT/'assets/synthetic/cases.json').read_text())[label]
    pool,n,h,g,d,m,f,p,t=shape;rng=np.random.default_rng(seed)
    host=[rng.standard_normal(s).astype('float32') for s in [(f,1,h*g,m,d),(pool,h,d),(pool,h,d),(f,t,h,d),(f,t,h,d)]]
    ids=rng.integers(pool,size=n,dtype='int64');ids[:min(t,n)]=np.arange(p,p+min(t,n));ids[-1]=pool-1
    return host+[ids],shape

def synthetic_reference(host, shape):
    pool,n,h,g,d,m,f,p,t=shape;qs,k,v,nks,nvs,ids=host;outs=[]
    for branch in range(f):
        kk=k.astype('float64').copy();vv=v.astype('float64').copy();kk[p:p+t]=nks[branch];vv[p:p+t]=nvs[branch]
        kr=np.repeat(kk[ids].transpose(1,0,2),g,axis=0)[None];vr=np.repeat(vv[ids].transpose(1,0,2),g,axis=0)[None]
        scores=qs[branch].astype('float64')@kr.transpose(0,1,3,2)/np.sqrt(d);scores-=scores.max(axis=-1,keepdims=True)
        weights=np.exp(scores);weights/=weights.sum(axis=-1,keepdims=True);outs.append(weights@vr)
    return outs

def compact(outputs, case):
    offset=1 if case.get('set')=='A' else 2
    result={}
    for i,x in enumerate(outputs):
        y=x.detach().cpu().numpy() if hasattr(x,'detach') else np.asarray(x)
        result['out'+str(i)]=y[:,:,-1:,:].copy() if case['kind']=='model' and i>=offset else y.copy()
    return result

def configure_torch():
    import torch
    torch.set_num_threads(1);torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    return torch

def model_modules():
    root=ROOT/'assets/model'
    sys.path.insert(0,str(root))
    import benchmark as b
    b.ROOT=root
    from llm_workloads import Decode,Lookahead
    return b,Decode,Lookahead

def ensure_model(reuse=None):
    root=ROOT/'assets/model';model=root/'model'
    if not (model/'model.safetensors').exists() and reuse:
        candidate=Path(reuse)/'model'
        if (candidate/'model.safetensors').is_file():
            if model.is_symlink(): model.unlink()
            if not model.exists(): model.symlink_to(candidate.resolve(),target_is_directory=True)
    if not (model/'model.safetensors').exists():
        subprocess.run([sys.executable,str(root/'scripts/download_model.py')],check=True)
    expected=json.loads((root/'checkpoint_sha256.json').read_text())
    for n,value in expected.items():
        if sha(model/n)!=value: raise RuntimeError('Model checksum mismatch '+n)
    return model

def prepare_cases(cases, cache, reuse=None):
    cache=Path(cache);torch=None;model=None;tokenizer=None;wrappers={}
    for case in cases:
        folder=cache/'data'/case['id'];folder.mkdir(parents=True,exist_ok=True)
        paths=[folder/'inputs.npz',folder/'ref_natural.npz']
        if case.get('set')=='B': paths += [folder/'ref_w0.npz',folder/'ref_w1.npz']
        done=folder/'fixture.json'
        if done.exists():
            manifest=json.loads(done.read_text())
            weights_ok=case['kind']!='model' or (cache/'weights'/case['set']/'weights.irpa').exists()
            if weights_ok and manifest.get('case')==case and all(p.exists() and sha(p)==manifest['sha256'].get(p.name) for p in paths): continue
        if case['kind']=='control':
            write_json(done,dict(case=case,note='Random streaming data created on device outside measured window'))
            continue
        print('PREPARE',case['id'],flush=True)
        if case['kind']=='synthetic':
            host,shape=synthetic_inputs(case['id']);refs=compact(synthetic_reference(host,shape),case)
            np.savez(folder/'inputs.npz',**{'arg'+str(i):a for i,a in enumerate(host)})
            np.savez(folder/'ref_natural.npz',**refs)
        else:
            if torch is None:
                torch=configure_torch();ensure_model(reuse);b,Decode,Lookahead=model_modules()
                from transformers import AutoTokenizer
                model=b.load_model();tokenizer=AutoTokenizer.from_pretrained(ROOT/'assets/model/model',local_files_only=True)
            which=case['set']
            if which not in wrappers:
                wrapper=(Decode if which=='A' else Lookahead)(model).eval();wrappers[which]=wrapper
                weights=cache/'weights'/which/'weights.irpa';weights.parent.mkdir(parents=True,exist_ok=True)
                if not weights.exists():
                    existing=Path(reuse)/'prepared'/which/'weights.irpa' if reuse else None
                    # Recreate from verified parameters; never trust an unverified old weights archive.
                    b.aot.save_module_parameters(weights,wrapper)
                write_json(weights.parent/'source.json',dict(model_revision='12fd25f77366fa6b3b4b768ec3050bf629380bac',sha256=sha(weights)))
            wrapper=wrappers[which]
            host_t,_=b.real_inputs(model,tokenizer,case['context'],case['prompt'],which)
            host=[x.detach().numpy().copy() for x in host_t]
            np.savez(folder/'inputs.npz',**{'arg'+str(i):x for i,x in enumerate(host)})
            with torch.inference_mode():
                np.savez(folder/'ref_natural.npz',**compact(wrapper(*host_t),case))
                if which=='B':
                    for wanted in [0,1]:
                        priors=torch.tensor([0.,-100.] if wanted==0 else [-100.,0.])
                        outputs=wrapper(host_t[0],priors,*host_t[2:]);assert int(outputs[1])==wanted
                        np.savez(folder/f'ref_w{wanted}.npz',**compact(outputs,case))
        write_json(done,dict(case=case,seed=171,sha256={p.name:sha(p) for p in paths},
                            reference='FP64 NumPy attention' if case['kind']=='synthetic' else 'Pinned CPU HF eager model; compact cache rows plus exact input-prefix contract'))

def load_fixture(cache, case, winner='natural'):
    folder=Path(cache)/'data'/case['id']
    with np.load(folder/'inputs.npz',allow_pickle=False) as z:host=[z['arg'+str(i)].copy() for i in range(len(z))]
    if case.get('set')=='B' and winner!='natural': host[1]=np.asarray([0.,-100.] if winner=='0' else [-100.,0.],dtype='float32')
    suffix='natural' if winner=='natural' else 'w'+winner
    with np.load(folder/f'ref_{suffix}.npz',allow_pickle=False) as z:refs={k:z[k].copy() for k in z.files}
    return host,refs

def compile_module(case, backend, cache, arch, compiler='iree-compile', device='cuda'):
    src=source_for(case,backend)
    family=(case['set'] if case['kind']=='model' else case['id'])
    folder=Path(cache)/'compiled'/device/(family+'_'+backend)
    folder.mkdir(parents=True,exist_ok=True);target=folder/'module.vmfb'
    version=subprocess.check_output([compiler,'--version'],text=True)
    flags=[*STRONG,*PRECISION]
    flags+=['--iree-hal-target-backends=cuda','--iree-cuda-target='+arch,*CUDA_SAFE] if device=='cuda' else ['--iree-hal-target-backends=llvm-cpu','--iree-llvmcpu-target-cpu=host']
    contract=dict(source_sha256=sha(src),compiler=version,flags=flags)
    manifest=folder/'build.json'
    if manifest.exists() and target.exists():
        prior=json.loads(manifest.read_text())
        if prior.get('contract')==contract and sha(target)==prior.get('vmfb_sha256'): return target
    cmd=[compiler,str(src),*flags,'--iree-hal-dump-executable-files-to='+str(folder/'codegen'),'-o',str(target)]
    start=time.perf_counter()
    with (folder/'compile.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
    write_json(manifest,dict(contract=contract,command=cmd,seconds=time.perf_counter()-start,vmfb_sha256=sha(target)))
    if device=='cuda' and case['kind']=='model':
        has_f64=any('f64' in p.read_text(errors='replace') for p in (folder/'codegen').rglob('*.ptx'))
        if not has_f64: raise RuntimeError('Expected preserved f64 trig reduction absent from PTX')
    return target

def load_iree(vmfb, case, cache, driver='cuda'):
    import iree.runtime as rt
    cfg=rt.Config(driver);context=rt.SystemContext(config=cfg);keep=[cfg,context]
    if case['kind']=='model':
        index=rt.ParameterIndex();index.load(str(Path(cache)/'weights'/case['set']/'weights.irpa'))
        provider=index.create_provider(scope='model');io=rt.create_io_parameters_module(cfg.vm_instance,provider)
        context.add_vm_module(io);keep.extend([index,provider,io])
    vm=rt.VmModule.mmap(cfg.vm_instance,str(vmfb));context.add_vm_module(vm);keep.append(vm)
    return cfg,context.modules.module.main,keep

class BranchSynthetic:
    """Natural Torch counterpart of the saved genuine C++ pass fixture."""
    def __init__(self,shape):self.shape=shape
    def __call__(self,qs,k,v,nks,nvs,ids):
        import torch
        pool,n,h,g,d,m,f,p,t=self.shape;out=[]
        for branch in range(f):
            kb=torch.slice_scatter(k,nks[branch],dim=0,start=p,end=p+t)
            vb=torch.slice_scatter(v,nvs[branch],dim=0,start=p,end=p+t)
            kk=kb.index_select(0,ids).permute(1,0,2).unsqueeze(0)
            vv=vb.index_select(0,ids).permute(1,0,2).unsqueeze(0)
            out.append(torch.nn.functional.scaled_dot_product_attention(qs[branch],kk,vv,enable_gqa=True))
        return tuple(out)

class Runner:
    def __init__(self,case,backend,cache,device='cuda',winner='natural'):
        self.case=case;self.backend=backend;self.cache=Path(cache);self.keep=[];self.device=device
        if case['kind']=='control':
            torch=configure_torch();self.torch=torch;n=case['size_mib']*1024*1024//4
            self.inputs=[torch.randn(n,device=device)];self.dst=torch.empty_like(self.inputs[0]);self.host=None;self.refs=None
            self.fn=lambda:torch.add(self.inputs[0],0.,out=self.dst)
            return
        self.host,self.refs=load_fixture(cache,case,winner)
        if backend.startswith('iree_'):
            import iree.runtime as rt
            family=case['set'] if case['kind']=='model' else case['id']
            vm=self.cache/'compiled'/device/(family+'_'+backend)/'module.vmfb'
            self.cfg,fn,keep=load_iree(vm,case,cache,'cuda' if device=='cuda' else 'local-sync');self.keep.extend(keep)
            self.inputs=[rt.asdevicearray(self.cfg.device,z.copy()) for z in self.host];self.fn=lambda:fn(*self.inputs)
        else:
            torch=configure_torch();self.torch=torch
            self.inputs=[torch.from_numpy(x.copy()).to(device) for x in self.host]
            if case['kind']=='synthetic':
                shape=json.loads((ROOT/'assets/synthetic/cases.json').read_text())[case['id']]
                workload=BranchSynthetic(shape)
            else:
                b,Decode,Lookahead=model_modules();model=b.load_model()
                if backend=='torch_sdpa':model.config._attn_implementation='sdpa'
                workload=(Decode if case['set']=='A' else Lookahead)(model).to(device).eval();self.keep.append(workload)
            if backend=='torch_compile':workload=torch.compile(workload,fullgraph=True,dynamic=False)
            self.fn=lambda:workload(*self.inputs);self.keep.append(workload)

    def inputs_unchanged(self):
        if self.host is None:return True
        return all(np.array_equal(x.detach().cpu().numpy() if hasattr(x,'detach') else x.to_host(),h) for x,h in zip(self.inputs,self.host))

    def validate(self,result):
        from common import check_arrays
        if self.case['kind']=='control':
            import torch
            return dict(pass_=bool(torch.equal(result,self.inputs[0])),expected_stream_read_bytes=self.inputs[0].numel()*4,expected_stream_write_bytes=self.inputs[0].numel()*4)
        return check_arrays(result,self.host,self.refs,self.case)

    def anchor(self):
        if self.backend.startswith('iree_'):
            # Borrow DLPack's first DLTensor field (data pointer) without consuming
            # the capsule. Keep the capsule alive until context discovery finishes.
            cap=self.cfg.device.create_dlpack_capsule(self.inputs[0]._buffer_view,2,0)
            get=ctypes.pythonapi.PyCapsule_GetPointer;get.restype=ctypes.c_void_p;get.argtypes=[ctypes.py_object,ctypes.c_char_p]
            managed=get(cap,b'dltensor');ptr=ctypes.cast(managed,ctypes.POINTER(ctypes.c_void_p))[0]
            return int(ptr)
        return self.inputs[0].data_ptr()

class CudaContext:
    """Profile the context which owns the actual IREE/Torch input, not an unrelated primary context."""
    def __init__(self,device_pointer):
        self.lib=ctypes.CDLL('libcuda.so.1');self.ctx=ctypes.c_void_p()
        self.lib.cuPointerGetAttribute.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_uint64]
        self.check(self.lib.cuPointerGetAttribute(ctypes.byref(self.ctx),1,device_pointer),'cuPointerGetAttribute(CONTEXT)')
        if not self.ctx.value:raise RuntimeError('Null owning CUDA context')
        self.push=getattr(self.lib,'cuCtxPushCurrent_v2');self.push.argtypes=[ctypes.c_void_p]
        self.pop=getattr(self.lib,'cuCtxPopCurrent_v2');self.pop.argtypes=[ctypes.POINTER(ctypes.c_void_p)]
    @staticmethod
    def check(code,api):
        if code:raise RuntimeError(f'{api}: CUDA error {code}')
    def __enter__(self):
        self.check(self.push(self.ctx),'cuCtxPushCurrent');return self
    def __exit__(self,*exc):
        old=ctypes.c_void_p();self.check(self.pop(ctypes.byref(old)),'cuCtxPopCurrent')
    def sync(self):self.check(self.lib.cuCtxSynchronize(),'cuCtxSynchronize')
