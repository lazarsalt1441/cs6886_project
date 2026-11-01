import sys,subprocess,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from test_concat_factor import source,opt,ROOT
import numpy as np
import iree.runtime as rt
out=ROOT/'local_checks/concat';out.mkdir(parents=True,exist_ok=True);checks=[]
for dynamic in [False,True]:
 funcs=[]
 for variant,s in [('baseline',source(None if dynamic else 8)),('optimized',opt(source(None if dynamic else 8)))]:
  p=out/f'{dynamic}-{variant}.mlir';p.write_text(s)
  with (out/'compile.log').open('a') as log:subprocess.run(['iree-compile',str(p),'--iree-hal-target-backends=llvm-cpu','--iree-llvmcpu-target-cpu=host','-o',str(p.with_suffix('.vmfb'))],stdout=log,stderr=subprocess.STDOUT,check=True)
  cfg=rt.Config('local-sync');ctx=rt.SystemContext(config=cfg);ctx.add_vm_module(rt.VmModule.mmap(cfg.vm_instance,str(p.with_suffix('.vmfb'))));funcs.append((ctx,ctx.modules.module.main))
 for n in ([2,8,33] if dynamic else [8]):
  for winner in [False,True]:
   h=np.random.default_rng(n).standard_normal((1,2,n,4),dtype=np.float32);a=np.array([0.,-0.,np.nan,np.inf,-np.inf,1.,-1.,2.],np.float32).reshape(1,2,1,4);b=np.flip(a,axis=-1).copy();expected=np.concatenate((h,a if winner else b),axis=2)
   for _,fn in funcs:np.testing.assert_array_equal(fn(h,-h,a,b,np.array(winner)).to_host().view('u4'),expected.view('u4'))
   checks.append(dict(dynamic=dynamic,past=n,winner=winner,bitwise=True))
(out/'results.json').write_text(json.dumps(checks,indent=2));print('PASS',len(checks),'bitwise cases incl. NaNs and signed zero')
