"""Actual IREE CPU compilation and execution of saved C++ before/after IR."""
import argparse,json,sys,tempfile,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import ROOT,check_arrays,write_json,verify_assets
from workloads import synthetic_inputs,synthetic_reference,compile_module,load_iree,prepare_cases,Runner
import iree.runtime as rt

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--compiler',default='iree-compile');a=p.parse_args();verify_assets();rows=[]
    for label in ['cache_small','cache_large','cache_dense']:
        case=dict(id=label,kind='synthetic')
        for backend in ['iree_stable','iree_forwarded']:
            vm=compile_module(case,backend,a.cache,None,a.compiler,'cpu');cfg,fn,keep=load_iree(vm,case,a.cache,'local-sync')
            for seed in [17,171,991]:
                host,shape=synthetic_inputs(label,seed);refs={'out'+str(i):x for i,x in enumerate(synthetic_reference(host,shape))}
                dev=[rt.asdevicearray(cfg.device,x.copy()) for x in host];result=check_arrays(fn(*dev),host,refs,case)
                unchanged=all(np.array_equal(x.to_host(),h) for x,h in zip(dev,host))
                row=dict(case=label,backend=backend,seed=seed,pass_=result['pass_'] and unchanged,max_abs=max(x['max_abs'] for x in result['outputs']))
                rows.append(row);print(row,flush=True);assert row['pass_']
    # Exercise the actual fixture/Runner interface used by the GPU worker too.
    runner_checks=[]
    for label in ['cache_small','cache_large','cache_dense']:
        case=dict(id=label,kind='synthetic');prepare_cases([case],a.cache)
        for backend in ['iree_stable','iree_forwarded']:
            r=Runner(case,backend,a.cache,'cpu');result=r.validate(r.fn())
            assert result['pass_'] and r.inputs_unchanged()
            runner_checks.append(dict(case=label,backend=backend,pass_=True))
    # Keep the separate single-fork ownership probe quarantined if stock
    # lowering changes caller inputs. Numerical agreement alone is insufficient.
    negative=[]
    case=dict(id='cache_single',kind='synthetic')
    for backend in ['iree_stable','iree_forwarded']:
        vm=compile_module(case,backend,a.cache,None,a.compiler,'cpu');cfg,fn,keep=load_iree(vm,case,a.cache,'local-sync')
        host,shape=synthetic_inputs('cache_single',17);refs={'out'+str(i):x for i,x in enumerate(synthetic_reference(host,shape))}
        dev=[rt.asdevicearray(cfg.device,x.copy()) for x in host];result=check_arrays(fn(*dev),host,refs,case)
        changed=[i for i,(x,h) in enumerate(zip(dev,host)) if not np.array_equal(x.to_host(),h)]
        negative.append(dict(backend=backend,numerical_pass=result['pass_'],changed_arguments=changed,eligible=not changed and result['pass_']))
    write_json(a.cache/'cpu_integration.json',dict(execution='CPU only',checks=rows,runner_checks=runner_checks,single_fork_ownership_probe=negative));print('CPU execution PASS',len(rows),'runner checks',len(runner_checks),'single-fork probe',negative)
if __name__=='__main__':main()
