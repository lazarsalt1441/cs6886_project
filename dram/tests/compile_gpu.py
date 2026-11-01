"""Compile-only check; does not claim GPU execution or profiler validation."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from workloads import compile_module
from common import write_json,verify_assets

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--arch',default='sm_75');p.add_argument('--compiler',default='iree-compile');a=p.parse_args();verify_assets();rows=[]
    programs=[({'kind':'model','set':'A','id':'A_n2048_p0'},['iree_stable']),({'kind':'model','set':'B','id':'B_n2048_p0'},['iree_stable','iree_commit'])]
    programs += [({'kind':'synthetic','id':x},['iree_stable','iree_forwarded']) for x in ['cache_small','cache_large','cache_dense']]
    for case,backends in programs:
        for backend in backends:
            compile_module(case,backend,a.cache,a.arch,a.compiler)
            rows.append(dict(case=case,backend=backend,compiled=True,gpu_execution_tested=False));print('COMPILE PASS',case['id'],backend,flush=True)
    write_json(a.cache/'cuda_compile_final.json',rows)
