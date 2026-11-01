"""Additional contexts and an explicit same-VMFB dense placebo. GPU required except --print-plan."""
import argparse,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'dram'))
from common import write_json,sha

def make_plan(contexts):
    if not contexts or len(set(contexts))!=len(contexts) or any(n<8 or n>4096 for n in contexts):raise ValueError('Contexts must be within the validated export range 8..4096')
    counter_contexts=set(contexts)&{1792,2048,2304} or {contexts[0]}
    cases=[dict(id='stream_control',kind='control',size_mib=64),dict(id='cache_dense',kind='synthetic')]
    for n in contexts:
        for p in range(3):cases.append(dict(id=f'B_n{n}_p{p}',kind='model',set='B',context=n,prompt=p))
    jobs=[dict(case='stream_control',backend='torch_stream',rep=0,cache='warm',calls=3,replay='app-range')]
    for c in cases[1:]:
        if c['kind']=='model' and (c['prompt']!=0 or c['context'] not in counter_contexts):continue
        for b in ['iree_stable','iree_forwarded'] if c['kind']=='synthetic' else ['iree_stable','iree_commit']:
            for rep in range(3):jobs.append(dict(case=c['id'],backend=b,rep=rep,cache='warm',calls=3,replay='app-range'))
    return dict(version=3,track='B',plan='evaluation_followup',cases=cases,profiles=jobs,
                dense_control='iree_forwarded label loads a byte-identical copy of the stable VMFB; no cache optimization is applied to this placebo',
                note='New protocol: more warmup, seven repetition blocks, previous outputs released outside timer. Do not pool with original timings.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--contexts',type=int,nargs='+',default=[512,1024,1536,1792,2048,2304,2560,3072,4096]);p.add_argument('--cache',type=Path,default=ROOT/'evaluation/cache');p.add_argument('--output',type=Path,default=ROOT/'evaluation/gpu_run');p.add_argument('--print-plan',action='store_true');a=p.parse_args();plan=make_plan(a.contexts)
    if a.print_plan:print(json.dumps(plan,indent=2));return
    import workloads,study
    original=workloads.compile_module
    def compile_with_placebo(case,backend,cache,arch=None,compiler='iree-compile',device='cuda'):
        if case['id']!='cache_dense' or backend!='iree_forwarded':return original(case,backend,cache,arch,compiler,device)
        base=original(case,'iree_stable',cache,arch,compiler,device);folder=base.parent.with_name('cache_dense_iree_forwarded');folder.mkdir(parents=True,exist_ok=True)
        target=folder/'module.vmfb';shutil.copy2(base,target)
        record=json.loads((base.parent/'build.json').read_text());record.update(same_binary_placebo=True,copied_from=str(base),vmfb_sha256=sha(target));write_json(folder/'build.json',record)
        assert sha(base)==sha(target)
        return target
    workloads.compile_module=compile_with_placebo
    study.build_plan=lambda *unused:plan
    sys.argv=[str(ROOT/'dram/study.py'),'--track','B','--plan','extended','--cache',str(a.cache.resolve()),'--output',str(a.output.resolve()),'--latency-repetitions','7','--latency-warmup-calls','100']
    study.main()
    # Keep full Systems traces at either side of 2048 as well as at 2048.
    from types import SimpleNamespace
    args=SimpleNamespace(cache=a.cache.resolve(),timeout=1800)
    for n in [1792,2304]:
        if n not in a.contexts:continue
        case=next(c for c in plan['cases'] if c.get('context')==n and c['prompt']==0)
        for backend in ['iree_stable','iree_commit']:
            study.nsys_profile(case,backend,args,a.output.resolve())
if __name__=='__main__':main()
