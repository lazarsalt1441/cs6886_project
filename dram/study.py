"""Compile, correctness-gate, time and profile the saved C++ attention optimizations."""
import argparse, hashlib, json, os, random, shutil, subprocess, sys, time, traceback
from pathlib import Path
from common import ROOT,backends,build_plan,ncu_command,ncu_export_command,parse_ncu,sha,verify_assets,write_json

def run(cmd,log,timeout=3600):
    log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w') as f:
        subprocess.run([str(x) for x in cmd],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=timeout)

def capture(cmd):
    try:return subprocess.check_output(cmd,text=True,stderr=subprocess.STDOUT,timeout=30)
    except Exception as e:return str(e)

def profile(job,case,args,out,environment):
    key=f"{case['id']}_{job['backend']}_{job['cache']}_{job['replay']}_r{job['rep']}"
    folder=out/'profiles'/key;folder.mkdir(parents=True,exist_ok=True)
    task={**job,'case':case,'cache_dir':str(args.cache.resolve()),'warmup':5,'result':str((folder/'worker.json').resolve())}
    stable_environment={k:environment[k] for k in ['torch','cuda','python','gpu','arch','iree','ncu','tf32','model_precision','assets_manifest_sha256','gpu_identity']}
    task['environment_sha256']=hashlib.sha256(json.dumps(stable_environment,sort_keys=True).encode()).hexdigest()
    task['fixture_sha256']=sha(args.cache/'data'/case['id']/'fixture.json')
    if job['backend'].startswith('iree_'):
        family=case['set'] if case['kind']=='model' else case['id']
        task['build_sha256']=sha(args.cache/'compiled/cuda'/(family+'_'+job['backend'])/'build.json')
    old=folder/'collection.json'
    if old.exists():
        row=json.loads(old.read_text())
        if row.get('valid') and row.get('task')==task:
            try:parse_ncu(folder/'metrics.csv',job['calls'],job['replay']);return row
            except Exception:pass
    write_json(folder/'task.json',task);row=dict(task=task,valid=False)
    stem=folder/'profile'
    command=ncu_command(args.ncu,ROOT/'worker.py',folder/'task.json',stem,sys.executable,job['replay'],job['replay']=='application')
    write_json(folder/'command.json',command);start=time.perf_counter()
    try:
        # Recover a completed capture whose export failed, without recapturing.
        reusable=False
        worker_path=folder/'worker.json'
        if (Path(str(stem)+'.ncu-rep').exists() and worker_path.exists()):
            prior_worker=json.loads(worker_path.read_text())
            reusable=prior_worker.get('task')==task and prior_worker.get('completed') and prior_worker.get('valid')
        if not reusable:run(command,folder/'ncu.log',args.timeout)
        row['capture_reused']=bool(reusable)
        export=ncu_export_command(args.ncu,str(stem)+'.ncu-rep')
        write_json(folder/'export_command.json',export)
        run(export,folder/'metrics.csv',120)
        worker=json.loads((folder/'worker.json').read_text())
        if not worker.get('valid') or not worker.get('completed'):raise RuntimeError('Worker not complete/valid')
        row['metrics']=parse_ncu(folder/'metrics.csv',job['calls'],job['replay'])
        if case['kind']=='control' and row['metrics']['total_bytes_per_call']<=0:raise RuntimeError('Zero traffic in streaming counter control')
        row.update(valid=True,seconds=time.perf_counter()-start)
    except Exception:row.update(error=traceback.format_exc(),seconds=time.perf_counter()-start)
    write_json(old,row);return row

def nsys_profile(case,backend,args,out):
    from analyze import nsys_summary
    nsys=shutil.which('nsys')
    if not nsys:return dict(status='not installed; copy-engine attribution unavailable')
    folder=out/'nsys'/(case['id']+'_'+backend);folder.mkdir(parents=True,exist_ok=True)
    task=dict(case=case,backend=backend,cache_dir=str(args.cache.resolve()),cache='warm',calls=3,warmup=5,result=str((folder/'worker.json').resolve()))
    write_json(folder/'task.json',task);stem=folder/'trace'
    cmd=[nsys,'profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none',
         '--capture-range=nvtx','--nvtx-capture=ATTENTION_MEASURED_STEP',
         '--capture-range-end=stop','--force-overwrite=true','--output='+str(stem),
         sys.executable,str(ROOT/'worker.py'),'--task',str(folder/'task.json')]
    write_json(folder/'command.json',cmd)
    try:
        run(cmd,folder/'nsys.log',args.timeout)
        if not json.loads((folder/'worker.json').read_text()).get('valid'):raise RuntimeError('Trace worker invalid')
        run([nsys,'export','--type=sqlite','--force-overwrite=true','--output='+str(stem)+'.sqlite',str(stem)+'.nsys-rep'],folder/'export.log',180)
        result=nsys_summary(str(stem)+'.sqlite',3);result['valid']=True
    except Exception:result=dict(valid=False,error=traceback.format_exc())
    write_json(folder/'collection.json',result);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--track',choices=['A','B'],default='B');p.add_argument('--plan',choices=['smoke','main','extended'],default='main')
    p.add_argument('--cache',type=Path,default=ROOT/'cache');p.add_argument('--output',type=Path,default=ROOT/'results_B_main')
    p.add_argument('--reuse-project',type=Path,default=Path('/content/attention_real_llm'));p.add_argument('--ncu',default=shutil.which('ncu'))
    p.add_argument('--timeout',type=int,default=1800);p.add_argument('--skip-nsys',action='store_true');p.add_argument('--skip-latency',action='store_true')
    p.add_argument('--extra-torch',action='store_true');p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    a.output=a.output.resolve();a.cache=a.cache.resolve();out=a.output;out.mkdir(parents=True,exist_ok=True)
    verify_assets();plan=build_plan(a.track,a.plan);write_json(out/'plan.json',plan)
    os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR',str(a.cache/'inductor'))
    os.environ['NSYS_NVTX_PROFILER_REGISTER_ONLY']='0'
    import torch
    if not torch.cuda.is_available():raise RuntimeError('Select a CUDA GPU runtime')
    arch='sm_%d%d'%torch.cuda.get_device_capability()
    environment=dict(torch=torch.__version__,cuda=torch.version.cuda,python=sys.version,gpu=torch.cuda.get_device_name(),arch=arch,
      iree=capture(['iree-compile','--version']),ncu=capture([a.ncu,'--version']) if a.ncu else 'not found',
      nvidia_smi=capture(['nvidia-smi']),nsys=capture(['nsys','--version']),tf32=False,model_precision='float32',clock_control='none',
      assets_manifest_sha256=sha(ROOT/'assets_sha256.json'),gpu_identity=capture(['nvidia-smi','--query-gpu=uuid,driver_version','--format=csv,noheader']))
    write_json(out/'environment.json',environment)
    if not a.ncu and not a.prepare_only:raise RuntimeError('ncu executable not found')
    from workloads import prepare_cases,compile_module
    if not a.prepare_only:
        helptext=capture([a.ncu,'--help'])
        if '--print-units' not in helptext:raise RuntimeError('Nsight lacks required --print-units export option; send ncu --help output')
        # This cheap control must precede model download, reference and builds.
        control=next(c for c in plan['cases'] if c['kind']=='control')
        prepare_cases([control],a.cache)
        smoke=next(j for j in plan['profiles'] if j['case']==control['id'])
        r=profile(smoke,control,a,out,environment)
        if not r['valid']:raise RuntimeError('Range counter control/export failed; inspect '+str(out/'profiles'))
    prepare_cases(plan['cases'],a.cache,a.reuse_project)
    built={}
    for case in plan['cases']:
        for backend in backends(case):
            if backend.startswith('iree_'):
                print('BUILD',case['id'],backend,flush=True)
                vm=compile_module(case,backend,a.cache,arch)
                build=vm.parent/'build.json';family=vm.parent.name
                built[family]=json.loads(build.read_text())
                write_json(out/'compilation.json',built)
    if a.prepare_only:return
    cases={c['id']:c for c in plan['cases']}
    # A range counter-control and real IREE window must work before a long sweep.
    target=next(j for j in plan['profiles'] if j['backend']=='iree_stable' and cases[j['case']]['kind']=='model')
    r=profile(target,cases[target['case']],a,out,environment)
    if not r['valid']:
        diagnostic={**target,'replay':'application','calls':1}
        profile(diagnostic,cases[target['case']],a,out,environment)
        raise RuntimeError('IREE range preflight failed; kernel diagnostic attempted, no whole-step traffic claim. Send the partial archive with ncu.log and worker.json')
    if not a.skip_latency:
        for case in plan['cases']:
            if case['kind']=='control':continue
            folder=out/'latency'/case['id'];folder.mkdir(parents=True,exist_ok=True);casepath=folder/'case.json';write_json(casepath,case)
            contract=dict(case=case,fixture=sha(a.cache/'data'/case['id']/'fixture.json'),builds=built,
                          environment={k:environment[k] for k in ['torch','cuda','gpu_identity','iree','assets_manifest_sha256']},
                          plan=a.plan,extra_torch=a.extra_torch)
            contract_path=folder/'latency_contract.json'
            if (folder/'latency.json').exists() and contract_path.exists() and json.loads(contract_path.read_text())==contract:continue
            write_json(contract_path,contract)
            cmd=[sys.executable,str(ROOT/'latency.py'),'--case',str(casepath),'--cache',str(a.cache),'--output',str(folder),
                 '--trials','11' if a.plan=='smoke' else '31','--repetitions','1' if a.plan=='smoke' else '3']
            if a.extra_torch:cmd+=['--extra-torch']
            try:run(cmd,folder/'latency.log',a.timeout)
            except Exception:write_json(folder/'error.json',dict(error=traceback.format_exc()))
            print('LATENCY_SAVED',case['id'],flush=True)
    jobs=list(plan['profiles']);random.Random(6886).shuffle(jobs);durations=[]
    for i,job in enumerate(jobs,1):
        t=time.perf_counter();result=profile(job,cases[job['case']],a,out,environment);durations.append(time.perf_counter()-t)
        remaining=statistics_median(durations[-8:])*(len(jobs)-i)/60
        print(f"PROFILE {i}/{len(jobs)} {job['case']} {job['backend']} {job['replay']} valid={result['valid']} recent_rate_remaining_min={remaining:.1f}",flush=True)
        from analyze import summarize
        summarize(out)
    traces=[]
    if not a.skip_nsys:
        for case in plan['cases']:
            if case['id'] not in ['cache_large',f'{a.track}_n2048_p0']:continue
            for backend in backends(case):
                print('NSYS',case['id'],backend,flush=True);traces.append(dict(case=case['id'],backend=backend,result=nsys_profile(case,backend,a,out)))
    write_json(out/'nsys_status.json',traces)
    from analyze import summarize
    result=summarize(out)
    try:
        from plot_results import plot
        plot(out)
    except Exception:write_json(out/'plot_status.json',dict(error=traceback.format_exc(),note='Raw results remain available'))
    write_json(out/'completion.json',dict(completed=True,profile_failures=len(result['failed_profiles']),
        counter_groups=len(result['counter_summaries']),note='Read failures and backend eligibility before claiming performance'))
    print('FINISHED',out,flush=True)

def statistics_median(seq):
    import statistics
    return statistics.median(seq)

if __name__=='__main__':main()
