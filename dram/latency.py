"""Separate unprofiled synchronized host latency, with forced-winner eligibility for every backend."""
import argparse, gc, json, random, statistics, subprocess, time, traceback
from pathlib import Path
import numpy as np
from common import backends, write_json
from workloads import Runner,configure_torch,CudaContext
from worker import validate_modes

def timed_invoke(runner, context):
    # Keep the output alive until completion; release the previous output in
    # the caller before starting this timer, rather than during assignment.
    start=time.perf_counter_ns(); result=runner.fn()
    with context:context.sync()
    return (time.perf_counter_ns()-start)/1e6, result

def measure(case,cache,out,trials=31,reps=3,extras=False,warmup_calls=5):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);torch=configure_torch();runners={};contexts={};reports={};errors={};rows=[]
    result=dict(warmup_calls=warmup_calls,output_release_timing='before timer',case=case,completed=False,valid=False,scope='Resident outputs; synchronized host call; setup and compilation excluded',telemetry=[])
    def telemetry():
        try:return subprocess.check_output(['nvidia-smi','--query-gpu=timestamp,name,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw','--format=csv,noheader'],text=True,timeout=10).strip()
        except Exception:return 'unavailable'
    with torch.inference_mode():
        for name in backends(case,extras):
            print('LATENCY_VALIDATE',case['id'],name,flush=True)
            try:
                start=time.perf_counter();runner=Runner(case,name,cache);before=validate_modes(runner)
                reports[name]=dict(before=before,setup_and_first_validation_s=time.perf_counter()-start)
                for _ in range(warmup_calls):hold=runner.fn()
                context=CudaContext(runner.anchor())
                with context:context.sync()
                runners[name]=runner;contexts[name]=context
            except Exception:errors[name]=traceback.format_exc()
        if not runners:raise RuntimeError('No numerically valid backends')
        for rep in range(reps):
            result['telemetry'].append(dict(repetition=rep,before=telemetry()))
            samples={name:[] for name in runners};rng=random.Random(6886+rep)
            for _ in range(trials):
                order=list(runners);rng.shuffle(order)
                for name in order:
                    hold=None  # buffer release is outside the measured window
                    with contexts[name]:contexts[name].sync()
                    elapsed,hold=timed_invoke(runners[name],contexts[name])
                    samples[name].append(elapsed)
            for name,values in samples.items():
                rows.append(dict(backend=name,repetition=rep,median_ms=statistics.median(values),p95_ms=float(np.percentile(values,95)),samples_ms=values))
            write_json(out/'latency_partial.json',dict(case=case,rows=rows,errors=errors))
            result['telemetry'][-1]['after']=telemetry()
        for name,runner in runners.items():
            reports[name]['after']=runner.validate(runner.fn());reports[name]['input_unchanged']=runner.inputs_unchanged()
            reports[name]['valid']=reports[name]['after']['pass_'] and reports[name]['input_unchanged']
        for row in rows:row['valid']=reports[row['backend']]['valid']
    result.update(completed=True,rows=rows,validation=reports,errors=errors,valid=all(x['valid'] for x in reports.values()),
                  missing_backends=sorted(set(backends(case,extras))-set(runners)))
    write_json(out/'latency.json',result)
    print('LATENCY_DONE',case['id'],{name:round(statistics.median(x['median_ms'] for x in rows if x['backend']==name),3) for name in runners},flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--trials',type=int,default=31);p.add_argument('--repetitions',type=int,default=3);p.add_argument('--extra-torch',action='store_true');p.add_argument('--warmup-calls',type=int,default=5);a=p.parse_args()
    measure(json.loads(a.case.read_text()),a.cache,a.output,a.trials,a.repetitions,a.extra_torch,a.warmup_calls)
