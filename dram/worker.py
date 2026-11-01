"""One isolated profiler invocation. No initialization, input upload or validation in NVTX range."""
import argparse, contextlib, gc, json, os, sys, time, traceback
from pathlib import Path
import numpy as np
from common import RANGE, write_json
from workloads import Runner, CudaContext, configure_torch, load_fixture

def validate_modes(runner):
    reports={'natural':runner.validate(runner.fn())}
    if runner.case.get('set')=='B':
        saved=(runner.inputs[1],runner.host,runner.refs)
        try:
            for winner in ['0','1']:
                host,refs=load_fixture(runner.cache,runner.case,winner)
                if runner.backend.startswith('iree_'):
                    import iree.runtime as rt
                    runner.inputs[1]=rt.asdevicearray(runner.cfg.device,host[1].copy())
                else:
                    import torch
                    runner.inputs[1]=torch.from_numpy(host[1].copy()).to(runner.device)
                runner.host=host;runner.refs=refs
                reports['winner_'+winner]=runner.validate(runner.fn())
        finally:runner.inputs[1],runner.host,runner.refs=saved
    if not all(x['pass_'] for x in reports.values()):raise RuntimeError('Correctness gate failed: '+json.dumps(reports))
    if not runner.inputs_unchanged():raise RuntimeError('Input mutation during correctness checks')
    return reports

def execute(task):
    torch=configure_torch()
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; this worker cannot collect CPU DRAM counters')
    result=dict(task=task,completed=False,valid=False,torch=torch.__version__,cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(),capability=list(torch.cuda.get_device_capability()),
                scope='Independent immutable-input inference calls; profiler wall time is not latency')
    out=Path(task['result']);write_json(out,result)
    try:
        with torch.inference_mode():
            t=time.perf_counter();runner=Runner(task['case'],task['backend'],task['cache_dir']);result['load_s']=time.perf_counter()-t
            result['before']=validate_modes(runner)
            context=CudaContext(runner.anchor())
            for _ in range(task.get('warmup',5)):hold=runner.fn()
            with context:context.sync()
            hold=None;gc.collect()
            if task['cache']=='evicted':
                # Conditioning experiment, not a claim of a perfect cold GPU.
                evict=torch.randn(128*1024*1024//4,device='cuda');evict.add_(.01);torch.cuda.synchronize()
            result['context_pointer']=int(context.ctx.value)
            def allocator_stats():
                if not hasattr(runner,'cfg'):return None
                try:return runner.cfg.device.allocator.statistics
                except Exception as e:return dict(unavailable=str(e))
            result['allocator_before']=allocator_stats()
            # The owning CUDA context is current and retained across the whole
            # process-wide start/end NVTX range, including synchronization.
            with context:
                context.sync();mark=torch.cuda.nvtx.range_start(RANGE)
                try:
                    for _ in range(task['calls']):hold=runner.fn()
                    context.sync()
                finally:torch.cuda.nvtx.range_end(mark)
            result['allocator_after']=allocator_stats()
            result['after']=runner.validate(hold)
            result['input_unchanged']=runner.inputs_unchanged()
            result['valid']=result['after']['pass_'] and result['input_unchanged']
            result['completed']=True
            if not result['valid']:raise RuntimeError('Post-profile correctness or immutability failure')
    except Exception:
        result['error']=traceback.format_exc();write_json(out,result);raise
    write_json(out,result)
    print('PROFILE_WINDOW_OK',task['case']['id'],task['backend'],task['cache'],task['calls'],flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',type=Path,required=True);a=p.parse_args()
    execute(json.loads(a.task.read_text()))
