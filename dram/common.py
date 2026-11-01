"""Pure Python experiment planning, strict counter parsing and data checks."""
from pathlib import Path
import csv, io, json, hashlib, math, statistics

ROOT = Path(__file__).resolve().parent
RANGE = 'ATTENTION_MEASURED_STEP'
METRICS = ['dram__bytes_read.sum', 'dram__bytes_write.sum']
EXTRA_METRICS = ['lts__t_sectors_op_read.sum', 'lts__t_sectors_op_write.sum']
STRONG = ['--iree-opt-level=O3', '--iree-opt-generalize-matmul',
          '--iree-dispatch-creation-enable-aggressive-reshape-movement']
PRECISION = ['--iree-input-demote-f64-to-f32=false',
             '--iree-opt-numeric-precision-reduction=false']
CUDA_SAFE = ['--iree-dispatch-creation-enable-split-reduction',
             '--iree-dispatch-creation-split-argmax-threshold=1024',
             '--iree-dispatch-creation-split-argmax-tile-size=256']
MODEL_BACKENDS = ['iree_stable', 'iree_commit']

def write_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, default=str)); tmp.replace(path)

def sha(path):
    with open(path, 'rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()

def verify_assets():
    for name, digest in json.loads((ROOT/'assets_sha256.json').read_text()).items():
        if sha(ROOT/name) != digest: raise RuntimeError('Asset mismatch: '+name)

def backends(case, extras=False):
    if case['kind']=='control': return ['torch_stream']
    if case['kind']=='synthetic': return ['iree_stable', 'iree_forwarded']
    values = ['iree_stable'] + (['iree_commit'] if case['set']=='B' else [])
    if extras: values += ['torch_eager', 'torch_sdpa']
    return values

def model_source(which, backend):
    base=ROOT/'assets/model/prepared'
    if backend=='iree_commit':
        if which!='B': raise ValueError('No commit rewrite in ordinary decode')
        return base/'B_stable_commit.mlir'
    if backend=='iree_stable': return base/(which+'_stable.mlir')
    raise ValueError(backend)

def source_for(case, backend):
    if case['kind']=='model': return model_source(case['set'], backend)
    if case['kind']=='synthetic':
        return ROOT/'assets/synthetic'/case['id']/('forwarded.mlir' if backend=='iree_forwarded' else 'baseline.mlir')
    raise ValueError(case)

def build_plan(track='B', plan='main'):
    if track not in ['A','B'] or plan not in ['smoke','main','extended']: raise ValueError('Unknown plan')
    cases=[dict(id='stream_control',kind='control',size_mib=64)]
    labels=['cache_large'] if plan=='smoke' else ['cache_small','cache_large','cache_dense']
    for label in labels: cases.append(dict(id=label,kind='synthetic'))
    contexts=[2048] if plan=='smoke' else [128,512,2048,3072,4096] if plan=='extended' else [2048,3072,4096]
    prompts=[0] if plan=='smoke' else [0,1,2]
    for c in contexts:
        for prompt in prompts:
            cases.append(dict(id=f'{track}_n{c}_p{prompt}',kind='model',set=track,context=c,prompt=prompt))
    profiles=[]
    for case in cases:
        # Main latency/correctness uses all prompts; counters initially use prompt 0.
        if case['kind']=='model' and case['prompt']!=0 and plan!='extended': continue
        reps=1 if plan=='smoke' else 3
        for backend in backends(case):
            for rep in range(reps):
                profiles.append(dict(case=case['id'],backend=backend,rep=rep,cache='warm',calls=3,replay='app-range'))
        if plan=='extended' and case['kind']!='control' and (case['kind']=='synthetic' or case['prompt']==0):
            for backend in backends(case):
                profiles.append(dict(case=case['id'],backend=backend,rep=0,cache='evicted',calls=1,replay='app-range'))
        if plan!='smoke' and case['id'] in ['cache_large',f'{track}_n2048_p0']:
            for backend in [b for b in backends(case) if b.startswith('iree_')]:
                profiles.append(dict(case=case['id'],backend=backend,rep=0,cache='warm',calls=1,replay='application'))
    return dict(version=2,track=track,plan=plan,cases=cases,profiles=profiles,
                note='Fixed immutable inputs; repeated one-step calls, not an autoregressive generation loop')

def parse_number(value):
    value=value.strip().replace(',','')
    if not value or value.lower() in ['n/a','nan','inf','-inf','not available']: raise ValueError('Missing metric value')
    x=float(value)
    if not math.isfinite(x) or x<0: raise ValueError('Invalid metric value')
    return x

def metric_bytes(value, unit):
    unit=unit.strip()
    scales={'byte':1,'bytes':1,'Kbyte':1000,'Mbyte':10**6,'Gbyte':10**9,
            'KByte':1000,'MByte':10**6,'GByte':10**9}
    if unit not in scales: raise ValueError('Unexpected byte unit: '+unit)
    return parse_number(value)*scales[unit]

def parse_ncu(path, calls=1, replay='app-range'):
    """Read both long metric rows and wide raw CSV with its separate units row.

    Nsight raw CSV can put each metric in a column, while details CSV puts
    each metric in a row. Missing units and invalid values remain errors.
    """
    lines=Path(path).read_text(errors='replace').splitlines()
    if any(s.lstrip().startswith('==ERROR==') for s in lines):raise ValueError('Nsight export reported an error')
    start=None;layout=None
    for i,line in enumerate(lines):
        fields=next(csv.reader([line]),[])
        if {'Metric Name','Metric Value','Metric Unit'}.issubset(fields):start=i;layout='long';break
        if 'ID' in fields and any(m in fields for m in METRICS):start=i;layout='wide';break
    if start is None: raise ValueError('No Nsight raw CSV header; profile is empty or failed')
    reader=csv.DictReader(io.StringIO('\n'.join(lines[start:])))
    values={k:{} for k in METRICS}; extras={};units={}
    def consume(row,metric,unit,value):
        action=row.get('ID','') or ''
        if not action.strip():raise ValueError('Missing action ID')
        key=(row.get('Process ID',''),row.get('Context',''),row.get('Stream',''),action)
        if metric in METRICS:
            if key in values[metric]: raise ValueError('Duplicate metric/action')
            values[metric][key]=metric_bytes(value,unit)
        else:
            records=extras.setdefault(metric,{})
            if key in records:raise ValueError('Duplicate extra metric/action')
            records[key]=parse_number(value)
    for row in reader:
        if layout=='long':
            metric=row.get('Metric Name','')
            if metric in METRICS+EXTRA_METRICS:consume(row,metric,row.get('Metric Unit',''),row.get('Metric Value',''))
        else:
            present=[m for m in METRICS+EXTRA_METRICS if m in (reader.fieldnames or [])]
            if not (row.get('ID','') or '').strip():
                # The units row has empty metadata fields. Never treat it as
                # a launch or silently assume units for an absent units row.
                if units:raise ValueError('Unexpected extra units or metadata row')
                units={m:(row.get(m,'') or '').strip() for m in present}
                if not all(units.get(m) for m in METRICS):raise ValueError('Missing wide CSV units')
                continue
            if not units:raise ValueError('Wide CSV units row absent')
            for metric in present:consume(row,metric,units.get(metric,''),row.get(metric,''))
    sets=[set(v) for v in values.values()]
    if not sets[0] or sets[0]!=sets[1]: raise ValueError('Missing or unmatched read/write actions')
    if replay=='app-range' and len(sets[0])!=1: raise ValueError('Expected exactly one marked range')
    if calls<1: raise ValueError('calls must be positive')
    reads=sum(values[METRICS[0]].values())/calls;writes=sum(values[METRICS[1]].values())/calls
    return dict(read_bytes_per_call=reads,write_bytes_per_call=writes,total_bytes_per_call=reads+writes,
                profiled_actions=len(sets[0]),replay=replay,calls=calls,
                l2_requested_sectors={k:sum(v.values())/calls for k,v in extras.items()},csv_format=layout,
                scope='range hardware DRAM counters' if replay=='app-range' else 'sum of profiled kernel DRAM counters; excludes copy-only intervals')

def ncu_export_command(ncu,report):
    # --units is an obsolete spelling rejected by the user's Nsight 2025.3.1.
    # Raw CSV already contains metric names. --print-metric-name is valid
    # only for the details page and is rejected for raw by Nsight 2025.3.1.
    return [str(ncu),'--import',str(report),'--page','raw','--csv',
            '--print-units','base']

def ncu_command(ncu, worker, task, outfile, python, replay='app-range', extra=False):
    cmd=[ncu,'--clock-control','none','--cache-control','none','--replay-mode',replay,
         '--nvtx','--nvtx-include',RANGE,'--target-processes','application-only',
         '--metrics',','.join(METRICS+(EXTRA_METRICS if extra else [])),
         '--force-overwrite','-o',str(outfile),str(python),str(worker),'--task',str(task)]
    if replay=='application': cmd[1:1]=['--app-replay-mode','strict']
    return cmd

def check_arrays(actual, host, refs, case):
    import numpy as np
    if not isinstance(actual,(list,tuple)): actual=(actual,)
    def array(x):
        if hasattr(x,'detach'): return x.detach().cpu().numpy()
        if hasattr(x,'to_host'): return np.asarray(x.to_host())
        return np.asarray(x)
    expected=len(refs); rows=[];digest=hashlib.sha256()
    if len(actual)!=expected: return dict(pass_=False,error='output count mismatch')
    offset=(1 if case.get('set')=='A' else 2) if case['kind']=='model' else expected
    for i,got in enumerate(actual):
        g=array(got);ref=np.asarray(refs['out'+str(i)]);prefix=True
        digest.update(str(g.dtype).encode());digest.update(str(g.shape).encode());digest.update(np.ascontiguousarray(g).tobytes())
        if case['kind']=='model' and i>=offset:
            index=(3 if case['set']=='A' else 4)+i-offset
            old=host[index]
            prefix=g.shape==(*old.shape[:2],old.shape[2]+1,old.shape[3]) and np.array_equal(g[:,:,:-1,:],old)
            measured=g[:,:,-1:,:] if g.ndim==4 else g
        else: measured=g
        if measured.shape!=ref.shape: rows.append(dict(output=i,pass_=False,error='shape mismatch'));continue
        if ref.dtype.kind not in 'fc': ok=bool(np.array_equal(measured,ref));error=0.
        else:
            diff=np.abs(measured.astype('float64')-ref.astype('float64'));tol=(5e-4+3e-3*np.abs(ref)) if case['kind']=='model' else (5e-5+2e-3*np.abs(ref))
            ok=bool(np.all(np.isfinite(diff)&(diff<=tol)));error=float(np.max(diff))
        rows.append(dict(output=i,pass_=ok and bool(prefix),prefix_exact=bool(prefix),max_abs=error))
    return dict(pass_=all(r['pass_'] for r in rows),outputs=rows,sha256=digest.hexdigest())
