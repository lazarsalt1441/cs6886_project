"""Counter and unprofiled latency summaries; missing data is never treated as zero."""
import argparse, csv, json, sqlite3, statistics
from pathlib import Path
from common import write_json

def nsys_summary(path,calls):
    db=sqlite3.connect(path);tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    result=dict(calls=calls,scope='CUDA memcpy logical payload bytes; do not add to hardware DRAM counters')
    if 'CUPTI_ACTIVITY_KIND_MEMCPY' in tables:
        rows=db.execute('SELECT copyKind, COUNT(*), SUM(bytes) FROM CUPTI_ACTIVITY_KIND_MEMCPY GROUP BY copyKind').fetchall()
        result['memcpy_by_kind']=[dict(copy_kind=int(k),count=int(n),logical_bytes_per_call=float(b)/calls) for k,n,b in rows]
        result['logical_memcpy_bytes_per_call']=sum(float(x[2]) for x in rows)/calls
    else:result['memcpy_table']='absent; no logical-copy total asserted'
    if 'CUPTI_ACTIVITY_KIND_MEMSET' in tables:
        columns={r[1] for r in db.execute('PRAGMA table_info(CUPTI_ACTIVITY_KIND_MEMSET)')}
        if 'bytes' in columns:
            count,payload=db.execute('SELECT COUNT(*), SUM(bytes) FROM CUPTI_ACTIVITY_KIND_MEMSET').fetchone()
            result['memset_count']=count;result['logical_memset_bytes_per_call']=float(payload or 0)/calls
    for name in ['CUPTI_ACTIVITY_KIND_KERNEL','CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL']:
        if name in tables:
            result['kernel_count_per_call']=db.execute('SELECT COUNT(*) FROM '+name).fetchone()[0]/calls;break
    db.close();return result

def summarize(out):
    out=Path(out);profiles=[];failures=[];latencies=[]
    for path in sorted((out/'profiles').glob('*/collection.json')):
        row=json.loads(path.read_text());(profiles if row.get('valid') else failures).append(row)
    for path in sorted((out/'latency').glob('*/latency.json')):
        r=json.loads(path.read_text())
        for row in r.get('rows',[]):
            if row.get('valid'):latencies.append(dict(case=r['case']['id'],**row))
    grouped={}
    for r in profiles:
        t=r['task'];key=(t['case']['id'],t['backend'],t['cache'],t['replay'])
        grouped.setdefault(key,[]).append(r['metrics'])
    summary=[]
    for (case,backend,cache,replay),values in grouped.items():
        row=dict(case=case,backend=backend,cache=cache,replay=replay,profile_repetitions=len(values))
        for name in ['read_bytes_per_call','write_bytes_per_call','total_bytes_per_call']:
            seq=[x[name] for x in values];row[name]=statistics.median(seq);row[name+'_min']=min(seq);row[name+'_max']=max(seq)
        summary.append(row)
    comparisons=[]
    for opt in summary:
        if opt['backend'] not in ['iree_commit','iree_forwarded']:continue
        baseline=next((r for r in summary if r['case']==opt['case'] and r['backend']=='iree_stable' and r['cache']==opt['cache'] and r['replay']==opt['replay']),None)
        if baseline and baseline['total_bytes_per_call']>0:
            comparisons.append(dict(case=opt['case'],cache=opt['cache'],replay=opt['replay'],
              baseline_dram_bytes=baseline['total_bytes_per_call'],optimized_dram_bytes=opt['total_bytes_per_call'],
              dram_reduction_percent=100*(1-opt['total_bytes_per_call']/baseline['total_bytes_per_call'])))
    result=dict(counter_summaries=summary,matched_iree_comparisons=comparisons,failed_profiles=failures,
                valid_latency_rows=latencies,
                limitations=['Profiler data collection changes execution; latency is from separate unprofiled runs',
                'Range counters measure DRAM during the marked window, not all cache-line writebacks after it',
                'Application kernel sums and application-range totals have different coverage and are never pooled',
                'Repeated fixed-input steps do not measure sustained autoregressive generation',
                'No cross-session confidence interval is inferred from three same-session profiles'])
    write_json(out/'summary.json',result)
    if summary:
        with (out/'dram_summary.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    lines=['# Attention DRAM study results','',f'Valid profiles: {len(profiles)}; retained failed profiles: {len(failures)}.','',
           '| Case | Cache conditioning | Replay | Baseline MB | Optimized MB | DRAM reduction |',
           '|---|---|---|---:|---:|---:|']
    for r in comparisons:lines.append(f"| {r['case']} | {r['cache']} | {r['replay']} | {r['baseline_dram_bytes']/1e6:.3f} | {r['optimized_dram_bytes']/1e6:.3f} | {r['dram_reduction_percent']:.2f}% |")
    lines+=['','Only rows with completed before/after validation and unchanged inputs are included.',
            'A negative reduction is increased traffic. No speedup is inferred from bytes alone.',
            'See summary.json and latency/*/latency.json for unprofiled latency and all backend eligibility reports.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n');return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();r=summarize(a.output)
    print('Counter groups',len(r['counter_summaries']),'matched comparisons',len(r['matched_iree_comparisons']))
