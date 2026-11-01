"""Recompute final GPU aggregates from retained raw CSVs and timing samples."""
from pathlib import Path
import argparse,csv,json,statistics,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'dram'))
from common import parse_ncu
from analyze import summarize

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'reproduced');args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True);aggregate=[];checks=[]
    for track,run in [('A','20261101_074854_B_main'),('B','20261101_081149_B_main')]:
        root=ROOT/'results/dram'/track/run/'main'
        assert json.loads((root/'plan.json').read_text())['track']==track
        assert json.loads((root/'completion.json').read_text())['completed']
        for p in sorted((root/'profiles').glob('*/collection.json')):
            record=json.loads(p.read_text())
            if not record.get('valid'):raise RuntimeError('Invalid final profile: '+str(p))
            task=record['task'];csvpath=p.with_name('metrics.csv')
            # Large per-kernel diagnostic CSVs are restored from the evidence archive.
            if not csvpath.exists():raise RuntimeError('Run scripts/unpack_artifacts.py first: '+str(csvpath))
            measured=parse_ncu(csvpath,task['calls'],task['replay'])
            for key in ['read_bytes_per_call','write_bytes_per_call','total_bytes_per_call']:
                if abs(measured[key]-record['metrics'][key])>max(1e-6,abs(measured[key])*1e-10):raise RuntimeError('Counter mismatch: '+str(p))
            checks.append(dict(track=track,profile=p.parent.name,valid=True))
        result=summarize(root)
        for context in [2048,3072,4096]:
            row=dict(track=track,context=context)
            for backend in ['iree_stable','iree_commit','torch_compile']:
                samples=[r['median_ms'] for r in result['valid_latency_rows'] if r['backend']==backend and r['case'].startswith(f'{track}_n{context}_')]
                if samples:row[backend+'_ms']=statistics.median(samples);row[backend+'_row_count']=len(samples)
            if 'iree_commit_ms' in row:
                row['speedup']=row['iree_stable_ms']/row['iree_commit_ms'];row['latency_reduction_percent']=100*(1-1/row['speedup'])
                pair=next(r for r in result['matched_iree_comparisons'] if r['case']==f'B_n{context}_p0' and r['cache']=='warm' and r['replay']=='app-range')
                row.update({k:pair[k] for k in ['baseline_dram_bytes','optimized_dram_bytes','dram_reduction_percent']})
            aggregate.append(row)
    (args.output/'model_results.json').write_text(json.dumps(aggregate,indent=2)+'\n')
    (args.output/'counter_verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    fields=list(dict.fromkeys(k for row in aggregate for k in row))
    with (args.output/'model_results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(aggregate)
    print('PASS independently recomputed',len(checks),'GPU profiles; aggregates:',args.output/'model_results.csv')
if __name__=='__main__':main()
