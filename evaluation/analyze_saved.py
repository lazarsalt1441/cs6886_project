"""Reanalyse paired IREE timings; no GPU execution and no cross-session CI claim."""
import argparse, hashlib, json, random, statistics, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results/dram/B/20261101_081149_B_main/main'

def percentile(values, q):
    x=sorted(values); k=(len(x)-1)*q; i=int(k)
    return x[i] if i==len(x)-1 else x[i]+(x[i+1]-x[i])*(k-i)

def summarize(groups, draws=10000):
    """Resample prompts, then paired repetition medians within prompt.

    The estimand is ratio of pooled medians, matching the report headline.
    Groups are prompts; tuples are baseline/optimized repetition medians.
    Timed invocations are not treated as independent experimental runs.
    """
    assert groups and all(g for g in groups)
    pairs=[p for g in groups for p in g]
    ratio=lambda ps:statistics.median(p[0] for p in ps)/statistics.median(p[1] for p in ps)
    rng=random.Random(6886); boot=[]
    for _ in range(draws):
        sample=[]
        for _ in groups:
            g=rng.choice(groups); sample.extend(rng.choice(g) for _ in g)
        boot.append(ratio(sample))
    ratios=[a/b for a,b in pairs]
    return dict(baseline_ms=statistics.median(p[0] for p in pairs),optimized_ms=statistics.median(p[1] for p in pairs),
                speedup=ratio(pairs),within_session_bootstrap_95=[percentile(boot,.025),percentile(boot,.975)],
                paired_repetition_speedups=ratios,paired_range=[min(ratios),max(ratios)],
                prompts=len(groups),paired_repetitions=len(pairs))

def paired(path,opt='iree_commit'):
    x=json.loads(path.read_text()); d={(r['backend'],r['repetition']):r for r in x['rows'] if r.get('valid')}
    return [(d['iree_stable',i]['median_ms'],d[opt,i]['median_ms']) for i in sorted({r['repetition'] for r in x['rows']})
            if ('iree_stable',i) in d and (opt,i) in d]

def audit(cache,iree_opt):
    outputs=[subprocess.check_output([iree_opt,str(ROOT/'dram/assets/synthetic/cache_dense'/n),'--mlir-print-op-generic']) for n in ['baseline.mlir','forwarded.mlir']]
    result=dict(canonical_input_equal=outputs[0]==outputs[1],canonical_input_sha256=hashlib.sha256(outputs[0]).hexdigest(),
                note='Equal canonical input is not by itself proof of equal runtime behavior.')
    directories=[cache/n/'codegen' for n in ['cache_dense_iree_stable','cache_dense_iree_forwarded']]
    files=sorted(p.name for p in directories[0].iterdir() if p.suffix in ['.ptx','.ll','.mlir'])
    assert files
    comparisons=[]
    for n in files:
        a=(directories[0]/n).read_bytes(); b=(directories[1]/n).read_bytes()
        comparisons.append(dict(file=n,equal=a==b,sha256=hashlib.sha256(a).hexdigest()))
    result['codegen_files']=comparisons;result['all_codegen_equal']=all(r['equal'] for r in comparisons)
    return result

def plot(data,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    rows=data['model']; x=list(range(len(rows)));vals=[r['speedup'] for r in rows]
    axes[0].errorbar(x,vals,yerr=[[v-r['within_session_bootstrap_95'][0] for v,r in zip(vals,rows)],
                                 [r['within_session_bootstrap_95'][1]-v for v,r in zip(vals,rows)]],fmt='o',capsize=5,color='#176b70')
    for j,r in enumerate(rows):axes[0].scatter([j]*len(r['paired_repetition_speedups']),r['paired_repetition_speedups'],s=12,color='#999999',alpha=.55)
    axes[0].set_xticks(x,[r['context'] for r in rows]);axes[0].set_xlabel('Cached tokens');axes[0].set_ylabel('Baseline / optimized latency')
    axes[0].set_title('Model: dots = nine paired repeats\nBars = within-session hierarchical 95% bootstrap')
    for j,r in enumerate(data['dense_controls']):
        axes[1].scatter([j]*len(r['paired_repetition_speedups']),r['paired_repetition_speedups'],s=35)
    axes[1].set_xticks([0,1],['Session A','Session B']);axes[1].set_yscale('log');axes[1].set_ylabel('Paired no-match latency ratio (log scale)')
    axes[1].set_title('Dense control: direction flips across repeats\nNo causal optimization gain inferred')
    for ax in axes:ax.axhline(1,color='black',lw=.7);ax.grid(axis='y',alpha=.2)
    fig.savefig(out/'latency_uncertainty.png',dpi=180);plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'evaluation/results');p.add_argument('--compiled-cache',type=Path);p.add_argument('--iree-opt',default='iree-opt');p.add_argument('--plot',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    data=dict(scope='One recorded GPU session, three fixed prompts; intervals exclude cross-session, compiler and device variability. No new context measurements.',model=[],dense_controls=[])
    for n in [2048,3072,4096]:
        row=summarize([paired(RUN/f'latency/B_n{n}_p{i}/latency.json') for i in range(3)]);row['context']=n
        stable=[]
        for i in range(3):stable.extend(paired(RUN/f'latency/B_n{n}_p{i}/latency.json')[1:])
        row['repetitions_1_and_2_only_speedup']=statistics.median(v[0] for v in stable)/statistics.median(v[1] for v in stable)
        data['model'].append(row)
    for track,stamp in [('A','20261101_074854_B_main'),('B','20261101_081149_B_main')]:
        row=summarize([paired(ROOT/f'results/dram/{track}/{stamp}/main/latency/cache_dense/latency.json','iree_forwarded')]);row['track']=track;data['dense_controls'].append(row)
    if a.compiled_cache:data['dense_compilation_audit']=audit(a.compiled_cache,a.iree_opt)
    data['conclusions']=['2048 improvement appears in all nine paired repetition medians and persists when repetition zero is excluded. This does not explain the kernel-level cause.',
        'Only 3.91% less whole-step DRAM at 2048 accompanies 26.46% less latency; bytes alone cannot explain the gap.',
        'Dense control changes direction across repetitions. Equal canonical IR/codegen narrows the cause to runtime/measurement or other unexamined effects; no pass regression mechanism established.',
        'Additional contexts and same-binary placebo tests require a new GPU run.']
    (a.output/'analysis.json').write_text(json.dumps(data,indent=2)+'\n')
    lines=['# Evaluation reanalysis','','No new GPU execution. Intervals describe the recorded session only.','','| Context | Speedup | Within-session 95% interval | Paired-repeat range | Without first repetition |','|---:|---:|---:|---:|---:|']
    for r in data['model']:lines.append(f"| {r['context']} | {r['speedup']:.3f}x | {r['within_session_bootstrap_95'][0]:.3f}–{r['within_session_bootstrap_95'][1]:.3f} | {r['paired_range'][0]:.3f}–{r['paired_range'][1]:.3f} | {r['repetitions_1_and_2_only_speedup']:.3f}x |")
    lines+=['','Bootstrap: 10,000 draws (seed 6886), resample the three prompts, then paired repetition medians within each selected prompt. The statistic is baseline median / optimized median. The 31 calls within a repetition are not independent replications. Three fixed prompts and one session cannot support an across-session confidence claim.','']+['* '+v for v in data['conclusions']]
    for r in data['dense_controls']:lines.append(f"* Dense session {r['track']}: paired ratios {', '.join(f'{v:.3f}' for v in r['paired_repetition_speedups'])}.")
    if 'dense_compilation_audit' in data:lines+=['',f"Canonical dense input equal: {data['dense_compilation_audit']['canonical_input_equal']}; all retained fresh codegen files equal: {data['dense_compilation_audit']['all_codegen_equal']}. Full VMFB hashes differ, so bitwise whole-executable equivalence is not asserted."]
    (a.output/'ANALYSIS.md').write_text('\n'.join(lines)+'\n')
    if a.plot:plot(data,a.output)
    print('PASS reanalysed model and dense controls; results:',a.output)
if __name__=='__main__':main()
