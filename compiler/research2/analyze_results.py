"""Generate tables/plots and bootstrap intervals from saved actual observations."""
import json,csv,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('research2');OUT=ROOT/'results';FIG=OUT/'figures'
def interval(base,opt):
 rng=np.random.default_rng(618);b=np.array(base);o=np.array(opt);indices=rng.integers(0,len(b),(2000,len(b)));oi=rng.integers(0,len(o),(2000,len(o)));ratios=np.median(b[indices],axis=1)/np.median(o[oi],axis=1);return [float(x) for x in np.quantile(ratios,[.025,.975])]
def main():
 data=json.loads((OUT/'final-timings.json').read_text());groups=data['groups'];metadata={}
 for r in json.loads((OUT/'cache-study.json').read_text())['rows']:metadata[('attention',r['case'])]=r
 for r in json.loads((OUT/'block-study.json').read_text()):metadata[('block',r['case'])]=r
 for r in json.loads((OUT/'commit-study.json').read_text()):metadata[('commit',r['case'])]=r
 if (OUT/'winner-study.json').exists():r=json.loads((OUT/'winner-study.json').read_text());metadata[('winner',r['case'])]=r
 rows=[];counts={};prefix={'attention':'A','block':'B','commit':'C','winner':'W'}
 for r in groups:
  fam=r['family'];counts[fam]=counts.get(fam,0)+1;label=prefix[fam]+str(counts[fam]);med=r['holdout']['median_us'];samples=r['holdout']['samples_us'];best=r['best_baseline'];opt='factor-masked' if fam=='winner' else 'patch-forwarded';cond='factor-conditional' if fam=='winner' else 'forwarded';stats=metadata[(fam,r['case'])]['variants'];selected=r['selected'];lo,hi=interval(samples[best],samples[selected]);row={'id':label,'family':fam,'case':r['case'],'baseline':best,'selected':selected,'baseline_us':med[best],'selected_us':med[selected],'patch_us':med[opt],'conditional_us':med[cond],'eager_us':med['torch-eager'],'selected_speedup':r['selected_speedup'],'selected_speedup_ci95':[lo,hi],'patch_speedup':r['patch_speedup'],'baseline_allocation_bytes':stats[best]['allocated_bytes'],'patch_allocation_bytes':stats[opt]['allocated_bytes'],'selected_allocation_bytes':stats[selected]['allocated_bytes'],'baseline_scheduled_copy_bytes':sum(stats[best]['copy_command_bytes']),'patch_scheduled_copy_bytes':sum(stats[opt]['copy_command_bytes'])};rows.append(row)
 report={'method':data['method'],'cpu':data['cpu'],'interval_method':'2000 independent bootstrap resamples of21 invocation observations; conditional on this host/run, not uncertainty across devices or workloads. Baseline selected by minimum holdout median favors baseline.','rows':rows};(OUT/'summary.json').write_text(json.dumps(report,indent=2))
 with (OUT/'summary.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
 FIG.mkdir(parents=True,exist_ok=True);x=np.arange(len(rows));width=.25;fig,ax=plt.subplots(figsize=(12,4.2));ax.bar(x-width,[r['baseline_us'] for r in rows],width,label='Best correct IREE baseline');ax.bar(x,[r['patch_us'] for r in rows],width,label='Masked forwarding (+ factoring for W1)');ax.bar(x+width,[r['conditional_us'] for r in rows],width,label='Conditional forwarding (+ factoring for W1)');ax.set_yscale('log');ax.set_ylabel('Median CPU invocation latency (µs; log scale)');ax.set_xticks(x,[r['id'] for r in rows]);ax.legend(fontsize=8);ax.grid(axis='y',alpha=.25);fig.tight_layout();fig.savefig(FIG/'cache-latency.png',dpi=180);fig.savefig(FIG/'cache-latency.svg');plt.close(fig)
 fig,ax=plt.subplots(figsize=(12,3.8));ax.bar(x-.18,[r['baseline_allocation_bytes']/1024 for r in rows],.36,label='Best correct IREE baseline');ax.bar(x+.18,[r['patch_allocation_bytes']/1024 for r in rows],.36,label='Masked forwarding (+ factoring for W1)');ax.set_yscale('log');ax.set_ylabel('Invocation allocation (KiB; log scale)');ax.set_xticks(x,[r['id'] for r in rows]);ax.legend(fontsize=8);ax.grid(axis='y',alpha=.25);fig.tight_layout();fig.savefig(FIG/'cache-allocation.png',dpi=180);fig.savefig(FIG/'cache-allocation.svg');plt.close(fig)
 text='# Measured CPU results\n\nThese are synthetic attention/block experiments on one AMD EPYC9V74 CPU, one thread. GPU runtime remains untested. Compilation is excluded. Read README.md for methodology and limitations.\n\n|ID|Best baseline µs|Masked µs|Conditional µs|Calibration choice|Choice speedup (95% bootstrap)|\n|---|---:|---:|---:|---|---:|\n'
 for r in rows:text+=f"|{r['id']}|{r['baseline_us']:.2f}|{r['patch_us']:.2f}|{r['conditional_us']:.2f}|{r['selected']}|{r['selected_speedup']:.2f}× ({r['selected_speedup_ci95'][0]:.2f}–{r['selected_speedup_ci95'][1]:.2f})|\n"
 text+='\nA1–A9 are the nine attention shapes in study_cache.py; B1/B2 are two synthetic full transformer blocks; C1 returns one fixed committed cache and C2 all cache versions. W1 returns the attention-derived winner cache and decision. The masked ratio2 ablation includes regressions; the revised default ratio4 gate skips A5. Calibration choice is not a universal static heuristic. Near1× no-op differences are timing noise. Bootstrap intervals describe invocation variability in this run.\n\n|ID|Baseline allocation bytes|Masked allocation bytes|Baseline scheduled copies bytes|Masked scheduled copies bytes|\n|---|---:|---:|---:|---:|\n'
 for r in rows:text+=f"|{r['id']}|{r['baseline_allocation_bytes']}|{r['patch_allocation_bytes']}|{r['baseline_scheduled_copy_bytes']}|{r['patch_scheduled_copy_bytes']}|\n"
 text+='\nAllocation bytes include workspace and outputs after inputs are uploaded. Scheduled copy lengths are not measured DRAM traffic. LLVM IR/disassembly and CUDA HAL schedules are retained separately. No GPU speedup or full pretrained-model throughput claim follows from these data. Raw incorrect gather code is excluded from all timing baselines.\n\n![CPU latency](results/figures/cache-latency.png)\n\n![Allocation](results/figures/cache-allocation.png)\n';(ROOT/'RESULTS.md').write_text(text);print('Analyzed',len(rows),'groups',flush=True)
if __name__=='__main__':main()
