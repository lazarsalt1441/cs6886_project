"""Generate manuscript tables/figures from measured JSON; no hand-entered timings."""
from pathlib import Path
import json, math, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'research/paper';(P/'figures').mkdir(exist_ok=True)
def read(name):return json.loads((ROOT/'research/results'/name).read_text())
def geo(x):return float(np.exp(np.mean(np.log(x))))
def sci(x):
 a,e=f'{x:.2e}'.split('e');return '$'+a+r'\times10^{'+str(int(e))+'}$'
def ratio(r):
 m=r['median_us'];return m['original']/m[r['selected']] if r['family']=='portfolio' else m.get('strong',m.get('native'))/m['optimized']
x=read('final-timings.json')['groups'];a=read('guard-validation.json')['rows']
dense=[r for r in x if r['family']=='decomposed'];rotary=[r for r in x if r['family']=='rotary'];portfolio=[r for r in x if r['family']=='portfolio']
error=max(v['max_abs'] for r in dense for v in r['checks'].values());gerror=max(r['variants']['guarded']['max_abs'] for r in a)
values=dict(AuditCount=str(len(a)),OriginalFailures=str(sum(not r['variants']['original']['correct'] for r in a)),GuardPasses=str(sum(r['variants']['guarded']['correct'] for r in a)),GuardMaxError=sci(gerror),DenseCount=str(len(dense)),DenseGeo=f'{geo([ratio(r) for r in dense]):.2f}'+r'$\times$',DenseMin=f'{min(ratio(r) for r in dense):.2f}'+r'$\times$',DenseMax=f'{max(ratio(r) for r in dense):.2f}'+r'$\times$',DenseError=sci(error),PortfolioGeo=f'{geo([ratio(r) for r in portfolio]):.2f}'+r'$\times$',PortfolioMatGeo=f'{geo([r["median_us"]["materialized"]/r["median_us"][r["selected"]] for r in portfolio]):.2f}'+r'$\times$')
(P/'numbers.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in values.items())+'\n')
def table(file,caption,label,columns,header,rows,small=True):
 out=r'\begin{table}[htbp]\centering'+('\n'+r'\small' if small else '')+'\n'+r'\caption{'+caption+'}'+r'\label{'+label+'}\n'+r'\begin{tabular}{'+columns+'}\n'+r'\toprule'+'\n'+header+r' \\ \midrule'+'\n'+'\n'.join(' & '.join(row)+r' \\' for row in rows)+'\n'+r'\bottomrule\end{tabular}\end{table}'+'\n'
 (P/file).write_text(out)
rows=[]
for d in sorted(set(r['shape'][-1] for r in a)):
 z=[r for r in a if r['shape'][-1]==d];bad=sum(not r['variants']['original']['correct'] for r in z)
 rows.append([str(d),str(len(z)),str(bad),str(sum(r['variants']['guarded']['correct'] for r in z)),sci(max(r['variants']['original']['max_abs'] for r in z)),sci(max(r['variants']['guarded']['max_abs'] for r in z))])
table('audit-table.tex','Independent audit, grouped by feature width. Both alias forms are included. Errors are maximum absolute differences from float64.','tab:audit','rrrrrr',r'$d$ & Cases & Original failures & Guarded passes & Original error & Guarded error',rows)
rows=[]
for i,r in enumerate(portfolio,1):
 m=r['median_us'];s=r['selected'];rows.append([f'P{i}',f'{m["original"]:.1f}',f'{m["materialized"]:.1f}','PM' if s=='projected-materialized' else 'M',f'{m[s]:.1f}',f'{ratio(r):.2f}',f'{m["materialized"]/m[s]:.2f}'])
table('portfolio-table.tex',r'Frozen portfolio choices on holdout inputs; latencies in $\mu$s. M: materialized, PM: projected materialized. Speedups compare the selected path with each separate baseline.','tab:portfolio','lrrlrrr',r'Case & Native & M & Choice & Selected & vs. native & vs. M',rows)
shapes={r['case']:r['shape'] for r in read('projection-study.json')['rows']+read('rotary-projection-study.json')['rows']}
rows=[]
for family,prefix in [('decomposed','D'),('rotary','R')]:
 for i,r in enumerate([v for v in x if v['family']==family],1):
  sh=shapes[r['case']];m=r['median_us'];first=m.get('strong',m.get('native'));er=max(v['max_abs'] for v in r['checks'].values())
  dims='$'+r'\times'.join(str(v) for v in sh[:6])+'$'
  if family=='rotary':dims+=f', r={sh[6]}'
  elif sh[7]:dims+=', causal'
  rows.append([prefix+str(i),dims,f'{first:.1f}',f'{m["optimized"]:.1f}',f'{ratio(r):.2f}',sci(er)])
table('operator-table.tex',r'Complete projection measurements. Dimensions are $H,M,N,\ell,d,d_v$; r denotes rotary width. Latencies are $\mu$s. D2--D3, D5--D10, D12--D13, D15--D16 and all rotary cases share the latent source across heads. The other cases use per-head latent sources.','tab:operators','llrrrr',r'ID & Dimensions & Baseline & Placed & Speedup & Max error',rows)
rows=[]
for i,r in enumerate(portfolio,1):rows.append([f'P{i}', '$'+r'\times'.join(str(v) for v in read('portfolio-study.json')['rows'][i-1]['shape'][:6])+'$'])
table('portfolio-shapes.tex',r'Portfolio dimensions $H,M,N,\ell,d,d_v$. P2--P6 share latent sources; P1 and P7 use per-head sources.','tab:pshapes','ll','ID & Dimensions',rows)
# Conditional timing uncertainty: fixed case set, bootstrap sample medians only.
rng=np.random.default_rng(6886);uncertainty={}
for family,group in [('decomposed',dense),('rotary',rotary),('portfolio_native',portfolio),('portfolio_materialized',portfolio)]:
 draws=np.zeros(4000)
 for r in group:
  left='materialized' if family=='portfolio_materialized' else 'original' if r['family']=='portfolio' else 'strong' if r['family']=='decomposed' else 'native';right=r['selected'] if r['family']=='portfolio' else 'optimized'
  aa=np.array(r['samples_us'][left]);bb=np.array(r['samples_us'][right]);am=np.median(rng.choice(aa,(4000,len(aa))),axis=1);bm=np.median(rng.choice(bb,(4000,len(bb))),axis=1);draws+=np.log(am/bm)/len(group)
 uncertainty[family]=dict(geomean=geo([r['median_us']['materialized']/r['median_us'][r['selected']] if family=='portfolio_materialized' else ratio(r) for r in group]),timing_bootstrap_95_interval=np.quantile(np.exp(draws),[.025,.975]).tolist())
(ROOT/'research/results/timing-uncertainty.json').write_text(json.dumps(uncertainty,indent=2))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
fig,ax=plt.subplots(figsize=(8.8,2.9));idx=np.arange(len(a));ax.scatter(idx,[r['variants']['original']['max_abs'] for r in a],label='Original CPU',marker='x',s=35,c='#b94435');ax.scatter(idx,[r['variants']['guarded']['max_abs'] for r in a],label='Guarded CPU',s=22,c='#236f96');ax.set_yscale('log');ax.set_ylim(3e-8,2);ax.set_ylabel('Maximum absolute error');ax.set_xlabel('Feature width (adjacent points: shape / alias cases)');ax.set_xticks(idx);ax.set_xticklabels([r['shape'][-1] for r in a],rotation=45,fontsize=8);ax.legend(loc='upper left',ncol=2);ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(P/'figures/backend-errors.png');plt.close(fig)
group=dense+rotary;ids=[f'D{i+1}' for i in range(len(dense))]+[f'R{i+1}' for i in range(len(rotary))];fig,ax=plt.subplots(figsize=(8.8,3.0));bars=ax.bar(ids,[ratio(r) for r in group],color=['#236f96']*len(dense)+['#d1842a']*len(rotary));ax.set_yscale('log');ax.set_ylim(.8,180);ax.axhline(1,color='black',linewidth=.8);ax.set_ylabel('Baseline / placed latency');ax.set_xlabel('Operator case (complete dimensions in appendix)');ax.tick_params(axis='x',labelsize=8);ax.set_yticks([1,2,5,10,20,50,100]);ax.set_yticklabels(['1','2','5','10','20','50','100']);ax.grid(axis='y',alpha=.2)
for b,r in zip(bars,group):ax.text(b.get_x()+b.get_width()/2,ratio(r)*1.08,f'{ratio(r):.1f}',ha='center',fontsize=7)
fig.tight_layout();fig.savefig(P/'figures/placement-speedups.png');plt.close(fig)
abl=read('final-ablation-timings.json');rows=[]
for r in abl:
 if r['family']=='fallback':m=r['median_us'];rows.append([r['case'].replace('_',r'\_'),f'{m["guarded"]:.1f}',f'{m["fallback"]:.1f}',f'{m["guarded"]/m["fallback"]:.1f}'])
table('fallback-table.tex',r'Correct online repair versus correct standard-Linalg fallback ($\mu$s). These shapes do not invoke projection reassociation.','tab:fallback','lrrr','Shape & Guarded online & Materialized & Ratio',rows)
print(json.dumps({'macros':values,'uncertainty':uncertainty},indent=2))
