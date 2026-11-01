"""Plots from validated collected data only. Error bars show min/max, not confidence intervals."""
import argparse,json,statistics
from pathlib import Path

def plot(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    out=Path(out);data=json.loads((out/'summary.json').read_text())
    rows=[r for r in data['counter_summaries'] if r['cache']=='warm' and r['replay']=='app-range']
    labels=sorted({r['case'] for r in rows if r['case']!='stream_control'})
    if not labels:return
    backend_order=['iree_stable','iree_commit','iree_forwarded','torch_compile']
    colors=['#405D78','#3C8A72','#8A6EAB','#D89B45']
    fig,ax=plt.subplots(figsize=(max(8,len(labels)*1.1),4.6),layout='constrained');width=.19
    for i,backend in enumerate(backend_order):
        xs=[];ys=[];lo=[];hi=[]
        for j,label in enumerate(labels):
            r=next((r for r in rows if r['case']==label and r['backend']==backend),None)
            if r:
                value=r['total_bytes_per_call'];xs.append(j+(i-1.5)*width);ys.append(value/1e6)
                lo.append((value-r['total_bytes_per_call_min'])/1e6);hi.append((r['total_bytes_per_call_max']-value)/1e6)
        if xs:ax.bar(xs,ys,width,label=backend,color=colors[i],yerr=np.array([lo,hi]),capsize=2)
    ax.set_xticks(range(len(labels)),labels,rotation=25,ha='right');ax.set_ylabel('Measured DRAM read plus write MB per call')
    ax.set_title('Warm fixed input inference  range counters');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.savefig(out/'dram_traffic.png',dpi=180);plt.close(fig)
    pairs=[r for r in data['matched_iree_comparisons'] if r['cache']=='warm' and r['replay']=='app-range']
    xs=[];traffic=[];latency=[]
    for r in pairs:
        case=r['case'];candidate='iree_forwarded' if case.startswith('cache_') else 'iree_commit'
        values={name:[x['median_ms'] for x in data['valid_latency_rows'] if x['case']==case and x['backend']==name] for name in ['iree_stable',candidate]}
        if not all(values.values()) or r['optimized_dram_bytes']<=0:continue
        xs.append(case);traffic.append(r['baseline_dram_bytes']/r['optimized_dram_bytes']);latency.append(statistics.median(values['iree_stable'])/statistics.median(values[candidate]))
    if xs:
        fig,ax=plt.subplots(figsize=(max(8,len(xs)*1.15),4.3),layout='constrained');x=np.arange(len(xs))
        ax.bar(x-.17,traffic,.34,label='Baseline / optimized DRAM bytes',color='#405D78')
        ax.bar(x+.17,latency,.34,label='Baseline / optimized unprofiled latency',color='#3C8A72')
        ax.axhline(1,color='black',linewidth=.8);ax.set_xticks(x,xs,rotation=25,ha='right');ax.set_ylabel('Ratio  higher is better')
        ax.set_title('Traffic and latency are separate outcomes');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
        fig.savefig(out/'traffic_and_latency.png',dpi=180);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();plot(a.output)
