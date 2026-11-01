import argparse,collections,csv,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('results',type=Path);a=p.parse_args();root=a.results
rows=json.loads((root/'latency.json').read_text());groups=collections.defaultdict(list)
for r in rows:
 if r['backend'].startswith('iree_') and r.get('valid',True):
  groups[(r['context'],r['mode'],r['backend'])].append(r['median_ms'])
summary=[]
for (context,mode,backend),values in sorted(groups.items()):
 ref=groups.get((context,mode,'iree_baseline'));summary.append(dict(context=context,mode=mode,backend=backend,median_of_medians_ms=statistics.median(values),iree_baseline_over_backend=statistics.median(ref)/statistics.median(values) if ref else None,observations=len(values)))
with (root/'table.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
(root/'table.json').write_text(json.dumps(summary,indent=2))
for r in summary:print(r)
