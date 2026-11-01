"""Use the recovered, rebuilt MLIR19 C++ optimizer, not a Python rewrite.
Run from audit parent. --project points to the recovered source package.
"""
from pathlib import Path
import argparse,json,subprocess,os
P=Path(__file__).resolve().parents[1];ap=argparse.ArgumentParser();ap.add_argument('--project',required=True);a=ap.parse_args();proj=Path(a.project).resolve();env=os.environ.copy();sdk=proj/'compiler/toolchain/usr/lib';env['LD_LIBRARY_PATH']=str(sdk/'llvm-19/lib')+':'+str(sdk/'x86_64-linux-gnu')+':'+env.get('LD_LIBRARY_PATH','');tool=proj/'compiler/build/gqa-opt';cases=json.loads((P/'reproducers/cases.json').read_text());records=[]
def transform(name,src,flags,meta):
 dest=P/'reproducers'/(name+'.mlir');cmd=[str(tool),str(src),'--canonicalize',*flags,'--canonicalize','-o',str(dest)];p=subprocess.run(cmd,env=env,text=True,capture_output=True);(P/'results/actual_pass'/(name+'.log')).write_text(p.stderr);p.check_returncode();cases.append(dict(meta,name=name));records.append(dict(name=name,command=cmd,markers=dest.read_text().count('attention.cache_read_forwarded')+dest.read_text().count('attention.commit_factored')))
for N in [512,8192]:
 meta=next(c for c in cases if c['name']==f'gather_forks_{N}_custom');transform(f'gather_forks_{N}_actual',P/'reproducers'/f'gather_forks_{N}_stock.mlir',['--attention-forward-cache-reads'],meta)
# Match the actual winner pass's linalg.generic selection representation.
s=(P/'reproducers/winner_stock.mlir').read_text();old='%r = arith.select %yes, %u, %v : tensor<4096xf32>'
s='module {\nfunc.func @main(%base: tensor<4096xf32>, %a: tensor<16xf32>, %b: tensor<16xf32>, %cond: tensor<i1>) -> tensor<4096xf32> {\n%u ='+s.split('%u =',1)[1]
new='''%e = tensor.empty() : tensor<4096xf32>
%r = linalg.generic {indexing_maps=[affine_map<(i)->()>,affine_map<(i)->(i)>,affine_map<(i)->(i)>,affine_map<(i)->(i)>],iterator_types=["parallel"]} ins(%cond,%u,%v : tensor<i1>,tensor<4096xf32>,tensor<4096xf32>) outs(%e : tensor<4096xf32>) {
^bb0(%gc:i1,%ga:f32,%gb:f32,%go:f32):
%selected = arith.select %gc,%ga,%gb : f32
linalg.yield %selected : f32
} -> tensor<4096xf32>'''
assert old in s;s=s.replace(old,new);src=P/'reproducers/winner_generic_stock.mlir';src.write_text(s);meta=dict(next(c for c in cases if c['name']=='winner_stock'),shapes=[[4096],[16],[16],[]],kind='winner_bool');cases.append(dict(meta,name='winner_generic_stock'));transform('winner_actual',src,['--attention-factor-cache-commit'],dict(meta,custom=True))
merged={c['name']:c for c in cases};(P/'reproducers/cases.json').write_text(json.dumps(list(merged.values()),indent=2));(P/'results/actual_pass/minimal_transform.json').write_text(json.dumps(records,indent=2));print(records)
