import hashlib,json,os,subprocess,sys
from pathlib import Path
from iree.compiler import ir
ROOT=Path(__file__).resolve().parents[1]
def apply(out):
 toolroot=ROOT/'compiler'
 sdk=toolroot/'toolchain/usr/lib';env=os.environ.copy();env['LD_LIBRARY_PATH']=str(sdk/'llvm-19/lib')+':'+str(sdk/'x86_64-linux-gnu')+':'+env.get('LD_LIBRARY_PATH','')
 for variant,passes in [('baseline',[]),('optimized',['--attention-factor-cache-commit','--attention-coalesce-cache-updates','--attention-forward-cache-reads'])]:
  cmd=[str(toolroot/'build/gqa-opt'),str(out/'generic.mlir'),'--allow-unregistered-dialect','--canonicalize','--cse',*passes,'--canonicalize','-o',str(out/(variant+'-generic.mlir'))]
  with (out/(variant+'-pass.log')).open('w') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  with ir.Context():m=ir.Module.parse((out/(variant+'-generic.mlir')).read_text());(out/(variant+'.mlir')).write_text(str(m))
 before=(out/'baseline.mlir').read_text();after=(out/'optimized.mlir').read_text()
 return dict(markers={s:after.count(s)-before.count(s) for s in ['attention.concat_commit_factored','attention.commit_factored','attention.cache_read_forwarded','attention.cache_updates_coalesced']},baseline_sha256=hashlib.sha256(before.encode()).hexdigest(),optimized_sha256=hashlib.sha256(after.encode()).hexdigest(),identical_ir=before==after)
if __name__=='__main__':
 for label in sys.argv[1:]:
  out=ROOT/'prepared'/label;audit=json.loads((out/'audit.json').read_text());audit.update(apply(out));audit['any_rewrite']=any(v>0 for v in audit['markers'].values());(out/'audit.json').write_text(json.dumps(audit,indent=2));print(label,audit['markers'])
