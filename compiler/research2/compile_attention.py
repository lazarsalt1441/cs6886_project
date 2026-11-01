"""Compile frontend tensor IR with C++ cache rewrites and compatibility fallback.
Auto chooses the first compilable strategy, not the fastest strategy.
"""
import argparse,json,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from study_cache import normalize,forward,run,STRONG
from add_patch_variant import patch
from lift_gather_baseline import lift
from winner_study import factor
from coalesce_study import coalesce
from snapshot_study import protect

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--target',choices=['cpu','cuda'],default='cpu');ap.add_argument('--arch',default='sm_80');ap.add_argument('--strategy',choices=['auto','conditional','masked'],default='auto');ap.add_argument('--pool-demand-ratio',type=int,default=4);ap.add_argument('--compiler',default='iree-compile');ap.add_argument('--no-factor-commit',action='store_true');ap.add_argument('--no-coalesce',action='store_true');ap.add_argument('--no-snapshot-protection',action='store_true');args=ap.parse_args()
 assert args.pool_demand_ratio>=2
 source=args.input.resolve();output=args.output.resolve();output.parent.mkdir(parents=True,exist_ok=True);work=output.with_suffix('.artifacts');work.mkdir(parents=True,exist_ok=True)
 record={'input':str(source),'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'target':args.target,'arch':args.arch if args.target=='cuda' else 'host','selection':'First compilable representation; no performance prediction or numerical validation','attempts':[]}
 # Normalizing the original form also protects explicit native gather inputs.
 normalize(source,work/'original-normalized.mlir',work/'original-normalize.log')
 base=work/'original-normalized.mlir'
 if not args.no_coalesce:
  coalesce(base,work/'coalesced.mlir',work/'coalesce.log');base=work/'coalesced.mlir';record['coalesced_updates']=base.read_text().count('cache_updates_coalesced')
 if not args.no_factor_commit:
  factor(base,work/'commit-factored.mlir',work/'commit-factor.log');record['factored_commits']=(work/'commit-factored.mlir').read_text().count('commit_factored');base=work/'commit-factored.mlir'
 if not args.no_snapshot_protection:
  protect(base,work/'snapshot-protected.mlir',work/'snapshot-protect.log');base=work/'snapshot-protected.mlir';record['snapshot_guards']=base.read_text().count('cache_snapshot_guard')
 candidates=['masked','conditional'] if args.strategy=='auto' else [args.strategy]
 for strategy in candidates:
  entry={'strategy':strategy};record['attempts'].append(entry)
  try:
   if strategy=='masked':
    entry['lifted_gathers']=lift(base,work/'lifted.mlir')
    normalize(work/'lifted.mlir',work/'lifted-normalized.mlir',work/'normalize.log');patch(work/'lifted-normalized.mlir',work/'masked.mlir',work/'masked-pass.log',ratio=args.pool_demand_ratio)
   else:forward(base,work/'conditional.mlir',work/'conditional-pass.log')
   flags=['--iree-hal-target-backends=llvm-cpu','--iree-llvmcpu-target-cpu=host'] if args.target=='cpu' else ['--iree-hal-target-backends=cuda','--iree-cuda-target='+args.arch]
   entry['compile_s']=run([args.compiler,work/(strategy+'.mlir'),*flags,*STRONG,'-o',output],work/(strategy+'-compile.log'));entry['compiled']=True;record['selected']=strategy;record['output_sha256']=hashlib.sha256(output.read_bytes()).hexdigest();break
  except Exception as e:entry.update(compiled=False,error=str(e))
 output.with_suffix('.compile.json').write_text(json.dumps(record,indent=2))
 if 'selected' not in record:raise RuntimeError('No strategy compiled. Inspect '+str(work))
 print(json.dumps({'output':str(output),'strategy':record['selected'],'attempts':len(record['attempts'])}),flush=True)
if __name__=='__main__':main()
