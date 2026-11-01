"""Coalesce per-token updates, then forward cache observations."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from probe_update_chain import ChainAttention,chain_inputs
from study_cache import *
from add_patch_variant import patch

def coalesce(source,dest,log):
 with ir.Context():mod=ir.Module.parse(Path(source).read_text());temp=Path(dest).with_name(Path(dest).stem+'-input.mlir');temp.write_text(mod.operation.get_asm(print_generic_op_form=True))
 return run(['build/gqa-opt',temp,'--allow-unregistered-dialect','--attention-coalesce-cache-updates','--canonicalize','-o',dest],log)
def main():
 torch.set_num_threads(1);rows=json.loads(Path('research2/results/update-chain-probe.json').read_text())
 for row in rows:
  t=row['tokens'];w=Path('research2/results/update-chain')/f't{t}';coalesce(w/'baseline.mlir',w/'coalesced.mlir',w/'coalesce.log');forward(w/'coalesced.mlir',w/'coalesced-forwarded.mlir',w/'coalesced-forward.log');lift(w/'coalesced.mlir',w/'coalesced-lifted.mlir');normalize(w/'coalesced-lifted.mlir',w/'coalesced-normalized.mlir',w/'coalesced-normalize.log');patch(w/'coalesced-normalized.mlir',w/'coalesced-masked.mlir',w/'coalesced-masked.log',ratio=4)
  for v,stem in [('coalesced','coalesced'),('coalesced-forwarded','coalesced-forwarded'),('coalesced-masked','coalesced-masked')]:
   flags=['--iree-hal-target-backends=llvm-cpu','--iree-llvmcpu-target-cpu=host',*STRONG];cs=run(['iree-compile',w/(stem+'.mlir'),*flags,'-o',w/(v+'.vmfb')],w/(v+'-compile.log'));run(['iree-compile',w/(stem+'.mlir'),*flags,'--compile-to=hal','-o',w/(v+'-hal.mlir')],w/(v+'-hal.log'));checks=[]
   for seed in [171,613,811]:
    x=chain_inputs(t,seed);refs=reference(ChainAttention(t),x);cfg,ctx,fn,args=load(w/(v+'.vmfb'),[a.clone() for a in x]);before=cfg.device.allocator.statistics;ys=fn(*args);after=cfg.device.allocator.statistics;assert len(ys)==len(refs);errors=[]
    for a,b in zip(ys,refs):actual=a.to_host();np.testing.assert_allclose(actual,b,atol=4e-5,rtol=1e-3);errors.append(float(np.max(np.abs(actual-b))))
    checks.append({'seed':seed,'max_abs':max(errors),'correct':True})
   text=(w/(stem+'.mlir')).read_text();hal=(w/(v+'-hal.mlir')).read_text();row['variants'][v]={'correct':True,'checks':checks,'allocated_bytes':after['device_bytes_allocated']-before['device_bytes_allocated'],'copies':copy_stats(hal),'remaining_insert_slices':text.count('tensor.insert_slice'),'coalesced_updates':text.count('cache_updates_coalesced'),'read_forwards':text.count('cache_read_forwarded'),'masked_forwards':text.count('cache_gather_forwarded'),'compile_s':cs,'vmfb_sha256':hashlib.sha256((w/(v+'.vmfb')).read_bytes()).hexdigest()}
  Path('research2/results/update-chain-study.json').write_text(json.dumps(rows,indent=2));print(t,{v:r['allocated_bytes'] for v,r in row['variants'].items()},flush=True)
if __name__=='__main__':main()
