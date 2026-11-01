import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));import benchmark as b
b.STRONG+=['--iree-input-demote-f64-to-f32=false','--iree-opt-numeric-precision-reduction=false']
r=Path(__file__).resolve().parent;out=r/'prepared';rows=[]
for label,extra in [('A_stable',False),('B_stable',False),('B_stable_commit',False),('B_stable_commit_aggressive',True)]:
 if extra:b.STRONG+=['--iree-dispatch-creation-enable-aggressive-fusion']
 folder=out/(label+'_cuda');src=out/('B_stable_commit.mlir' if extra else label+'.mlir');sec=b.compile_iree(src,folder/'module.vmfb','cuda','sm_75',folder)
 ptx=''.join(p.read_text() for p in (folder/'codegen').glob('*.ptx'));assert 'f64' in ptx,'Repair was demoted!'
 rows.append(dict(label=label,seconds=sec,execution_tested=False,contains_f64=True,aggressive_fusion=extra));b.write_json(out/'cuda_manifest.json',rows);print(label,'F64 CUDA COMPILE PASS',flush=True)
