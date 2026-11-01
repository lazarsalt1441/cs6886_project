"""Exercise latest runner (SDPA, post-timing checks, resident generation).
Reuse independently compiled CPU modules; these tiny trial counts are not evidence
of a speedup. No GPU timings are reported by this local test.
"""
import gc,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import benchmark as b
from benchmark import write_json

def reuse(src,target,device,arch,folder):
 assert device=='cpu';old=ROOT/'local_checks'/('final_'+src.parent.name)/('iree_'+src.stem)/'module.vmfb';folder.mkdir(parents=True,exist_ok=True);shutil.copy2(old,target);write_json(folder/'compile_command.json',dict(local_test_reused_cpu_module=str(old),new_compilation=False));return 0.
b.compile_iree=reuse
for which in 'AB':
 sys.argv=['benchmark.py','--set',which,'--device','cpu','--contexts','8','--prompts','1','--trials','1','--repetitions','1','--warmup','1','--steps','2','--output','local_checks/protocol_'+which]
 b.main();gc.collect()
print('PASS latest full runner, both workloads incl. SDPA and input immutability')
