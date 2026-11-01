"""Ensure final stricter C++ guard reproduces each measured configured update."""
import sys,json,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from projection_study import ROOT
from guard_config import walk
from iree.compiler import ir
rows=[]
for source in sorted(Path('research/results/backend-audit').glob('*/fix-bridge/input-0.mlir')):
 expected=source.with_name('output-0.mlir')
 result=subprocess.run(['build/gqa-opt',source,'--allow-unregistered-dialect','--attention-guard-online-reductions'],check=True,capture_output=True,text=True).stdout
 with ir.Context():
  original=ir.Module.parse(expected.read_text());actual=ir.Module.parse(result)
  configs=lambda module:[str(op.attributes['lowering_config']) for op in walk(module.operation) if op.name=='iree_linalg_ext.online_attention']
  same=configs(original)==configs(actual);assert same,str(source)
 rows.append({'fixture':str(source),'same_lowering_config':same})
assert len(rows)==20,len(rows)
Path('research/results/final-guard-verification.json').write_text(json.dumps(rows,indent=2));print(len(rows),'measured configurations reproduced')
