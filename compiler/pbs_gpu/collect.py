import sys,shutil
from pathlib import Path
root=Path.cwd();out=Path(sys.argv[1]);dest=out/'artifacts'
patterns=['research2/results/gpu-*runtime*.json','research/results/gpu-*runtime*.json']
for pattern in patterns:
 for p in root.glob(pattern):
  q=dest/p.relative_to(root);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
for directory in ['research2/results/gpu','research2/results/gpu-extensions','research/results/gpu']:
 for p in (root/directory).rglob('*'):
  if p.is_file() and (p.suffix in ['.log','.ptx'] or p.name=='repaired.mlir'):
   q=dest/p.relative_to(root);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
shutil.copy2(root/'pbs_gpu/README.md',out/'TEST_PROTOCOL.md')
