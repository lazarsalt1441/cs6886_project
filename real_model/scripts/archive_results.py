"""Checkpoint only reproducibility artifacts; tolerate active atomic file writes."""
import argparse,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def archive(which,destination):
 destination=Path(destination);tmp=destination.with_suffix('.tmp')
 paths=[ROOT/('results_'+which),ROOT/('runner_'+which+'.log'),ROOT/'setup.log',ROOT/'checkpoint_sha256.json',ROOT/'requirements-colab.txt',ROOT/'llm_workloads.py',ROOT/'benchmark.py',ROOT/'analyze.py',ROOT/'STATUS.md']
 files=[]
 for path in paths:
  if path.is_dir():files.extend(path.rglob('*'))
  elif path.exists():files.append(path)
 files.extend(p for p in (ROOT/'prepared'/which).glob('*') if p.suffix in ['.mlir','.json','.log'])
 with tarfile.open(tmp,'w:gz') as tar:
  for path in sorted(files):
   if path.suffix=='.tmp' or not path.is_file():continue
   try:tar.add(path,arcname=path.relative_to(ROOT),recursive=False)
   except FileNotFoundError:pass  # JSON writer atomically replaced this file.
 tmp.replace(destination);return destination
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--set',required=True,choices=['A','B']);p.add_argument('--destination',required=True);a=p.parse_args();print(archive(a.set,a.destination))
