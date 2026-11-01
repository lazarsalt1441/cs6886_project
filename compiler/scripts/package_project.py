"""Atomically package source, git checkpoints and results; exclude rebuildable SDKs."""
from pathlib import Path
import zipfile,os,json,subprocess
ROOT=Path(__file__).resolve().parents[1];dest=ROOT.parent/'gqa-compiler-project.zip';temp=dest.with_suffix('.staging.zip')
skip={'vendor','toolchain','build','.venv','__pycache__','.tmp'}
required=['src/CacheReadForwarding.cpp','src/GatherIndexProjection.cpp','src/CacheGatherForwarding.cpp','src/CacheCommitFactoring.cpp','src/CacheUpdateCoalescing.cpp','src/CacheSnapshotProtection.cpp','research2/TECHNICAL_REPORT.md','research2/GPU_RUN.md','research2/results/extension-timings.json','research2/results/final-structural-tests.json','research2/results/winner-study.json','research2/results/final-timings.json','README.md','RESUME.md','STATUS.md','.git/HEAD']
with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for parent,dirs,files in os.walk(ROOT):
  dirs[:]=sorted(d for d in dirs if d not in skip)
  for name in sorted(files):
   if name.endswith(('.tmp','.pyc','.aux','.out','.lock')) or name=='ninja-deps-corrupt.bin':continue
   path=Path(parent)/name
   if path.is_symlink():continue
   if path.suffix in ['.mlir','.vmfb'] and path.stat().st_size==0:raise RuntimeError('Empty compiler artifact: '+str(path))
   z.write(path,'gqa-compiler/'+str(path.relative_to(ROOT)))
with zipfile.ZipFile(temp) as z:
 assert z.testzip() is None
 for p in required:
  info=z.getinfo('gqa-compiler/'+p);assert info.file_size>0,p
 count=len(z.infolist())
os.replace(temp,dest)
print(json.dumps({'path':str(dest),'bytes':dest.stat().st_size,'members':count,'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}))
