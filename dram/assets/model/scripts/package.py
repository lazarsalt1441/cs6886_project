from pathlib import Path
import zipfile
ROOT=Path(__file__).resolve().parents[1];out=ROOT.parent/'Attention_Real_LLM_Tests.zip'
with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in sorted(ROOT.rglob('*')):
  rel=p.relative_to(ROOT)
  if not p.is_file() or any(s in rel.parts for s in ['model','toolchain','build','.git','__pycache__']):continue
  if '.irpa' in p.name:continue
  if p.suffix in ['.irpa','.vmfb','.pt2','.tmp','.deb'] or p.stat().st_size>10*1024*1024:continue
  z.write(p,rel)
print(out, out.stat().st_size)
