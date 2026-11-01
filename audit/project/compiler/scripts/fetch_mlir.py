"""Extract an immutable MLIR 19.1.1 SDK without root or a system upgrade.
Ubuntu24.04 x86_64 only. Normal package-manager installation also works.
"""
import concurrent.futures, hashlib, pathlib, subprocess
ROOT=pathlib.Path(__file__).resolve().parents[1]
NAMES=['libmlir-19-dev','libmlir-19','libllvm19','llvm-19-dev','llvm-19','llvm-19-runtime','llvm-19-linker-tools','mlir-19-tools','llvm-19-tools']
VER='19.1.1-1ubuntu1~24.04.2'
expected={line.split()[1]:line.split()[0] for line in (ROOT/'docs/toolchain.sha256').read_text().splitlines()}
def fetch(n):
 root=ROOT/'toolchain';(root/'debs').mkdir(parents=True,exist_ok=True)
 f=root/'debs'/f'{n}_{VER}_amd64.deb'
 if not f.exists() or hashlib.sha256(f.read_bytes()).hexdigest()!=expected[f.name]:
  section='main' if n=='libllvm19' else 'universe'
  url=f'https://archive.ubuntu.com/ubuntu/pool/{section}/l/llvm-toolchain-19/{f.name}'
  subprocess.run(['curl','-fL','--retry','2','--max-time','180','-o',str(f),url],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 if hashlib.sha256(f.read_bytes()).hexdigest()!=expected[f.name]:raise RuntimeError(f'Hash mismatch: {f.name}')
 return f
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:files=list(executor.map(fetch,NAMES))
# Extract sequentially because Debian packages share directories and symlinks.
for f in files:
 subprocess.run(['dpkg-deb','-x',str(f),str(ROOT/'toolchain')],check=True)
 print('Verified and extracted',f.name,flush=True)
