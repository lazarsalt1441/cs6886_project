"""Dependency-free source inventory, integrity and prepared-input checks."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1]
def main():
    manifest=json.loads((ROOT/'SOURCE_MANIFEST.json').read_text());bad=[]
    for name,expected in manifest.items():
        p=ROOT/name
        if not p.is_file() or hashlib.file_digest(p.open('rb'),'sha256').hexdigest()!=expected:bad.append(name)
    if bad:raise SystemExit('Integrity failures: '+', '.join(bad))
    sys.path.insert(0,str(ROOT/'dram'))
    from common import verify_assets
    verify_assets()
    assert (ROOT/'compiler/src/CacheReadForwarding.cpp').exists()
    assert (ROOT/'real_model/compiler/src/StableTrig.cpp').exists()
    print('PASS',len(manifest),'tracked source/evidence hashes and all DRAM prepared assets')
if __name__=='__main__':main()
