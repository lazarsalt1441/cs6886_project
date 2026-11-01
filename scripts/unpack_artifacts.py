"""Restore SHA-verified generated evidence and prepared IR to original paths."""
from pathlib import Path
import argparse, hashlib, json, tarfile
ROOT=Path(__file__).resolve().parents[1]
def unpack():
    manifest=json.loads((ROOT/'artifacts/manifest.json').read_text())
    for archive,entry in manifest.items():
        destination=ROOT/'artifacts'/archive
        if not destination.exists() or hashlib.file_digest(destination.open('rb'),'sha256').hexdigest()!=entry['sha256']:
            temporary=destination.with_suffix('.joining')
            with temporary.open('wb') as out:
                for name in entry['parts']:
                    part=ROOT/'artifacts'/name
                    if hashlib.file_digest(part.open('rb'),'sha256').hexdigest()!=entry['part_sha256'][name]:raise RuntimeError('Bad archive part: '+name)
                    with part.open('rb') as source:
                        while chunk:=source.read(1024*1024):out.write(chunk)
            temporary.replace(destination)
        if hashlib.file_digest(destination.open('rb'),'sha256').hexdigest()!=entry['sha256']:raise RuntimeError('Bad archive: '+archive)
        with tarfile.open(destination) as tar:
            members=tar.getmembers()
            for m in members:
                p=Path(m.name)
                if p.is_absolute() or '..' in p.parts or not m.isfile():raise RuntimeError('Unsafe member: '+m.name)
                target=ROOT/p
                target.parent.mkdir(parents=True,exist_ok=True)
                data=tar.extractfile(m).read()
                # Preserve a deliberate local source modification.
                if target.exists() and target.read_bytes()!=data:raise RuntimeError('Would overwrite modified file: '+str(p))
                target.write_bytes(data)
        print('Restored',archive,len(members),'files')
if __name__=='__main__':unpack()
