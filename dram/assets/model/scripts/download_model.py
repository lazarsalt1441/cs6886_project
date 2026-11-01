import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from llm_workloads import MODEL_ID,REVISION
from huggingface_hub import snapshot_download
names=['model.safetensors','config.json','generation_config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json']
snapshot_download(MODEL_ID,revision=REVISION,local_dir=ROOT/'model',allow_patterns=names)
actual={n:hashlib.file_digest((ROOT/'model'/n).open('rb'),'sha256').hexdigest() for n in names}
p=ROOT/'checkpoint_sha256.json'
if p.exists() and json.loads(p.read_text())!=actual:raise RuntimeError('Pinned checkpoint hash mismatch')
p.write_text(json.dumps(actual,indent=2));print('PINNED CHECKPOINT VERIFIED',REVISION,flush=True)
