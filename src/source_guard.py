import json,hashlib
from pathlib import Path
def verify_release(root=None):
 root=Path(root or Path(__file__).resolve().parents[1]);m=json.loads((root/'RELEASE_SOURCE_MANIFEST.json').read_text())
 for name,r in m['files'].items():
  p=root/name;h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
  if h.hexdigest()!=r['sha256'] or p.stat().st_size!=r['bytes']:raise ValueError('Frozen release source/config/tokenizer changed: '+name)
 return {'verified':True,'files':len(m['files']),'identity':m['candidate_checkpoint_sha256']}
