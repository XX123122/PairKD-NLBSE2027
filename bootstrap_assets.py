"""Prepare local asset layout from author-supplied original checkpoint; no download/model."""
import argparse,hashlib,shutil,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--assets-root',required=True);a=p.parse_args();cp=Path(a.checkpoint)
 if sha(cp)!='3a902b6b82b48efcd718fabe120ebc1650510c1b4e1efd45625ca3c60cf2b1c4':raise RuntimeError('Original PKP2 checkpoint SHA mismatch')
 dest=Path(a.assets_root);dest.mkdir(exist_ok=False);(dest/'assets/checkpoints').mkdir(parents=True);shutil.copy2(cp,dest/'assets/checkpoints/PKP2-seed13.pt');shutil.copytree(ROOT/'assets/tokenizer',dest/'assets/tokenizer');shutil.copy2(ROOT/'candidate_manifest.json',dest/'candidate_manifest.json');print(json.dumps({'asset_root':str(dest),'checkpoint_sha256':sha(cp),'downloaded':False,'model_initialized':False}))
if __name__=='__main__':main()
