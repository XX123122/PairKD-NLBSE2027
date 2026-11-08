"""Verify and import only the original PKP2 checkpoint from its exact archive."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]
ZIP_SHAS={
    '7e668e63951709c22d7f7536d294a0586bee446e1117f7f685b9e7e828a8a7fb',
    '42e2a11e47fce3c1a1dc6c2594840a4e02f5a25fd3f0e332ab73dc5be03a359d',
}
CP_SHA='3a902b6b82b48efcd718fabe120ebc1650510c1b4e1efd45625ca3c60cf2b1c4'

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    sys.path.insert(0,str(ROOT))
    from src.source_guard import verify_release
    verify_release()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-zip',required=True);parser.add_argument('--assets-root',required=True)
    args=parser.parse_args()
    archive=Path(args.model_zip).resolve();dest=Path(args.assets_root).resolve()
    if dest.exists():raise FileExistsError('Use a fresh assets-root')
    if sha(archive) not in ZIP_SHAS:raise ValueError('Not the exact selected deployment ZIP')
    with zipfile.ZipFile(archive) as z, tempfile.TemporaryDirectory(prefix='pairkd-import-') as temp:
        matches=[i for i in z.infolist() if i.filename=='assets/checkpoints/PKP2-seed13.pt' or i.filename.endswith('/assets/checkpoints/PKP2-seed13.pt')]
        if len(matches)!=1 or matches[0].file_size!=215102090:
            raise ValueError('Selected checkpoint identity/size mismatch')
        checkpoint=Path(temp)/'PKP2-seed13.pt'
        # Never extract archive paths or nested historical source archives.
        with z.open(matches[0]) as source,checkpoint.open('xb') as out:
            while block:=source.read(8*1024*1024):out.write(block)
        if sha(checkpoint)!=CP_SHA:raise ValueError('Checkpoint SHA mismatch')
        subprocess.run([sys.executable,str(ROOT/'bootstrap_assets.py'),'--checkpoint',str(checkpoint),
                        '--assets-root',str(dest)],check=True)

if __name__=='__main__':main()
