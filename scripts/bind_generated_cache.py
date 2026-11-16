"""Create a train job from prepared inputs and a verified generated TRAIN cache."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.reproduce import verify_job
from src.tool import sha

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepared',required=True);a=p.parse_args()
    root=Path(a.prepared).resolve();job=json.loads((root/'train_job_base.json').read_text())
    paths={'teacher_logits':root/'teacher_cache/teacher_logits.npy','teacher_indices':root/'teacher_cache/teacher_indices.npy',
           'cache_lineage':root/'teacher_cache/CACHE_ACCEPTANCE.json'}
    for name,path in paths.items():job['inputs'][name]={'root':'data','file':str(path.relative_to(root)),'sha256':sha(path)}
    job['explicit_future_execution_consent']=False
    verify_job('train',job)
    with (root/'train_job.json').open('x') as f:json.dump(job,f,indent=2)
    print('Bound TRAIN-only cache; training remains disabled until explicitly enabled.')

if __name__=='__main__':main()
