"""Prepare the original fixed split, packed TRAIN and fresh seed13 head; zero forwards."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.tool import sha
from src.reproduce import SPLIT,TEACHER

ROOT=Path(__file__).resolve().parents[1]
HEAD_TENSOR='cdf8862eb1de993a0d9ad94a9f4e49e4d1bf4fc55edd52d3be33f8ab6a70cacb'

def tensor_hash(tensors):
    h=hashlib.sha256()
    for name,t in sorted(tensors.items()):
        t=t.detach().cpu().contiguous()
        h.update(name.encode());h.update(str((str(t.dtype),tuple(t.shape))).encode())
        h.update(bytes(t.reshape(-1).view(__import__('torch').uint8).tolist()))
    return h.hexdigest()

def create_head(path):
    import torch
    from src.task_aware_fusion import TaskAwareFusionModel
    torch.manual_seed(13)
    # Identity consumes no random numbers and has no parameters. Exactly the same
    # classifier/gate constructors as the historical seeded head, no encoder needed.
    model=TaskAwareFusionModel(torch.nn.Identity(),768,mode='cls')
    head=model.state_dict()
    actual=tensor_hash(head)
    # Initialization kernels can differ across PyTorch versions. A generated
    # template is used only in new_recipe_reproduction, never relabeled original.
    torch.save(head,path)
    return actual

def pack_train(frame,indices,tokenizer):
    import numpy as np
    from src.canonical_extractor import extract_code_and_comments
    from src.task_aware_fusion import pack_views
    rows=[]
    for i in indices:
        record=frame.iloc[int(i)]
        code,comment,_=extract_code_and_comments(record['Function'])
        row=pack_views(tokenizer,code,comment,512,comment_budget=96)
        row.update(index=int(i),labels=[int(v) for v in record['Label']])
        rows.append(row)
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-parquet',required=True,help='Unmodified official train split in source row order, 214586 rows')
    p.add_argument('--reproduction-assets',required=True,help='Unpacked Pair42+fixed-split release attachment')
    p.add_argument('--output',required=True);a=p.parse_args()
    import numpy as np
    import pandas as pd
    from transformers import AutoTokenizer
    assets=Path(a.reproduction_assets).resolve();out=Path(a.output).resolve()
    if sha(assets/'split.npz')!=SPLIT or sha(assets/'Pair42.pt')!=TEACHER:raise ValueError('Teacher/split mismatch')
    frame=pd.read_parquet(a.train_parquet)
    if len(frame)!=214586 or not {'Function','Label'}.issubset(frame.columns):raise ValueError('Unexpected official train split')
    with np.load(assets/'split.npz',allow_pickle=False) as z:indices=z['train_indices'];valid=z['validation_indices']
    if len(indices)!=193124 or len(valid)!=21462 or len(np.unique(np.r_[indices,valid]))!=214586:raise ValueError('Split error')
    out.mkdir(parents=True,exist_ok=False)
    tokenizer=AutoTokenizer.from_pretrained(str(ROOT/'assets/tokenizer'),local_files_only=True,use_fast=True)
    tokenizer.padding_side='right'
    rows=pack_train(frame,indices,tokenizer)
    if np.asarray([r['labels'] for r in rows]).sum(0).tolist()!=[63334,3184]:raise ValueError('Label support drift')
    pd.DataFrame(rows).to_parquet(out/'train_packed.parquet',index=False)
    headhash=create_head(out/'head_seed13.pt')
    config=json.loads((ROOT/'FROZEN_CONFIG.json').read_text())
    roots={'artifact_root':str(out),'model_root':str(assets),'data_root':str(out)}
    def item(path):
        path=path.resolve()
        root='model' if path.is_relative_to(assets) else 'data'
        return {'root':root,'file':str(path.relative_to(assets if root=='model' else out)),'sha256':sha(path)}
    # Tokenizer/config can live in the repo, represented by an explicit model root.
    import shutil
    shutil.copytree(ROOT/'assets/tokenizer',out/'tokenizer')
    inputs={'split':item(assets/'split.npz'),'teacher_checkpoint':item(assets/'Pair42.pt'),
            'train_packed':item(out/'train_packed.parquet'),'encoder_config':item(out/'tokenizer/config.json')}
    common=dict(candidate='PKP2-seed13',scope='fixed training partition only',**roots,
                seed=13,max_length=512,comment_reservation=96,device='cuda:0',explicit_future_execution_consent=False)
    cache=dict(common,inputs=inputs,output_subdir='teacher_cache')
    (out/'cache_job.json').write_text(json.dumps(cache,indent=2)+'\n')
    train=dict(common,inputs={**inputs,'head_template':item(out/'head_seed13.pt')},mode='new_recipe_reproduction',
               output_subdir='student',schedule_hashes=config['schedule_hashes'],head_tensor_sha256=headhash)
    (out/'train_job_base.json').write_text(json.dumps(train,indent=2)+'\n')
    receipt=dict(source_train_parquet_sha256=sha(a.train_parquet),train_packed_sha256=sha(out/'train_packed.parquet'),
                 head_file_sha256=sha(out/'head_seed13.pt'),head_tensor_sha256=headhash,rows=193124,
                 model_forwards=0,teacher_checkpoint_sha256=TEACHER,split_sha256=SPLIT)
    (out/'PREPARATION.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('Prepared TRAIN features and seed13 head. Set consent explicitly before cache/train. No model forward performed.')

if __name__=='__main__':main()
