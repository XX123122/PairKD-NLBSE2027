"""Explicit model inference on caller-supplied Parquet; raw-first, no threshold search."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parquet',required=True)
    p.add_argument('--candidate-assets',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    from src.source_guard import verify_release
    verify_release()
    import pyarrow.parquet as pq
    t=pq.read_table(a.parquet,columns=['Function'])
    if not a.execute:
        print(json.dumps({'rows':len(t),'batch':600,'model_calls':0,'requires_execute':True}));return
    from src.tool import load_candidate,infer_rows
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    rows=[{'index':i,'Function':s} for i,s in enumerate(t['Function'].to_pylist())]
    model=load_candidate(a.candidate_assets,a.device)
    prediction=infer_rows(rows,model)
    with (out/'predictions.jsonl').open('x',encoding='utf-8') as f:
        for row in prediction:f.write(json.dumps(row,allow_nan=False)+'\n')
    # Labels enter the report only after predictions have been persisted.
    labels=pq.read_table(a.parquet,columns=['Label'])['Label'].to_pylist()
    gold=[{'index':i,'Label':list(label)} for i,label in enumerate(labels)]
    with (out/'gold.jsonl').open('x') as f:
        for row in gold:f.write(json.dumps(row)+'\n')
    from src.saved_evaluation import evaluate
    (out/'metrics.json').write_text(json.dumps(evaluate(prediction,gold),indent=2)+'\n')
    import numpy as np
    probs=np.asarray([[row['prob_weakness'],row['prob_mat']] for row in prediction],np.float32)
    np.savez_compressed(out/'probabilities.npz',probabilities=probs,indices=np.arange(len(rows),dtype=np.int64))
    np.savez_compressed(out/'gold.npz',labels=np.asarray(labels,np.int8),indices=np.arange(len(rows),dtype=np.int64))
    print(json.dumps({'rows':len(rows),'output':str(out),'guard':model.guard_record(),'threshold_searches':0}))

if __name__=='__main__':main()
