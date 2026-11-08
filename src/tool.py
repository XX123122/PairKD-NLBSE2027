"""Label-free ordered Function entry. Frozen PKP2 only; no labels/teacher/tuning."""
import json,hashlib
from pathlib import Path
import numpy as np
CHECKPOINT='3a902b6b82b48efcd718fabe120ebc1650510c1b4e1efd45625ca3c60cf2b1c4'
THRESHOLDS=np.asarray([0.5899999737739563,0.9700000286102295],np.float32)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def normalize(rows):
 out=[]
 for position,row in enumerate(rows):
  if isinstance(row,str):row={'index':position,'Function':row}
  if not isinstance(row,dict) or set(row)!= {'index','Function'} or not isinstance(row['Function'],str) or type(row['index'])is not int:raise ValueError('Input requires only integer index and string Function; labels/project/filepath forbidden')
  out.append(row)
 return out
def read_jsonl(path):
 with Path(path).open(encoding='utf-8') as f:return normalize([json.loads(line) for line in f if line.strip()])
def postprocess(indices,probabilities):
 p=np.asarray(probabilities,np.float32)
 if p.shape!=(len(indices),2) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Expected finite FP32 probabilities [N,2]')
 pred=p>=THRESHOLDS
 return [{'index':int(index),'prob_weakness':float(v[0]),'prob_mat':float(v[1]),'pred_weakness':int(q[0]),'pred_mat':int(q[1])} for index,v,q in zip(indices,p,pred)]
def infer_rows(rows,backend):
 rows=normalize(rows)
 if not rows:return []
 p,trace,logits=backend.probabilities([r['Function'] for r in rows],batch_size=600)
 return postprocess([r['index'] for r in rows],p)
def load_candidate(assets_root,device):
 # Purpose-built legacy factory reused; no teacher or input-label loader.
 from .inference import PackagedCandidate
 root=Path(assets_root);m=json.loads((root/'candidate_manifest.json').read_text());spec=m['candidates']['PKP2-seed13']
 tokenizer_manifest=json.loads((Path(__file__).resolve().parents[1]/'TOKENIZER_ASSETS.json').read_text())
 for relative,r in tokenizer_manifest.items():
  if sha(root/'assets/tokenizer'/relative)!=r['sha256']:raise ValueError('Frozen tokenizer asset hash mismatch: '+relative)
 if spec['checkpoint_sha256']!=CHECKPOINT or spec['source_layer_indices']!=[0,11] or spec['parameter_count']!=53771526 or not np.array_equal(np.asarray(spec['thresholds'],np.float32),THRESHOLDS):raise ValueError('Candidate graph/threshold identity mismatch')
 if sha(root/'assets/checkpoints'/spec['package_checkpoint_name'])!=CHECKPOINT:raise ValueError('Checkpoint content mismatch before real model construction')
 return PackagedCandidate('PKP2-seed13',device,root)
def tensor_logits_hook(candidate,features):
 """Reference integration hook: [N,2] FP32 logits; caller owns reference scoring."""
 return candidate.forward_logits(features)
def score(mean_f1,runtime_sec,gflops):return .6*mean_f1+.2*max(1-runtime_sec/1000,0)+.2*max(1-gflops/3000000,0)
