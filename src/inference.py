"""Offline inference runtime for the frozen NLBSE PKP2 two-layer candidate package."""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from canonical_extractor import extract_code_and_comments
from task_aware_fusion import TaskAwareFusionModel, collate_packed, pack_views

MAX_LENGTH=512
COMMENT_RESERVATION=96
GLOBAL_WIDTH=512

def sha_file(path:Path)->str:
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()

def load_manifest(root:Path|None=None):
 root=Path(root or ROOT)
 m=json.loads((root/'candidate_manifest.json').read_text(encoding='utf-8'))
 if m.get('status')!='frozen_private_offline_package':raise RuntimeError('Package manifest status is not frozen')
 if m.get('threshold_calibration_enabled') is not False:raise RuntimeError('Packaged runtime must not enable threshold calibration')
 return m

def extract_and_pack(functions:Sequence[str],tokenizer):
 feats=[];trace=[]
 for row,text in enumerate(functions):
  code,comment,ext=extract_code_and_comments(text if isinstance(text,str) else '')
  f=pack_views(tokenizer,code,comment,max_length=MAX_LENGTH,comment_budget=COMMENT_RESERVATION)
  n=len(f['input_ids'])
  if n>GLOBAL_WIDTH:raise RuntimeError(f'Canonical pack exceeded global width: row={row}, length={n}')
  pad=GLOBAL_WIDTH-n;pad_id=int(tokenizer.pad_token_id)
  for key,value in (('input_ids',pad_id),('attention_mask',0),('code_mask',0),('comment_mask',0)):
   f[key]=list(f[key])+[value]*pad
  if any(c and cm for c,cm in zip(f['code_mask'],f['comment_mask'])):raise RuntimeError(f'Evidence masks overlap at row {row}')
  if any((c or cm) and not a for c,cm,a in zip(f['code_mask'],f['comment_mask'],f['attention_mask'])):raise RuntimeError(f'Evidence mask covers padding at row {row}')
  feats.append(f);trace.append({'row':row,'has_comment':bool(f['has_comment']),'comment_tokens':int(f['comment_tokens']),
   'packed_tokens':n,'extractor':ext})
 return feats,trace

class PackagedCandidate:
 def __init__(self,candidate:str,device:str|torch.device='cpu',root:Path|None=None):
  self.root=Path(root or ROOT);self.manifest=load_manifest(self.root)
  if candidate not in self.manifest['candidates']:raise ValueError(f'Unknown candidate {candidate!r}; choose {list(self.manifest["candidates"])}')
  self.candidate=candidate;self.spec=self.manifest['candidates'][candidate]
  self.device=torch.device(device)
  if self.device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
  tokdir=self.root/'assets/tokenizer'
  self.tokenizer=AutoTokenizer.from_pretrained(str(tokdir),local_files_only=True,use_fast=True)
  self.tokenizer.padding_side='right'
  if self.tokenizer.pad_token_id is None:raise RuntimeError('Frozen tokenizer has no pad token')
  cfg=AutoConfig.from_pretrained(str(tokdir),local_files_only=True)
  cfg.num_hidden_layers=int(self.spec["encoder_layers"])
  cfg._attn_implementation='sdpa'
  encoder=AutoModel.from_config(cfg)
  self.model=TaskAwareFusionModel(encoder,int(cfg.hidden_size),mode='cls')
  cp=self.root/'assets/checkpoints'/self.spec['package_checkpoint_name']
  actual=sha_file(cp)
  if actual!=self.spec['checkpoint_sha256']:raise RuntimeError(f'{candidate}: checkpoint hash mismatch')
  state=torch.load(cp,map_location='cpu',weights_only=True)
  missing=self.model.load_state_dict(state,strict=True)
  if missing.missing_keys or missing.unexpected_keys:raise RuntimeError(f'{candidate}: strict load mismatch')
  self.model.to(self.device).eval()
  for p in self.model.parameters():p.requires_grad_(False)
  expected_layers=int(self.spec['encoder_layers'])
  actual_layers=len(self.model.encoder.encoder.layer)
  backend=getattr(self.model.encoder.config,'_attn_implementation',None)
  if actual_layers!=expected_layers or backend!='sdpa':
   raise RuntimeError(f'Expected {expected_layers}-layer SDPA encoder; got {actual_layers} layers/backend={backend}')
  self.parameter_count=sum(p.numel() for p in self.model.parameters())
  expected_params=int(self.spec['parameter_count'])
  if self.parameter_count!=expected_params:raise RuntimeError(f'Unexpected inference parameter count: expected={expected_params} actual={self.parameter_count}')
  self._forward_calls=0;self._grad_ok=True;self._detached_ok=True
  original=self.model.forward
  def guard(*args,**kwargs):
   if torch.is_grad_enabled() or not torch.is_inference_mode_enabled():self._grad_ok=False;raise RuntimeError('Forward outside inference_mode')
   out=original(*args,**kwargs);logits=out[0] if isinstance(out,tuple) else out
   if not torch.is_tensor(logits) or logits.requires_grad:self._detached_ok=False;raise RuntimeError('Forward output must be detached')
   self._forward_calls+=1;return out
  self.model.forward=guard

 def pack(self,functions:Sequence[str]):return extract_and_pack(functions,self.tokenizer)
 @torch.inference_mode()
 def forward_outputs(self,features):
  if not features:
   empty=np.empty((0,2),np.float32);return empty,empty.copy()
  batch=collate_packed(list(features),int(self.tokenizer.pad_token_id))
  ids=batch.input_ids.to(self.device);att=batch.attention_mask.to(self.device)
  cm=batch.code_mask.to(self.device);mm=batch.comment_mask.to(self.device);hc=batch.has_comment.to(self.device)
  if self.device.type=='cuda':
   with torch.autocast('cuda',dtype=torch.float16): logits,_=self.model(ids,att,cm,mm,hc)
  else: logits,_=self.model(ids,att,cm,mm,hc)
  logits32=logits.float();probs=torch.sigmoid(logits32)
  if self.device.type=='cpu':
   # Preserve exact FP32 values without depending on the local Torch/NumPy ABI.
   return np.asarray(logits32.cpu().tolist(),np.float32),np.asarray(probs.cpu().tolist(),np.float32)
  return logits32.cpu().numpy().astype(np.float32),probs.cpu().numpy().astype(np.float32)
 def forward_logits(self,features):return self.forward_outputs(features)[0]
 def probabilities(self,functions:Sequence[str],batch_size:int=600):
  if batch_size!=600:raise RuntimeError('Frozen evaluation policy requires batch size 600; no silent microbatch fallback')
  feats,trace=self.pack(functions);logs=[];probs=[]
  for start in range(0,len(feats),batch_size):
   l,p=self.forward_outputs(feats[start:start+batch_size]);logs.append(l);probs.append(p)
  logits=np.concatenate(logs,axis=0) if logs else np.empty((0,2),np.float32)
  probability=np.concatenate(probs,axis=0) if probs else np.empty((0,2),np.float32)
  return probability,trace,logits
 def frozen_predictions(self,probabilities):
  return (np.asarray(probabilities,np.float32)>=np.asarray(self.spec['thresholds'],np.float32)[None,:]).astype(np.int8)
 def guard_record(self):return {'calls':self._forward_calls,'all_grad_disabled':self._grad_ok,'all_outputs_detached':self._detached_ok}
 def close(self):
  del self.model
  if torch.cuda.is_available():torch.cuda.empty_cache()

