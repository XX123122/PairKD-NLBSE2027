"""Portable explicit opt-in reproduction,not exact measured-checkpoint regeneration."""
import json,math,random
from pathlib import Path
import numpy as np
from .tool import sha
SPLIT='f0e361f9bf741dd8f6c81137e9977d5a28332c0949627f0d407cc067de0eafe9'
TEACHER='447baa87fd51a5e3519c6662f60d2a42e5e88aebcf42ac84441559b418110c0f'
LOGITS='5e953e450ba83e0eeaef805dc72b9bad6857e89fa21c9c085dbd1282432dd3c8'
WEIGHTS=[2.151758909225464,59.65452194213867]
def verify_job(command,job):
 if command not in ('train','cache','tune') or job.get('candidate')!='PKP2-seed13' or job.get('scope')!=('fixed training partition only' if command!='tune' else 'explicit saved tuning probabilities'):raise ValueError('Wrong candidate/scope; inference never tunes or loads teacher')
 root=Path(job['artifact_root']).resolve();assets=Path(job['model_root']).resolve();data=Path(job['data_root']).resolve();paths={}
 for key,r in job['inputs'].items():
  base={'data':data,'model':assets,'artifact':root}[r['root']];p=(base/r['file']).resolve()
  if not p.is_relative_to(base) or sha(p)!=r['sha256']:raise ValueError('Input root/hash mismatch: '+key)
  paths[key]=p
 if command in ('train','cache'):
  if job['inputs']['split']['sha256']!=SPLIT or job['inputs']['teacher_checkpoint']['sha256']!=TEACHER:raise ValueError('Original PKP2 teacher/split required,not grouped substitutes')
  if job['seed']!=13 or job['max_length']!=512 or job['comment_reservation']!=96:raise ValueError('Frozen recipe fields')
 if command=='train' and job['inputs']['teacher_logits']['sha256']!=LOGITS:
  if job.get('mode')!='new_recipe_reproduction' or 'cache_lineage' not in paths:raise ValueError('Original measured-target branch requires original TRAIN logits; fresh cache needs separately bound reproduction mode')
  lineage=json.loads(paths['cache_lineage'].read_text())
  if lineage['teacher_sha256']!=TEACHER or lineage['split_sha256']!=SPLIT or lineage['scope']!='fixed training partition only' or lineage['rows']!=193124 or lineage['logits_sha256']!=sha(paths['teacher_logits']) or lineage['indices_sha256']!=sha(paths['teacher_indices']):raise ValueError('Fresh reproduction cache provenance mismatch')
 return root,paths
def run_job(command,path,execute=False):
 job=json.loads(Path(path).read_text());root,paths=verify_job(command,job)
 plan={'command':command,'source_bound_inputs':{k:{'path':str(v),'sha256':sha(v)} for k,v in paths.items()},'model_initializations':0,'dry_run':not execute,'candidate':'PKP2-seed13','reproduction_not_exact_checkpoint_hash_promise':True}
 if not execute:return plan
 if not job.get('explicit_future_execution_consent'):raise ValueError('This preparation did not grant training/cache/tuning execution')
 out=root/job['output_subdir']
 if not out.resolve().is_relative_to(root):raise ValueError('Output outside artifact root')
 out.mkdir(exist_ok=False)
 if command=='tune':
  # Separate explicit opt-in saved probability tuning; NEVER reached by inference.
  with np.load(paths['saved_probabilities'],allow_pickle=False) as a:p=a['probabilities'];indices=a['indices']
  with np.load(paths['gold'],allow_pickle=False) as a:
   if not np.array_equal(indices,a['indices']):raise ValueError('Tuning order mismatch')
   y=a['labels']
  chosen=[]
  for j in range(2):
   values=[];grid=np.linspace(.01,.99,99,dtype=np.float32)
   for t in grid:q=p[:,j]>=t;tp=((y[:,j]==1)&q).sum();fp=((y[:,j]==0)&q).sum();fn=((y[:,j]==1)&~q).sum();values.append(2*tp/max(2*tp+fp+fn,1))
   chosen.append(float(grid[int(np.argmax(values))]))
  (out/'NEW_REPRODUCTION_THRESHOLDS.json').write_text(json.dumps({'thresholds':chosen,'scope':job['selection_disclosure'],'NOT_original_candidate_thresholds':True}));return {'output':str(out),'new_thresholds':chosen}
 import torch
 import pandas as pd
 from transformers import AutoConfig,RobertaModel,get_linear_schedule_with_warmup
 from torch.utils.data import DataLoader
 from .task_aware_fusion import TaskAwareFusionModel,collate_packed
 device=torch.device(job['device']);frame=pd.read_parquet(paths['train_packed']);indices=frame['index'].to_numpy(np.int64)
 with np.load(paths['split'],allow_pickle=False) as a:
  expected=a['train_indices'];validation=a['validation_indices'] if 'validation_indices' in a else a['val_indices']
 if len(frame)!=193124 or len(validation)!=21462 or not np.array_equal(indices,expected):raise ValueError('Original TRAIN-only193124 order required')
 labels=np.asarray(frame['labels'].tolist(),np.int8)
 if labels.shape!=(193124,2) or not np.isin(labels,[0,1]).all() or labels.sum(0).tolist()!=[63334,3184]:raise ValueError('Original two-label TRAIN support contract')
 feats=frame.to_dict('records');cfg=AutoConfig.from_pretrained(str(paths['encoder_config'].parent),local_files_only=True);cfg._attn_implementation='sdpa';state=torch.load(paths['teacher_checkpoint'],map_location='cpu',weights_only=True)
 if command=='cache':
  cfg.num_hidden_layers=12;model=TaskAwareFusionModel(RobertaModel(cfg),768,mode='cls');model.load_state_dict(state,strict=True);model.to(device).eval();values=[];calls=0
  for start in range(0,len(feats),48):
   b=collate_packed(feats[start:start+48],1)
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=device.type=='cuda'):logs,_=model(b.input_ids.to(device),b.attention_mask.to(device),b.code_mask.to(device),b.comment_mask.to(device),b.has_comment.to(device))
   values.append(logs.float().cpu().numpy());calls+=1
  np.save(out/'teacher_logits.npy',np.concatenate(values));np.save(out/'teacher_indices.npy',indices);(out/'CACHE_ACCEPTANCE.json').write_text(json.dumps({'teacher_sha256':TEACHER,'split_sha256':SPLIT,'scope':'fixed training partition only','rows':193124,'logits_sha256':sha(out/'teacher_logits.npy'),'indices_sha256':sha(out/'teacher_indices.npy'),'precision':'BF16 forward / FP32 logits','teacher_forwards':calls,'new_recipe_reproduction_only':True}));return {'output':str(out),'teacher_forwards':calls,'scope':'TRAIN only','new_cache_does_not_overwrite_original':True}
 cfg.num_hidden_layers=2;encoder=RobertaModel(cfg);mapped={}
 for k in encoder.state_dict():
  source=k
  if k.startswith('encoder.layer.'):
   parts=k.split('.');parts[2]=str([0,11][int(parts[2])]);source='.'.join(parts)
  mapped[k]=state['encoder.'+source]
 encoder.load_state_dict(mapped,strict=True);torch.manual_seed(13);model=TaskAwareFusionModel(encoder,768,mode='cls');head=torch.load(paths['head_template'],map_location='cpu',weights_only=True);r=model.load_state_dict(head,strict=False)
 if set(r.missing_keys)!={k for k in model.state_dict() if k.startswith('encoder.')} or r.unexpected_keys:raise ValueError('Original fresh seed13 head template required')
 if sum(p.numel() for p in model.parameters())!=53771526:raise ValueError('Compacted graph count mismatch')
 model.to(device).train();random.seed(13);np.random.seed(13);torch.manual_seed(13);teacher=np.load(paths['teacher_logits'],mmap_mode='r',allow_pickle=False);teacher_idx=np.load(paths['teacher_indices'],allow_pickle=False)
 if not np.array_equal(teacher_idx,indices) or teacher.shape!=(193124,2):raise ValueError('TRAIN teacher order')
 from .augmentation_protocol import epoch_slots,window_samples,schedule_hash
 orders=[epoch_slots(len(feats),13,e) for e in range(4)]
 if [schedule_hash(v,indices.tolist()) for v in orders]!=job['schedule_hashes']:raise ValueError('Original fixed schedules')
 opt=torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=.01);sched=get_linear_schedule_with_warmup(opt,482,8048);steps=calls=0;w=torch.tensor(WEIGHTS,device=device,dtype=torch.float32)
 def collator(fs):return collate_packed(fs,1)
 # workers0 portable baseline: exact sample/window math,NOT promised bit-identical old worker4 timing.
 for e,order in enumerate(orders):
  opt.zero_grad(set_to_none=True)
  for mb,start in enumerate(range(0,len(order),12)):
   slots=order[start:start+12];b=collator([feats[s] for s in slots]);y=b.labels.to(device)
   with torch.autocast('cuda',dtype=torch.bfloat16,enabled=device.type=='cuda'):logs,_=model(b.input_ids.to(device),b.attention_mask.to(device),b.code_mask.to(device),b.comment_mask.to(device),b.has_comment.to(device))
   logs=logs.float();t=torch.from_numpy(np.array(teacher[slots],copy=True)).to(device)/2;s=logs/2;prob=torch.sigmoid(t);hard=torch.nn.functional.binary_cross_entropy_with_logits(logs,y,pos_weight=w,reduction='none');kl=4*(prob*(torch.nn.functional.logsigmoid(t)-torch.nn.functional.logsigmoid(s))+(1-prob)*(torch.nn.functional.logsigmoid(-t)-torch.nn.functional.logsigmoid(-s)));loss=(.5*hard+.5*kl).mean(1);(loss.sum()/window_samples(mb,len(feats),12,8)).backward();calls+=1
   if (mb+1)%8==0 or start+12>=len(order):torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step();sched.step();opt.zero_grad(set_to_none=True);steps+=1
  if e in (1,3):torch.save(model.state_dict(),out/f'checkpoint-epoch{e+1}.pt')
 if steps!=8048:raise ValueError('Fixed epoch4 update budget')
 return {'output':str(out),'updates':steps,'student_forwards_backwards':calls,'selection':'epoch4 fixed;epoch2diagnostic only','NOT_original_checkpoint_hash_guarantee':True,'portable_workers0_difference_disclosed':True}
