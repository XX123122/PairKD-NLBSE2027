import argparse,json
from pathlib import Path
from src.tool import read_jsonl,infer_rows,load_candidate,score,sha
def main():
 from src.source_guard import verify_release
 verify_release()
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
 q=sub.add_parser('infer');q.add_argument('--input',required=True);q.add_argument('--output',required=True);q.add_argument('--assets-root',required=True);q.add_argument('--device',default='cpu');q.add_argument('--dry-run',action='store_true')
 q=sub.add_parser('verify-assets');q.add_argument('--assets-root',required=True)
 q=sub.add_parser('reference-score');q.add_argument('--mean-f1',type=float,required=True);q.add_argument('--seconds',type=float,required=True);q.add_argument('--gflops',type=float,required=True)
 for name in ('train','cache','tune'):
  q=sub.add_parser(name);q.add_argument('--job',required=True);q.add_argument('--execute',action='store_true')
 a=p.parse_args()
 if a.command=='infer':
  rows=read_jsonl(a.input)
  if a.dry_run:print(json.dumps({'rows':len(rows),'indices':[r['index'] for r in rows],'model_calls':0,'schema':'index,Function only'}));return
  if not rows:result=[]
  else:result=infer_rows(rows,load_candidate(a.assets_root,a.device))
  with Path(a.output).open('x',encoding='utf8') as f:
   for row in result:f.write(json.dumps(row,allow_nan=False)+'\n')
 elif a.command=='verify-assets':
  m=json.loads((Path(a.assets_root)/'candidate_manifest.json').read_text());s=m['candidates']['PKP2-seed13'];cp=Path(a.assets_root)/'assets/checkpoints'/s['package_checkpoint_name'];assert sha(cp)==s['checkpoint_sha256'];print(json.dumps({'verified':True,'checkpoint_sha256':sha(cp),'model_initialized':False}))
 elif a.command=='reference-score':print(json.dumps({'provisional_reference_score':score(a.mean_f1,a.seconds,a.gflops),'official_final_or_hidden':False}))
 else:
  from src.reproduce import run_job
  print(json.dumps(run_job(a.command,a.job,execute=a.execute),default=str))
if __name__=='__main__':main()
