"""No-model evaluation of saved JSONL; never selects thresholds."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.saved_evaluation import evaluate,read_rows
from src.source_guard import verify_release

def main():
    verify_release()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions',required=True);p.add_argument('--gold',required=True);p.add_argument('--output',required=True)
    args=p.parse_args()
    report=evaluate(read_rows(args.predictions),read_rows(args.gold))
    with Path(args.output).open('x',encoding='utf-8') as f:
        json.dump(report,f,indent=2,allow_nan=False)
        f.write('\n')
    print(json.dumps({'rows':report['rows'],'model_calls':0,'threshold_searches':0}))

if __name__=='__main__':main()
