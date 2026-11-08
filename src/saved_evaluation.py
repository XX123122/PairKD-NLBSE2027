"""Evaluate durable student probabilities against separately supplied ordered gold."""
import json
from pathlib import Path
import numpy as np
from .tool import postprocess, THRESHOLDS


def read_rows(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate(predictions, gold):
    if not predictions or len(predictions) != len(gold):
        raise ValueError('Nonempty predictions and gold must have equal lengths')
    probabilities=[]; labels=[]; indices=[]
    fields={'index','prob_weakness','prob_mat','pred_weakness','pred_mat'}
    for p,y in zip(predictions,gold):
        if set(p)!=fields or set(y)!={'index','Label'}:
            raise ValueError('Incorrect prediction/gold schema')
        if type(p['index']) is not int or type(y['index']) is not int or p['index']!=y['index']:
            raise ValueError('Ordered indices must match, including duplicates')
        target=y['Label']
        if not isinstance(target,list) or len(target)!=2 or any(type(x) is not int or x not in (0,1) for x in target):
            raise ValueError('Label must contain two integer binary targets')
        if any(type(p[k]) not in (int,float) for k in ('prob_weakness','prob_mat')):
            raise ValueError('Probabilities must be numeric')
        if any(type(p[k]) is not int or p[k] not in (0,1) for k in ('pred_weakness','pred_mat')):
            raise ValueError('Saved predictions must be binary integers')
        probabilities.append([p['prob_weakness'],p['prob_mat']]);labels.append(target);indices.append(p['index'])
    probs=np.asarray(probabilities,np.float32)
    expected=postprocess(indices,probs)
    if any(p['pred_weakness']!=e['pred_weakness'] or p['pred_mat']!=e['pred_mat'] for p,e in zip(predictions,expected)):
        raise ValueError('Saved predictions disagree with frozen float32 policy')
    y=np.asarray(labels,bool)
    report={'rows':len(gold),'task_order':['Weakness','MAT'],'zero_division':0,'model_calls':0,'threshold_searches':0}
    for name,t in [('frozen',THRESHOLDS),('fixed_0.5',np.asarray([.5,.5],np.float32))]:
        q=probs>=t
        tp=(q&y).sum(0);fp=(q&~y).sum(0);fn=(~q&y).sum(0);tn=(~q&~y).sum(0)
        def ratio(a,b):
            return np.divide(a,b,out=np.zeros(2,float),where=b!=0)
        f1=ratio(2*tp,2*tp+fp+fn)
        report[name]={'thresholds':t.tolist(),'comparator':'>=',
                      'TP':tp.tolist(),'FP':fp.tolist(),'FN':fn.tolist(),'TN':tn.tolist(),
                      'precision':ratio(tp,tp+fp).tolist(),'recall':ratio(tp,tp+fn).tolist(),
                      'F1':f1.tolist(),'mean_F1':float(f1.mean())}
    return report
