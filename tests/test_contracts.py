"""Synthetic boundary tests. No Torch, Transformer, checkpoint or CUDA use."""
import copy
import importlib.abc
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

class DenyModelImports(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'torch','transformers'}:
            raise AssertionError('Model import forbidden during contract checks')
sys.meta_path.insert(0,DenyModelImports())
from src.tool import normalize,postprocess,infer_rows,THRESHOLDS,score
from src.saved_evaluation import evaluate
from src.source_guard import verify_release

class Contracts(unittest.TestCase):
    def test_release_integrity(self):
        self.assertTrue(verify_release()['verified'])

    def test_order_and_duplicates(self):
        p=postprocess([8,8,2],[[.8,.99],[.1,.2],[.6,.98]])
        self.assertEqual([x['index'] for x in p],[8,8,2])
        self.assertEqual([x['pred_weakness'] for x in p],[1,0,1])

    def test_exact_float32_boundary(self):
        r=postprocess([0,1],np.stack([THRESHOLDS,np.nextafter(THRESHOLDS,np.float32(0))]))
        self.assertEqual([(x['pred_weakness'],x['pred_mat']) for x in r],[(1,1),(0,0)])

    def test_invalid_probabilities(self):
        for p in ([[np.nan,.9]],[[1.1,.5]],[[-.1,0]],[[.8]],[[np.inf,0]]):
            with self.subTest(p=p),self.assertRaises(ValueError):postprocess([1],p)

    def test_input_schema(self):
        for row in ({'index':1,'Function':'','Label':[0,1]}, {'index':True,'Function':''},
                    {'index':1,'Function':None},{'index':1,'Function':'','Filepath':'x'}):
            with self.subTest(row=row),self.assertRaises(ValueError):normalize([row])
        self.assertEqual(normalize(['']),[{'index':0,'Function':''}])

    def test_empty_never_calls_backend(self):
        class Deny:
            def probabilities(self,*a,**kw):raise AssertionError('Empty input invoked backend')
        self.assertEqual(infer_rows([],Deny()),[])

    def test_stub_tail_batch(self):
        class Stub:
            def probabilities(self,functions,batch_size):
                assert batch_size==600
                return np.tile([.8,.99],(len(functions),1)),[],None
        rows=[{'index':i%7,'Function':'synthetic'} for i in range(601)]
        r=infer_rows(rows,Stub())
        self.assertEqual(len(r),601);self.assertEqual([x['index'] for x in r],[x['index'] for x in rows])

    def test_evaluation_known_confusions(self):
        pred=postprocess([1,1,3],[[.8,.99],[.7,.2],[.1,.98]])
        gold=[{'index':1,'Label':[1,1]},{'index':1,'Label':[0,0]},{'index':3,'Label':[1,0]}]
        report=evaluate(pred,gold)
        self.assertEqual(report['frozen']['TP'],[1,1]);self.assertEqual(report['frozen']['FP'],[1,1])
        self.assertEqual(report['frozen']['FN'],[1,0]);self.assertAlmostEqual(report['frozen']['mean_F1'],7/12)

    def test_evaluation_rejects_wrong_policy_and_order(self):
        pred=postprocess([3],[[.8,.99]]);gold=[{'index':3,'Label':[1,1]}]
        bad=copy.deepcopy(pred);bad[0]['pred_mat']=0
        with self.assertRaises(ValueError):evaluate(bad,gold)
        with self.assertRaises(ValueError):evaluate(pred,[{'index':2,'Label':[1,1]}])
        with self.assertRaises(ValueError):evaluate(pred,[{'index':3,'Label':[True,1]}])

    def test_zero_positive_policy(self):
        r=evaluate(postprocess([0],[[.1,.1]]),[{'index':0,'Label':[0,0]}])
        self.assertEqual(r['frozen']['F1'],[0.,0.])

    def test_score_sessions_separate(self):
        self.assertAlmostEqual(score(.9084194916924127,33.16337677800004,444468.6284385284),.9087877777639457)
        self.assertAlmostEqual(score(.9084194916924127,38.76940846443176,444468.6284385284),.9076665714266594)

    def test_cli_empty_and_dryrun(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);input_=p/'empty.jsonl';input_.write_text('')
            c=subprocess.run([sys.executable,str(ROOT/'cli.py'),'infer','--input',str(input_),'--output',str(p/'out'),
                              '--assets-root',str(p/'absent')],capture_output=True,text=True,check=True)
            self.assertEqual((p/'out').read_text(),'')
            c=subprocess.run([sys.executable,str(ROOT/'cli.py'),'infer','--input',str(ROOT/'examples/functions.jsonl'),
                              '--output',str(p/'dry'),'--assets-root',str(p/'absent'),'--dry-run'],capture_output=True,text=True,check=True)
            self.assertEqual(json.loads(c.stdout)['model_calls'],0);self.assertFalse((p/'dry').exists())

if __name__=='__main__':unittest.main()
