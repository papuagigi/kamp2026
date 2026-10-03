"""Evaluate one fixed final checkpoint; only validation chooses thresholds."""
import argparse
import hashlib
import json
import subprocess
import sys
from xray_config import ROOT, V2_SPLIT_MD5
from xray_eval import evaluate, MATCH_POLICIES

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--device',required=True)
    a=p.parse_args();meta=json.loads((ROOT/'runs'/a.run/'execution.json').read_text())
    assert meta['status']=='complete'
    expected_sha=hashlib.sha256((ROOT/meta['checkpoint']).read_bytes()).hexdigest()
    result=dict(run=a.run,execution=meta,metrics={},normal_real_product_images=0,test_used_for_selection=False)
    for split,n in [('val',107),('test',97)]:
        csv=ROOT/'reports'/f'preds_{a.run}_{split}.csv'
        if not csv.exists():
            subprocess.run([sys.executable,'scripts/xray_predict_controlled.py','--run',a.run,'--split',split,'--device',a.device],cwd=ROOT,check=True)
        info=json.loads(csv.with_suffix('.run.json').read_text())
        assert info['images']==n and info['checkpoint_sha256']==expected_sha
        for mode in ['iou50','iou75','legacy']:
            thr=None if split=='val' else result['metrics'][mode]['val']['thr']
            # If validation emits no boxes, use a predeclared reject-all threshold.
            if split=='test' and thr is None:thr=1.000001
            metric,_,_=evaluate(csv,split,thr=thr,matching=mode)
            metric['threshold_policy']='validation_best_f1_frozen_for_test'
            metric['split_md5_expected']=V2_SPLIT_MD5
            result['metrics'].setdefault(mode,{})[split]=metric
            dest=csv.with_name(csv.stem+'_'+MATCH_POLICIES[mode][2]+'_eval.json')
            dest.write_text(json.dumps(metric,ensure_ascii=False,indent=2))
    out=ROOT/'reports/overnight_20261003';out.mkdir(parents=True,exist_ok=True)
    (out/f'{a.run}_evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
