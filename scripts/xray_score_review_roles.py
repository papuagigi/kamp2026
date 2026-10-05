"""Score existing role predictions through xray_eval; keep route diagnostics separate."""
import argparse
import json
import pandas as pd
from xray_config import ROOT
from xray_eval import evaluate
from xray_cascade_runtime import agreement
from xray_recovery import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);a=p.parse_args()
    out=ROOT/'reports/review_roles_20261005'/a.name
    policy=json.loads((out/'policy.json').read_text());execution=json.loads((out/'execution.json').read_text())
    first=pd.read_csv(ROOT/'runs/v2_common20_yolo_mps_20261004/validation/epoch_020.csv')
    first=first[first.score>=policy['first_threshold']]
    results={}
    for role in ['candidate','miss']:
        file=out/f'{role}.csv'
        if not file.exists():continue
        fixed,_,_=evaluate(file,'val',thr=policy['review_frozen_threshold'],matching='iou50')
        tuned,_,_=evaluate(file,'val',matching='iou50')
        b=pd.read_csv(file);b=b[b.score>=policy['review_frozen_threshold']]
        diagnostics=[]
        for entry in execution['routes']:
            stem=entry['stem'];x=first[first.stem==stem];y=b[b.stem==stem]
            diagnostics.append(dict(stem=stem,routed_here=(entry['route']==role),
                first_boxes=len(x),review_boxes=len(y),comparison=agreement(x,y)))
        results[role]=dict(fixed_threshold=fixed,exploratory_best_f1=tuned,diagnostics=diagnostics,
            routed_images=sum(x['routed_here'] for x in diagnostics),
            scope='All validation, with forced miss-mode evaluation; no synthetic normal and no new test.',
            no_pass=True,operational_fn_unverified=True)
    atomic_json(out/'evaluation.json',results)
    print(json.dumps({k:dict(fixed_FN=v['fixed_threshold']['fn'],fixed_FP=v['fixed_threshold']['fp'],
        best_FN=v['exploratory_best_f1']['fn'],best_FP=v['exploratory_best_f1']['fp'],
        routed=v['routed_images']) for k,v in results.items()}),flush=True)

if __name__=='__main__':main()
