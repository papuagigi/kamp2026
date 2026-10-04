"""Describe latest model tests with validation-frozen thresholds; no selection."""
import json
from xray_config import ROOT
from xray_eval import evaluate,match
from xray_recovery import sha256,atomic_json
import pandas as pd


def main():
    base=ROOT/'reports/design_review_20261004'
    frozen=json.loads((base/'proposal_frozen.json').read_text());rows=[]
    for run in frozen['test_runs']:
        s=frozen['specs'][run];directory=base/'inference'/run/'test'
        e=json.loads((directory/'execution.json').read_text());assert e['status']=='complete'
        policy=json.loads((directory/'policy.json').read_text())
        assert policy['frozen_proposal_sha256']==sha256(base/'proposal_frozen.json')
        path=directory/'preds_repeat0.csv';assert sha256(path)==e['csv_sha256'][path.name]
        m,_,gt=evaluate(path,'test',thr=s['threshold'],matching='iou50')
        raw=pd.read_csv(path);selected=raw[raw.score>=s['threshold']]
        tagged,_=match(selected,gt,iou_thr=.5,iou_only=True)
        rows.append(dict(run=run,threshold=s['threshold'],metrics={k:m[k] for k in ['tp','fp','fn','f1','precision','recall','ap']},
            fn_boxes=gt[~gt.detected][['stem','cx','cy','w','h']].to_dict(orient='records'),
            fp_boxes=tagged[tagged.hit<0][['stem','cx','cy','w','h','score']].to_dict(orient='records'),
            predictions=str(path.relative_to(ROOT)),sha256=sha256(path),
            condition_recall=m['recall_by_condition']))
    atomic_json(base/'latest_test_descriptive.json',dict(models=rows,no_selection_change=True,
        prior_test_exposure=True,new_independent_holdout=False,images=97,gt=225))
    print(json.dumps([dict(run=r['run'],threshold=r['threshold'],**r['metrics']) for r in rows],ensure_ascii=False,indent=2))


if __name__=='__main__':main()
