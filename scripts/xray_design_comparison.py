"""Replay six frozen validation models and pairwise error complementarity."""
import json
import numpy as np
import pandas as pd
from xray_config import ROOT
from xray_eval import evaluate,match
from xray_selected_common import REPORT,RUNS
from xray_recovery import atomic_json,sha256


def main():
    latest=json.loads((ROOT/'reports/design_review_20261004/latest_validation_comparison.json').read_text())
    assert latest['status']=='complete'
    sources=[dict(name=r['name'],path=r['validation_csv'],thr=r['threshold'],run=r['run']) for r in latest['models']]
    frozen=json.loads((REPORT/'frozen_selection.json').read_text())['runs']
    for name,run in zip(['Faster R-CNN','RF-DETR-S','D-FINE-S'],RUNS[1:]):
        sources.append(dict(name=name,run=run,path=str((REPORT/'diagnostics'/f'{run}_benchmark_validation.csv').relative_to(ROOT)),thr=frozen[run]['thresholds']['iou50']))
    models=[];detected={};cases=[]
    for s in sources:
        m,_,gt=evaluate(ROOT/s['path'],'val',thr=s['thr'],matching='iou50')
        detected[s['name']]=gt.detected.to_numpy(bool)
        raw=pd.read_csv(ROOT/s['path']);kept=raw[raw.score>=s['thr']]
        tagged,_=match(kept,gt,iou_thr=.5,iou_only=True)
        fps=tagged[tagged.hit<0][['stem','cx','cy','w','h','score']].to_dict(orient='records')
        fns=gt[~gt.detected][['stem','cx','cy','w','h']].to_dict(orient='records')
        models.append(dict(**s,sha256=sha256(ROOT/s['path']),metrics={k:m[k] for k in ['tp','fp','fn','f1','precision','recall','ap']},fp_boxes=fps,fn_boxes=fns))
    pairs=[]
    for a in sources[:3]:
        for b in sources[3:]:
            da,db=detected[a['name']],detected[b['name']]
            pairs.append(dict(first=a['name'],second=b['name'],first_fn_covered=int((~da&db).sum()),shared_fn=int((~da&~db).sum()),first_correct_second_wrong=int((da&~db).sum())))
    out=dict(validation_images=107,gt=243,models=models,pairs=pairs,
             no_operational_f1=True,oracle_analysis_not_route=True,
             physical_omission_not_proven_by_iou=True)
    atomic_json(ROOT/'reports/design_review_20261004/six_model_validation.json',out)
    print(json.dumps(dict(models=[dict(name=r['name'],**r['metrics']) for r in models],pairs=pairs),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
