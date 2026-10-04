"""Paired capture-group bootstrap for completed YOLO validation comparisons.

Conditional on already chosen checkpoints/thresholds; not model-selection-adjusted
or an independent held-out confidence interval.
"""
import json
import numpy as np
import pandas as pd
from xray_config import ROOT
from xray_eval import load_gt,match
from xray_recovery import atomic_json,sha256


def main():
    folder=ROOT/'reports/design_review_20261004'
    source=folder/'latest_validation_comparison.json';comparison=json.loads(source.read_text())
    gt,man=load_gt('val');groups=sorted(man.burst_id.unique());mapping=man.set_index('stem').burst_id
    counts={}
    for model in comparison['models']:
        path=ROOT/model['validation_csv'];assert sha256(path)==model['predictions_sha256']
        raw=pd.read_csv(path);chosen=raw[raw.score>=model['threshold']]
        tagged,hits=match(chosen,gt,iou_thr=.5,iou_only=True)
        found=np.isfinite(hits)
        assert (int((tagged.hit>=0).sum()),int((tagged.hit<0).sum()),int((~found).sum()))==tuple(model['metrics'][k] for k in ['tp','fp','fn'])
        rows=[]
        for group in groups:
            mask=gt.stem.map(mapping).eq(group).to_numpy();pmask=tagged.stem.map(mapping).eq(group)
            rows.append([int((found&mask).sum()),int(((tagged.hit<0)&pmask).sum()),int((~found&mask).sum())])
        counts[model['name']]=np.array(rows)
    rng=np.random.default_rng(0);sample=rng.integers(0,len(groups),size=(2000,len(groups)))
    def f1(values):
        tp,fp,fn=values.T
        return 2*tp/(2*tp+fp+fn)
    base=f1(counts['YOLOv8n'][sample].sum(axis=1));results=[]
    base_score=next(m['metrics']['f1'] for m in comparison['models'] if m['name']=='YOLOv8n')
    for m in comparison['models']:
        if m['name']=='YOLOv8n':continue
        delta=f1(counts[m['name']][sample].sum(axis=1))-base
        results.append(dict(model=m['name'],observed_f1_difference=m['metrics']['f1']-base_score,
                            percentile_95=[float(x) for x in np.quantile(delta,[.025,.975])],
                            contains_zero=bool(np.quantile(delta,.025)<=0<=np.quantile(delta,.975))))
    result=dict(status=comparison['status'],source_sha256=sha256(source),images=len(man),bursts=len(groups),
                resamples=2000,seed=0,unit='capture group, paired across models',results=results,
                limits=['Conditional on selected checkpoints and thresholds; excludes selection uncertainty.',
                        'Repeated validation use and earlier test exposure remain; not a fresh generalization estimate.',
                        'Few capture groups and shared specimen types limit bootstrap validity.',
                        'No statement about physical detection or production safety.'])
    atomic_json(folder/'latest_validation_uncertainty.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
