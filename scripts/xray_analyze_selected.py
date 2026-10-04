"""Audit selected-checkpoint errors and test conditions, using the canonical scorer."""
import json
import numpy as np
import pandas as pd
from xray_selected_common import RUNS, REPORT, freeze
from xray_config import ROOT
from xray_eval import evaluate, load_gt
from xray_condition_audit import features, distribution, FEATURES
from xray_recovery import atomic_json
import xray_analyze_errors as audit

def main():
    freeze()
    audit.RUNS=RUNS; audit.REPORT=REPORT; audit.OUT=REPORT/'diagnostics'; audit.main()
    out=audit.OUT
    train, tm=load_gt('train'); train=features(train,tm,'train')
    cuts={k:[float(train[k].quantile(.25)),float(train[k].quantile(.75))] for k in FEATURES}
    rows=[]; summaries=[]; details=[]
    for run in RUNS:
        recorded=json.loads((REPORT/f'{run}_evaluation.json').read_text())
        for split in ['val','test']:
            saved=recorded['metrics']['iou50'][split]
            pred=ROOT/'reports'/f'preds_{run}_{split}.csv'
            metric,_,gt=evaluate(pred,split,thr=saved['thr'],matching='iou50')
            _,manifest=load_gt(split); gt=features(gt,manifest,split)
            raw=pd.read_csv(pred); alarms=set(raw.loc[raw.score>=saved['thr'],'stem'])
            positives=set(gt.stem)
            summaries.append(dict(run=run,split=split,images=len(manifest),boxes=len(gt),
                positive_images_without_any_alarm=len(positives-alarms),
                positive_images_with_any_correct_localization=int(gt.groupby('stem').detected.any().sum()),
                positive_images_with_all_boxes_found=int(gt.groupby('stem').detected.all().sum()),
                real_normal_images=len(manifest)-len(positives)))
            for feature,(lo,hi) in cuts.items():
                groups=np.where(gt[feature]<=lo,'low_le_train_q25',np.where(gt[feature]>hi,'high_gt_train_q75','middle'))
                for label in ['low_le_train_q25','middle','high_gt_train_q75']:
                    sub=gt[groups==label]
                    rows.append(dict(run=run,split=split,feature=feature,group=label,q25=lo,q75=hi,
                        n_gt=len(sub),n_images=int(sub.stem.nunique()),n_bursts=int(sub.burst_id.nunique()),
                        tp=int(sub.detected.sum()),fn=int((~sub.detected).sum()),
                        miss_rate=float((~sub.detected).mean()) if len(sub) else None))
            details.append(gt.assign(run=run,split=split))
    pd.DataFrame(rows).to_csv(out/'conditions.csv',index=False)
    pd.concat(details,ignore_index=True).to_csv(out/'box_features.csv',index=False)
    atomic_json(out/'condition_summary.json',dict(cuts_source='official training boxes only',cuts=cuts,
        train_distribution=distribution(train),image_outcomes=summaries,
        caveats=['Exploratory associations, not causal effects or model-selection evidence.',
        'Boxes within a capture group are correlated; tiny subsets do not establish general superiority.',
        'Edge is an approximate product-mask distance; aspect ratio describes TXT boxes, not segmented shape.',
        'Local grayscale/contrast/texture are image proxies; GT-derived conditions cannot themselves trigger live review.',
        'No real normal product images: no normal-product false-alarm rate or factory reinspection capacity estimate.',
        'A product with an unmatched prediction can still alarm; any alarm does not imply correct localization.',
        'The test split has been viewed in earlier experiments. No new threshold or checkpoint selection uses these test results.']))
    print(json.dumps(summaries,indent=2))

if __name__=='__main__':main()
