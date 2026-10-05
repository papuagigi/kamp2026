"""Summarize completed role experiments; never choose a deployment policy."""
import json
import pandas as pd
from xray_config import ROOT
from xray_eval import evaluate
from xray_recovery import atomic_json


def main():
    base=ROOT/'reports/review_roles_20261005'
    rows=[]
    for file in sorted(base.glob('*/evaluation.json')):
        result=json.loads(file.read_text())
        timing=json.loads((file.parent/'execution.json').read_text())
        policy=json.loads((file.parent/'policy.json').read_text())
        for role,score in result.items():
            fixed=score['fixed_threshold'];best=score['exploratory_best_f1']
            rows.append(dict(name=file.parent.name,role=role,review=policy['review'],
                fixed_threshold=fixed['thr'],fixed_f1=fixed['f1'],fixed_tp=fixed['tp'],
                fixed_fp=fixed['fp'],fixed_fn=fixed['fn'],validation_tuned_f1=best['f1'],
                validation_tuned_fp=best['fp'],validation_tuned_fn=best['fn'],
                validation_tuned_threshold=best['thr'],routed_images=score['routed_images'],
                review_mean_ms=timing['means_ms'][role]))
    # A reference from saved full-frame predictions, not newly timed inference.
    native=ROOT/'runs/v2_common20_rfdetr_mps_20261004/validation/epoch_008.csv'
    ref,_,_=evaluate(native,'val',thr=.70370466,matching='iou50')
    pd.DataFrame(rows).to_csv(base/'summary.csv',index=False)
    atomic_json(base/'summary.json',dict(rows=rows,fullframe_rf_reference={k:ref[k] for k in ['f1','tp','fp','fn','thr']},
        scope='Review detector alone; not combined-system F1 or product escape rate.',
        no_test=True,real_normal_products=0,all_first_predictions_positive=True,
        final_deployment_selection=False,source='saved CSV scored by xray_eval iou50'))
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':main()
