"""Re-score completed latest YOLO selected checkpoints' saved validation CSVs."""
import argparse
import json
from pathlib import Path
import numpy as np
from xray_config import ROOT
from xray_eval import evaluate
from xray_recovery import atomic_json,sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');a=p.parse_args()
    runs=[('YOLOv8n','v2_common20_yolo_mps_20261004'),
          ('YOLO11n','v2_latest20_yolo11n_guard_mps_20261004'),
          ('YOLO26n','v2_latest20_yolo26n_guard_mps_20261004')]
    rows=[];pending=[]
    for name,run in runs:
        directory=ROOT/'runs'/run
        if not (directory/'execution.json').exists():pending.append(name);continue
        e=json.loads((directory/'execution.json').read_text())
        if e['status']!='complete':pending.append(name);continue
        selection=json.loads((directory/'selection.json').read_text());assert selection['completed_epochs']==20
        assert sha256(ROOT/selection['checkpoint'])==selection['checkpoint_sha256']
        epoch=selection['best_epoch'];pred=directory/'validation'/f'epoch_{epoch:03d}.csv'
        record=json.loads((directory/'validation'/f'epoch_{epoch:03d}.json').read_text())
        assert sha256(pred)==record['predictions_sha256']
        threshold=selection['metrics']['iou50']['thr']
        metrics,_,gt=evaluate(pred,'val',thr=threshold,matching='iou50')
        for k in ['tp','fp','fn','f1','precision','recall']:
            assert np.isclose(metrics[k],selection['metrics']['iou50'][k],atol=1e-12,rtol=0)
        invalid={}
        for ep in range(1,21):
            r=json.loads((directory/'validation'/f'epoch_{ep:03d}.json').read_text())
            invalid[str(ep)]=r.get('invalid_predictions',0)
            assert sha256(ROOT/r['checkpoint'])==r['checkpoint_sha256']
        rows.append(dict(name=name,run=run,epochs=20,selected_epoch=epoch,
            checkpoint=selection['checkpoint'],checkpoint_sha256=selection['checkpoint_sha256'],
            validation_csv=str(pred.relative_to(ROOT)),predictions_sha256=sha256(pred),
            threshold=threshold,metrics={k:metrics[k] for k in ['tp','fp','fn','f1','precision','recall','ap']},
            training_seconds=e['seconds'],end2end=e.get('end2end',False),
            invalid_boxes_by_epoch=invalid,selected_invalid_predictions=record.get('invalid_predictions',0)))
    if a.final:assert not pending,('Training incomplete',pending)
    result=dict(status='complete' if not pending else 'partial',pending=pending,models=rows,
        split='val',metric='official IoU50 matching, validation-selected threshold and epoch',
        no_test_evaluation=True,accuracy_difference_significance_established=False,
        latest_model_recipe='common AdamW setting, not native-recipe optimum',
        inference_latency_benchmarked_for_latest=False,
        limits=['Repeated validation, one seed, correlated capture groups, no real normal products.',
                'Higher validation F1 does not establish lower physical omission or independent test superiority.',
                'YOLO26 uses the observed end2end=false output path; do not claim NMS-free execution.',
                'Negative geometry quarantined in latest trials; raw outputs retained and selected checkpoint validity reported.'])
    atomic_json(ROOT/'reports/design_review_20261004/latest_validation_comparison.json',result)
    print(json.dumps(dict(status=result['status'],pending=pending,models=[{k:r[k] for k in ['name','selected_epoch','metrics','selected_invalid_predictions']} for r in rows]),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
