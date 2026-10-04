"""Describe frozen test localization errors without changing their TP/FP/FN labels."""
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'reports/common_epoch_20261004/evaluation'
OUT = BASE / 'diagnostics'


def main():
    errors = pd.read_csv(OUT / 'errors.csv')
    errors = errors[(errors.split == 'test') & (errors.error == 'FN')]
    rows, sources = [], [OUT / 'errors.csv', BASE / 'comparison.csv']
    for _, gt in errors.iterrows():
        if gt.category != 'nearby_selected_box_iou_below_0.5':
            continue
        path = ROOT / f'reports/preds_{gt.run}_test.csv'
        pred = pd.read_csv(path).iloc[int(gt.companion_pred_index)]
        assert pred.stem == gt.stem and pred.score >= gt.threshold
        sources.append(path)
        rows.append(dict(
            run=gt.run, stem=gt.stem, gt_index=int(gt.gt_index),
            iou=gt.companion_iou, score=pred.score,
            gt_center_inside_prediction=bool(
                abs(pred.cx-gt.cx) <= pred.w/2 and abs(pred.cy-gt.cy) <= pred.h/2),
            prediction_to_gt_area=pred.w*pred.h/(gt.w*gt.h),
            center_distance_px=gt.companion_distance_px))
    details = pd.DataFrame(rows)
    details.to_csv(OUT / 'containment_review.csv', index=False)
    comparison = pd.read_csv(BASE / 'comparison.csv').set_index('model')
    report = dict(
        scope='Exploratory geometry diagnosis. Not a replacement scorer or new TP labels.',
        test_fn_events=len(errors), nearby_localization_events=len(details),
        unique_localization_gt=len(details.drop_duplicates(['stem', 'gt_index'])),
        gt_center_inside_prediction=int(details.gt_center_inside_prediction.sum()),
        center_distance_range=details.center_distance_px.agg(['min', 'max']).to_dict(),
        area_ratio_range=details.prediction_to_gt_area.agg(['min', 'max']).to_dict(),
        yolo_minus_faster_f1=float(comparison.loc['YOLOv8n', 'f1']-comparison.loc['Faster R-CNN', 'f1']),
        sources={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(set(sources))},
        limitation='A TXT box center is not a pixel-level segmentation of the actual defect. '
                   'Visual inspection is recorded separately and cannot be automated by this calculation.')
    (OUT / 'containment_review.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'sources'}, indent=2))


if __name__ == '__main__':
    main()
