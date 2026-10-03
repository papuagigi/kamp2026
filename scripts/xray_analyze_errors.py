"""Audit frozen predictions with xray_eval; describe errors without tuning models."""
import hashlib
import json
from collections import Counter

import numpy as np
import pandas as pd

from xray_config import DATA, ROOT, V2_SPLIT_MD5
from xray_eval import evaluate, iou, match

REPORT = ROOT / 'reports/overnight_20261003'
OUT = REPORT / 'diagnostics'
RUNS = ['v2_mix2453_yolo_mps', 'v2_mix2453_faster_cuda', 'v2_mix2453_rfdetr_mps']
BOX = ['cx', 'cy', 'w', 'h']


def geometry(row, reference):
    return float(iou(row[BOX].to_numpy(float), reference[BOX].to_numpy(float))), float(
        np.hypot(row.cx - reference.cx, row.cy - reference.cy))


def nearest(rows, reference):
    if rows.empty:
        return None
    choices = [(int(idx), *geometry(row, reference), float(row.get('score', 0)))
               for idx, row in rows.iterrows()]
    # Geometry only: this does not replace the scorer's one-to-one matches.
    return sorted(choices, key=lambda x: (-x[1], x[2], -x[3]))[0]


def main():
    assert hashlib.md5((DATA / 'split.csv').read_bytes()).hexdigest() == V2_SPLIT_MD5
    OUT.mkdir(parents=True, exist_ok=True)
    errors, gt_rows, prediction_rows, summaries, conditions, sources = [], [], [], [], [], []
    for run in RUNS:
        recorded = json.loads((REPORT / f'{run}_evaluation.json').read_text())
        threshold = recorded['metrics']['iou50']['val']['thr']
        for split in ['val', 'test']:
            path = ROOT / 'reports' / f'preds_{run}_{split}.csv'
            sources.append({'path': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            saved = recorded['metrics']['iou50'][split]
            assert saved['thr'] == threshold
            result, _, gt = evaluate(path, split, thr=threshold, matching='iou50')
            for key in ['tp', 'fp', 'fn', 'f1', 'precision', 'recall', 'n_images', 'n_gt']:
                assert np.isclose(result[key], saved[key]), (run, split, key)
            raw = pd.read_csv(path)
            scored, _ = match(raw, gt, iou_thr=.5, iou_only=True)
            selected = scored[scored.score >= threshold]
            false_preds = selected[selected.hit < 0]
            missed = gt[~gt.detected]
            assert len(false_preds) == result['fp'] and len(missed) == result['fn']
            context = {'run': run, 'split': split, 'threshold': threshold}
            for idx, g in gt.iterrows():
                gt_rows.append(dict(context, gt_index=int(idx), **g.to_dict()))
            for idx, p in selected.iterrows():
                overlap = geometry(p, gt.iloc[int(p.hit)])[0] if p.hit >= 0 else None
                prediction_rows.append(dict(context, pred_index=int(idx), matched_iou=overlap, **p.to_dict()))
            local_errors = []
            for idx, g in missed.iterrows():
                same_fp = false_preds[false_preds.stem == g.stem]
                nearby = nearest(same_fp, g)
                same_low = raw[(raw.stem == g.stem) & (raw.score < threshold)]
                low_match = [(int(i), *geometry(p, g), float(p.score)) for i, p in same_low.iterrows()
                             if geometry(p, g)[0] >= .5]
                best_low = max(low_match, key=lambda x: x[3]) if low_match else None
                if nearby and (nearby[1] >= .3 or nearby[2] <= 8):
                    category, companion = 'nearby_selected_box_iou_below_0.5', nearby
                elif best_low:
                    category, companion = 'matching_box_below_threshold', best_low
                else:
                    category, companion = 'no_matching_box_in_saved_predictions', nearest(raw[raw.stem == g.stem], g)
                row = dict(context, error='FN', stem=g.stem, gt_index=int(idx), pred_index=None,
                           category=category, machine=g.machine, **{k: float(g[k]) for k in BOX + ['size', 'contrast']},
                           companion_pred_index=companion[0] if companion else None,
                           companion_iou=companion[1] if companion else None,
                           companion_distance_px=companion[2] if companion else None,
                           companion_score=companion[3] if companion else None)
                local_errors.append(row)
            for idx, p in false_preds.iterrows():
                same_gt = gt[gt.stem == p.stem]
                near_missed = nearest(missed[missed.stem == p.stem], p)
                near_any = nearest(same_gt, p)
                if near_missed and (near_missed[1] >= .3 or near_missed[2] <= 8):
                    category, companion = 'nearby_missed_gt_iou_below_0.5', near_missed
                elif near_any and near_any[1] >= .5:
                    category, companion = 'duplicate_of_already_matched_gt', near_any
                else:
                    category, companion = 'unmatched_prediction_requires_visual_review', near_any
                row = dict(context, error='FP', stem=p.stem, gt_index=None, pred_index=int(idx), category=category,
                           machine=same_gt.iloc[0].machine, **{k: float(p[k]) for k in BOX + ['score']},
                           companion_gt_index=companion[0] if companion else None,
                           companion_iou=companion[1] if companion else None,
                           companion_distance_px=companion[2] if companion else None)
                local_errors.append(row)
            errors.extend(local_errors)
            assert sum(e['error'] == 'FN' for e in local_errors) == result['fn']
            assert sum(e['error'] == 'FP' for e in local_errors) == result['fp']
            for condition in ['machine', 'size_bin', 'contrast_bin']:
                for value, group in gt.groupby(condition, observed=True):
                    conditions.append(dict(context, condition=condition, value=str(value), gt=len(group),
                                           tp=int(group.detected.sum()), fn=int((~group.detected).sum())))
            summaries.append(dict(context, metrics=result,
                                  error_images=len({e['stem'] for e in local_errors}),
                                  fn_images=len(set(missed.stem)), fp_images=len(set(false_preds.stem)),
                                  images_with_any_selected_box=int(selected.stem.nunique()),
                                  fn_categories=dict(Counter(e['category'] for e in local_errors if e['error'] == 'FN')),
                                  fp_categories=dict(Counter(e['category'] for e in local_errors if e['error'] == 'FP'))))
            print(run, split, result['tp'], result['fp'], result['fn'], summaries[-1]['fn_categories'], flush=True)
    pd.DataFrame(errors).to_csv(OUT / 'errors.csv', index=False)
    pd.DataFrame(gt_rows).to_csv(OUT / 'ground_truth_detail.csv', index=False)
    pd.DataFrame(prediction_rows).to_csv(OUT / 'selected_predictions.csv', index=False)
    pd.DataFrame(conditions).to_csv(OUT / 'recall_conditions.csv', index=False)
    audit = {'scope': 'stored prediction rescore; no retraining, new predictions, or threshold tuning',
             'matching': 'xray_eval.match IoU >= 0.5 score-ordered one-to-one',
             'diagnostic_only': 'Nearby means center distance <=8px OR IoU>=0.3. This is a geometric association, not a changed TP criterion or proof of cause. FN category priority: nearby selected unmatched box, then lower-score IoU>=0.5 box, then no IoU>=0.5 candidate in saved exports. Sources exclude candidates below export score 0.001.',
             'split_md5': V2_SPLIT_MD5, 'sources': sources, 'results': summaries,
             'unique_error_images': len({(e['split'], e['stem']) for e in errors})}
    (OUT / 'error_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False))
    print('Unique error images:', audit['unique_error_images'])


if __name__ == '__main__':
    main()
