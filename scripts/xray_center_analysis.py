#!/usr/bin/env python3
"""Frozen, post-hoc center evaluation. All metric calculation uses xray_eval."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from xray_config import ROOT, DATA
from xray_eval import evaluate, load_gt, match, iou

OUT = ROOT / 'reports/center_evaluation_20261007'


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    assert hashlib.md5((DATA / 'split.csv').read_bytes()).hexdigest() == '8e58184ae24dfd1b48da2e9dfd88fedc'
    rows, targets, extras, conditions, photos, audits = [], [], [], [], [], []
    seen = {}
    for spec in protocol['models']:
        model, thr = spec['model'], spec['thr']
        for split in ['val', 'test']:
            source = spec['files'][split]
            path = ROOT / source['path']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == source['sha256']
            raw = pd.read_csv(path)
            gts, man = load_gt(split)
            accepted = raw[raw.score >= thr]
            for radius in protocol['radii_px']:
                result, _, gt = evaluate(path, split, thr=thr, matching=f'center{radius}')
                result['pred_csv'] = source['path']
                result['protocol'] = 'reports/center_evaluation_20261007/protocol.json'
                dump(f'{model}_{split}_center{radius}.json', result)
                row = {k: result[k] for k in ['n_images', 'n_gt', 'tp', 'fp', 'fn', 'precision', 'recall', 'f1']}
                row.update(model=model, split=split, radius_px=radius, thr=thr,
                           median_px=result['center_error']['median_px'], p95_px=result['center_error']['p95_px'])
                rows.append(row)
                matched, _ = match(accepted, gts, dist_thr=radius, iou_thr=None, center_only=True)
                assert int((matched.hit >= 0).sum()) == result['tp']
                seen[model, split, radius] = gt.detected.to_numpy()
                for c in ['machine', 'size_bin', 'contrast_bin', 'edge_bin']:
                    for name, subset in gt.groupby(c, observed=True):
                        conditions.append(dict(model=model, split=split, radius_px=radius, condition=c,
                                               group=str(name), n=len(subset), fn=int((~subset.detected).sum())))
                for gi, g in gt.iterrows():
                    local = raw[raw.stem == g.stem].copy()
                    local['distance'] = np.hypot(local.cx - g.cx, local.cy - g.cy)
                    near_acc = local[local.score >= thr].sort_values('distance')
                    near_low = local[(local.score < thr) & (local.distance <= radius)].sort_values('score', ascending=False)
                    got = matched[matched.hit == gi]
                    diag, error, score, overlap = 'matched', None, None, None
                    if len(got):
                        p = got.iloc[0]
                        error, score = float(p.center_error_px), float(p.score)
                        overlap = iou((p.cx,p.cy,p.w,p.h), (g.cx,g.cy,g.w,g.h))
                        if overlap < .5:
                            diag = 'center_success_overlap_below_05'
                    elif len(near_acc) and near_acc.iloc[0].distance <= radius:
                        diag = 'near_prediction_already_assigned'
                    elif len(near_acc) and near_acc.iloc[0].distance <= 8:
                        diag = 'accepted_center_outside_radius_within_8px'
                    elif len(near_low):
                        diag = 'near_prediction_below_threshold'
                    elif (local.distance <= 8).any():
                        diag = 'stored_center_outside_radius_within_8px'
                    else:
                        diag = 'no_stored_prediction_within_8px'
                    nearest = local.sort_values('distance').iloc[0] if len(local) else None
                    nearest_acc = near_acc.iloc[0] if len(near_acc) else None
                    best_below = near_low.iloc[0] if len(near_low) else None
                    targets.append(dict(model=model, split=split, radius_px=radius, gt_index=int(gi),
                        stem=g.stem, cx=g.cx, cy=g.cy, w=g.w, h=g.h, machine=g.machine,
                        size=g['size'], contrast=g.contrast, edge=g.edge, detected=bool(g.detected),
                        diagnostic=diag, center_error_px=error, matched_score=score, iou_internal=overlap,
                        nearest_raw_distance=None if nearest is None else float(nearest.distance),
                        nearest_raw_score=None if nearest is None else float(nearest.score),
                        nearest_accepted_distance=None if nearest_acc is None else float(nearest_acc.distance),
                        nearby_below_score=None if best_below is None else float(best_below.score)))
                for p in matched[matched.hit < 0].itertuples():
                    extras.append(dict(model=model, split=split, radius_px=radius, stem=p.stem,
                                       cx=p.cx, cy=p.cy, w=p.w, h=p.h, score=p.score))
                print(model, split, radius, result['tp'], result['fp'], result['fn'], flush=True)
            for stem in man.stem:
                photos.append(dict(model=model, split=split, stem=stem,
                                   accepted_predictions=int((accepted.stem == stem).sum()),
                                   no_alarm=bool(not (accepted.stem == stem).any())))
    # Old modes must reproduce their saved metrics exactly at their original thresholds.
    old = pd.read_csv(ROOT / 'reports/direct_analysis_20261007/rescore_summary.csv')
    by_model = {s['model']: s for s in protocol['models']}
    for r in old.itertuples():
        res, _, _ = evaluate(ROOT / by_model[r.model]['files'][r.split]['path'], r.split, thr=r.thr, matching=r.matching)
        diffs = {k: float(res[k]) - float(getattr(r,k)) for k in ['tp','fp','fn','precision','recall','f1','ap']}
        assert all(abs(v) < 1e-12 for v in diffs.values()), diffs
        audits.append(dict(model=r.model, split=r.split, matching=r.matching, max_difference=max(map(abs,diffs.values()))))
    pairs = []
    for split in ['val','test']:
        for radius in protocol['radii_px']:
            a,b = seen['YOLO11s',split,radius],seen['RF-DETR-S',split,radius]
            pairs.append(dict(split=split, radius_px=radius, n_gt=len(a), both=int((a&b).sum()),
                yolo_only=int((a&~b).sum()), rf_only=int((~a&b).sum()), neither=int((~a&~b).sum()),
                either=int((a|b).sum()), scope='GT-based coverage only; not operational fused F1'))
    for name, records in [('summary',rows),('targets',targets),('extra_predictions',extras),
                          ('conditions',conditions),('photos',photos),('complementarity',pairs)]:
        pd.DataFrame(records).to_csv(OUT / f'{name}.csv', index=False)
    dump('checks.json', dict(status='passed', source_hashes_verified=6, original_modes_regressions=audits,
                            scoring_implementation='scripts/xray_eval.py', protocol_sha256=hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest(),
                            no_training=True, no_new_inference=True, no_threshold_tuning=True))
    print('All scoring and source checks passed.', flush=True)


if __name__ == '__main__':
    main()
