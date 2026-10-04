"""Validation-only replay of referral rules and detector complementarity.

This does not infer, train, recalibrate, or certify production PASS decisions.
Existing xray_eval is the only detection scorer. Oracle recovery is diagnostic,
not a deployed ensemble score. Fixed-threshold replacement is a separate control.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from xray_config import DATA, ROOT, V2_SPLIT_MD5
from xray_condition_audit import FEATURES, features
from xray_eval import PRED_COLUMNS, evaluate, load_gt
from xray_selected_common import REPORT, RUNS

OUT = REPORT / 'cascade_audit'
NAMES = dict(zip(RUNS, ['YOLOv8n', 'Faster R-CNN', 'RF-DETR-S', 'D-FINE-S']))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def referral_sets(raw, stems, threshold, margin, seed, sample_fraction):
    """Uses only model outputs and identifiers. Ground truth is not accepted."""
    stems = set(stems)
    none = stems - set(raw.loc[raw.score >= threshold, 'stem'])
    near = set(raw.loc[(raw.score >= threshold-margin) &
                       (raw.score <= threshold+margin), 'stem'])
    cautious = none | near
    remaining = sorted(stems-cautious)
    rng = np.random.default_rng(seed)
    n = int(np.ceil(sample_fraction * len(remaining)))
    audit = set(rng.choice(remaining, n, replace=False)) if n else set()
    return {'no_detection': none, 'score_margin': cautious,
            'score_margin_plus_sample': cautious | audit, 'all_images_control': stems}


def main():
    assert hashlib.md5((DATA/'split.csv').read_bytes()).hexdigest() == V2_SPLIT_MD5
    OUT.mkdir(parents=True, exist_ok=True)
    frozen_path = REPORT/'frozen_selection.json'
    timing_path = REPORT/'diagnostics/benchmark_mps.json'
    frozen = json.loads(frozen_path.read_text())
    timing = json.loads(timing_path.read_text())
    sources = {run: REPORT/'diagnostics'/f'{run}_benchmark_validation.csv' for run in RUNS}
    for run, path in sources.items():
        assert sha(path) == timing['models'][run]['validation_csv_sha256']
        assert timing['models'][run]['checkpoint_sha256'] == frozen['runs'][run]['checkpoint_sha256']
    policy = dict(version='cascade_validation_replay_v1', split='val', device_of_predictions='mps',
        no_training=True, no_new_inference=True, test_used_in_this_analysis=False,
        earlier_test_exposure=True, validation_reused_for_model_selection=True,
        confidence_margin=.05, random_audit_fraction_of_other_images=.2, seed=0,
        thresholds={r: frozen['runs'][r]['thresholds']['iou50'] for r in RUNS},
        rules=['no_detection', 'score_margin', 'score_margin_plus_sample', 'all_images_control'],
        matching='iou50', first=RUNS[0], seconds=RUNS[1:],
        replacement_control='Keep first output on unrefered photos; replace by second output on referred photos. Each model is filtered at its frozen threshold before xray_eval thr=0. No combined AP.',
        deployment='Retain both predictions and hold disagreement. No implemented production PASS, fusion, or specialist.',
        source_hashes={str(p.relative_to(ROOT)): sha(p) for p in [frozen_path, timing_path, DATA/'split.csv', *sources.values()]})
    pp = OUT/'policy.json'
    if pp.exists():
        assert json.loads(pp.read_text()) == policy, 'Policy/source change requires a new experiment version'
    else:
        save(pp, policy)  # written before evaluation

    gt, manifest = load_gt('val')
    stems = set(manifest.stem)
    assert stems == set(timing['image_stems']) and len(stems) == 107
    scores, raw, kept, found = {}, {}, {}, {}
    for run, path in sources.items():
        raw[run] = pd.read_csv(path)
        assert set(raw[run].stem) <= stems
        threshold = policy['thresholds'][run]
        kept[run] = raw[run][raw[run].score >= threshold].copy()
        result, _, g = evaluate(path, 'val', thr=threshold, matching='iou50')
        assert g[['stem','cx','cy','w','h']].equals(gt[['stem','cx','cy','w','h']])
        expected = timing['models'][run]['validation_on_benchmark_device']
        if 'metrics' in expected:
            expected = expected['metrics']
        for key in ['tp', 'fp', 'fn', 'f1']:
            if key in expected:
                assert np.isclose(result[key], expected[key], rtol=0, atol=1e-12)
        scores[run] = {k:result[k] for k in ['tp','fp','fn','f1','precision','recall','thr','n_images','n_gt']}
        scores[run]['photos_with_no_box'] = len(stems-set(kept[run].stem))
        found[run] = g.detected.to_numpy(bool)

    first = RUNS[0]
    gates = referral_sets(raw[first], stems, policy['thresholds'][first], policy['confidence_margin'],
                          policy['seed'], policy['random_audit_fraction_of_other_images'])
    save(OUT/'routes.json', {k:sorted(v) for k,v in gates.items()})
    base_t = np.array(timing['models'][first]['individual_seconds']).reshape(timing['repeats'], -1)
    assert base_t.shape == (3, len(stems))
    rows, pair_rows = [], []
    for second in RUNS[1:]:
        pair_rows.append(dict(second=NAMES[second], first_fn=int((~found[first]).sum()),
            recoverable_first_fn=int((~found[first] & found[second]).sum()),
            first_tp_second_fn=int((found[first] & ~found[second]).sum()),
            missed_by_both=int((~found[first] & ~found[second]).sum())))
        second_t = np.array(timing['models'][second]['individual_seconds']).reshape(3, -1)
        for name, selected in gates.items():
            mask = gt.stem.isin(selected).to_numpy()
            path = OUT/f'{name}_{frozen["runs"][second]["model"]}_replacement_val.csv'
            output = pd.concat([kept[first][~kept[first].stem.isin(selected)],
                                kept[second][kept[second].stem.isin(selected)]], ignore_index=True)
            output[PRED_COLUMNS].to_csv(path, index=False)
            metrics, _, combined = evaluate(path, 'val', thr=0, matching='iou50')
            assert np.array_equal(combined.detected.to_numpy(), np.where(mask, found[second], found[first]))
            recovering = int((mask & ~found[first] & found[second]).sum())
            losing = int((mask & found[first] & ~found[second]).sum())
            assert metrics['fn'] == scores[first]['fn'] - recovering + losing
            time_gate = np.array([s in selected for s in timing['image_stems']])
            replay_time = (base_t + second_t*time_gate).ravel()
            rows.append(dict(rule=name, second=NAMES[second], referred_images=len(selected),
                referred_fraction=len(selected)/len(stems),
                first_fn_sent_to_second=int((mask & ~found[first]).sum()),
                first_fn_recoverable=recovering, first_tp_lost_if_replaced=losing,
                oracle_remaining_fn=int((~found[first] & ~(mask & found[second])).sum()),
                replacement_tp=metrics['tp'], replacement_fp=metrics['fp'], replacement_fn=metrics['fn'],
                replacement_f1=metrics['f1'],
                replacement_photos_with_no_box=len(stems-set(output.stem)),
                estimated_mean_ms=float(replay_time.mean()*1000),
                estimated_p95_ms=float(np.quantile(replay_time,.95)*1000)))

    train, tm = load_gt('train')
    train = features(train, tm, 'train')
    valfeat = features(gt, manifest, 'val')
    train.to_csv(OUT/'training_box_features.csv', index=False)
    condition_rows = []
    for feature in FEATURES:
        q25, q75 = float(train[feature].quantile(.25)), float(train[feature].quantile(.75))
        for side, label in [('low','le_train_q25'), ('high','gt_train_q75')]:
            a = train[train[feature]<=q25] if side=='low' else train[train[feature]>q75]
            mask = (valfeat[feature]<=q25 if side=='low' else valfeat[feature]>q75).to_numpy()
            sub = valfeat[mask]
            for run in RUNS:
                condition_rows.append(dict(feature=feature, group=label, q25=q25, q75=q75,
                    training_boxes=len(a), training_images=a.stem.nunique(), training_bursts=a.burst_id.nunique(),
                    validation_boxes=len(sub), validation_images=sub.stem.nunique(), validation_bursts=sub.burst_id.nunique(),
                    model=NAMES[run], validation_fn=int((mask & ~found[run]).sum())))
    pd.DataFrame(condition_rows).to_csv(OUT/'conditions.csv', index=False)
    all_detail = gt[['stem','cx','cy','w','h']].copy()
    for run in RUNS:all_detail[NAMES[run]+'_detected'] = found[run]
    all_detail.to_csv(OUT/'validation_complementarity.csv', index=False)
    pd.DataFrame(rows).to_csv(OUT/'replay.csv', index=False)
    pd.DataFrame(pair_rows).to_csv(OUT/'pairs.csv', index=False)
    summary = dict(policy_sha256=sha(pp), scope='exploratory validation replay of frozen MPS predictions',
        baseline={NAMES[r]:scores[r] for r in RUNS}, pairs=pair_rows, replay=rows,
        training=dict(images=len(tm), boxes=len(train), bursts=tm.burst_id.nunique()),
        validation=dict(images=len(manifest), boxes=len(gt), bursts=manifest.burst_id.nunique(), real_normal_images=len(stems-set(gt.stem))),
        first_fn_details=all_detail[~found[first]].to_dict(orient='records'),
        limitations=['IoU50 FN is localization failure, not verified physical invisibility.',
            'Oracle recovery uses official truth and is not a deployed fusion result.',
            'Replacement is a diagnostic control; disagreement should be retained for review, not silently cleared.',
            'Latency is a sum of separately measured per-image timings, not a live cascade benchmark; excludes loading, routing, queues and factory I/O.',
            'Validation has already selected checkpoints and thresholds; results are exploratory, not an independent final estimate.',
            'Condition descriptors use GT boxes for analysis and cannot route a missed target in production.',
            'No real normal products: normal false alarm, production referral fraction and PASS safety unavailable.',
            'One model per family and few misses: no statistically established specialist or family superiority.'])
    save(OUT/'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
