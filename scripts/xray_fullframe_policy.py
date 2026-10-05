"""Freeze the approved full-frame policy, then audit saved outputs without tuning.

The existing scorer defines localization TP/FP/FN. Agreement compares predictions
only; it is never treated as ground truth. This does not release physical products.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import pandas as pd
from xray_config import ROOT, DATA
from xray_eval import evaluate
from xray_cascade_runtime import agreement
from xray_inspection_gate import ProductInspection

DEST = ROOT / 'docs/evidence/fullframe_policy_20261005'
LIVE = 'reports/design_review_20261004/yolo8_rfdetr_all_20261005'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def freeze():
    p = json.loads((ROOT/LIVE/'policy.json').read_text())
    records = {}
    for stage in ('first', 'second'):
        spec = dict(p[stage])
        assert digest(ROOT/spec['checkpoint']) == spec['sha256']
        selection = json.loads((ROOT/'runs'/spec['run']/'selection.json').read_text())
        spec['selected_epoch'] = selection['best_epoch']
        spec['predictions'] = {
            'val': f'{LIVE}/{stage}_repeat0_val.csv',
            'test': f"reports/preds_{spec['run']}_test.csv"}
        spec['prediction_sha256'] = {k:digest(ROOT/v) for k,v in spec['predictions'].items()}
        records[stage] = spec
    policy = dict(version='approved_fullframe_20261005', user_approved_date='2026-10-05',
        **records, input='both models independently inspect every full masked image',
        resolution=512, seed=0, flip=False, crop_or_tile=False,
        split_sha256=digest(DATA/'split.csv'), matching='iou50', agreement_iou=0.5,
        threshold_source='existing validation selection; no retuning',
        on_disagreement='preserve both predictions; hold and request reinspection',
        on_any_detection='preserve warning; hold for confirmation even if predictions agree',
        on_both_empty='normality unverified; retain existing inspection; no AI-only release',
        on_error_or_missing_result='hold; preserve existing warning',
        automatic_pass=False, automatic_reject=False, factory_deadline_ms=None,
        deadline_note='no factory requirement supplied; state tests use simulated time',
        real_normal_images=0, role_experiments='preserved separately; not deployed',
        previous_test_exposure=True, original_labels_unchanged=True)
    target = DEST/'policy.json'
    if target.exists() and json.loads(target.read_text()) != policy:
        raise ValueError('Frozen policy differs. Create a reviewed new version.')
    write_json(target, policy)
    return policy


def audit(policy_path):
    policy = json.loads(policy_path.read_text())
    if digest(DATA/'split.csv') != policy['split_sha256']:
        raise ValueError('Split mismatch')
    for stage in ('first','second'):
        spec = policy[stage]
        if digest(ROOT/spec['checkpoint']) != spec['sha256']:
            raise ValueError('Checkpoint mismatch')
    manifest = pd.read_csv(DATA/'manifest.csv')
    summaries = {}; photo_rows = []
    for split in ('val', 'test'):
        predictions = {}; gt = {}; metrics = {}
        for stage in ('first','second'):
            spec = policy[stage]; path = ROOT/spec['predictions'][split]
            if digest(path) != spec['prediction_sha256'][split]:
                raise ValueError('Prediction mismatch')
            m,_,g = evaluate(path,split,thr=spec['threshold'],matching='iou50')
            metrics[stage] = {k:m[k] for k in ('tp','fp','fn','precision','recall','f1','threshold') if k in m}
            df = pd.read_csv(path)
            predictions[stage] = df[df.score >= spec['threshold']]
            gt[stage] = g
        keys = ['stem','cx','cy','w','h']
        assert gt['first'][keys].equals(gt['second'][keys])
        da=gt['first'].detected.to_numpy(bool);db=gt['second'].detected.to_numpy(bool)
        stems = manifest.loc[manifest.split.eq(split),'stem'].tolist()
        states=Counter();patterns=Counter();both_empty=[];first_empty_second_alarm=[]
        for stem in stems:
            a=predictions['first'][predictions['first'].stem.eq(stem)]
            b=predictions['second'][predictions['second'].stem.eq(stem)]
            comparison=agreement(a,b)
            # No operational deadline invented. Missing-result behavior is unit-tested separately.
            frame=digest(DATA/'images'/split/f'{stem}.png')
            gate=ProductInspection(stem,frame,float('inf'))
            gate.accept('first',stem,frame,a.to_dict('records'),0.)
            gate.accept('second',stem,frame,b.to_dict('records'),0.)
            state=gate.state(0.,agree=comparison['agree'])
            assert not state.startswith('PASS')
            assert gate.warning_preserved == bool(len(a) or len(b))
            states[state]+=1
            pattern=f'first_{bool(len(a))}_second_{bool(len(b))}'
            patterns[pattern]+=1
            if not len(a) and not len(b):both_empty.append(stem)
            if not len(a) and len(b):first_empty_second_alarm.append(stem)
            photo_rows.append(dict(split=split,stem=stem,first_boxes=len(a),second_boxes=len(b),
                predictions_agree=comparison['agree'],state=state,warning_preserved=gate.warning_preserved))
        summaries[split]=dict(images=len(stems),gt_boxes=len(da),real_normal_images=0,
            metrics=metrics,alarm_patterns=dict(patterns),states=dict(states),
            both_empty_images=both_empty,first_empty_second_alarm_images=first_empty_second_alarm,
            oracle_localization_diagnostic=dict(first_fn_covered=int((~da&db).sum()),
                second_fn_covered=int((da&~db).sum()),shared_iou_fn=int((~da&~db).sum()),
                note='uses official labels after inference; not a deployable merged detector or physical escape rate'),
            agreement_is_truth=False,normal_false_alarm_rate=None,production_recall=None)
    result=dict(status='passed',policy_sha256=digest(policy_path),splits=summaries,
        new_training=False,new_inference=False,retuned_thresholds=False,
        scope='saved prediction rescore and deterministic state replay; not factory safety validation',
        previous_test_exposure=True,fresh_environment=False,
        limitations=['No real normal products', 'Validation repeatedly reused',
                     'No new held-out production sample', 'Shared IoU errors may still contain the physical object',
                     'No measured reinspection capacity or equipment deadline'])
    write_json(DEST/'audit.json',result)
    pd.DataFrame(photo_rows).to_csv(DEST/'image_states.csv',index=False,lineterminator='\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true')
    args=parser.parse_args()
    if args.freeze: freeze()
    audit(DEST/'policy.json')
