"""Shared immutable-checkpoint checks for the common-epoch post-training audit."""
import hashlib
import json
import os
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
from xray_config import ROOT, DATA, V2_SPLIT_MD5
from xray_recovery import sha256, atomic_json

RUNS = ['v2_common20_yolo_mps_20261004', 'v2_common20_faster_cuda_20261004',
        'v2_common20_rfdetr_mps_20261004', 'v2_common20_dfine_cuda_20261004']
REPORT = ROOT / 'reports/common_epoch_20261004/evaluation'

def verified(run):
    assert run in RUNS
    assert hashlib.md5((DATA/'split.csv').read_bytes()).hexdigest() == V2_SPLIT_MD5
    directory = ROOT/'runs'/run
    meta = json.loads((directory/'execution.json').read_text())
    selection = json.loads((directory/'selection.json').read_text())
    assert meta['status'] == 'complete' and selection['completed_epochs'] == 20
    assert meta['resolution'] == 512
    assert selection['selected_from'] == 'validation only'
    from xray_epoch_validation import better, POLICY
    assert selection['policy'] == POLICY
    assert [r['epoch'] for r in selection['history']] == list(range(1,21))
    best=None
    for row in selection['history']:
        if better(row,best):best=row
    assert best['epoch'] == selection['best_epoch'], 'Selection rule mismatch'
    checkpoint = ROOT/selection['checkpoint']
    assert sha256(checkpoint) == selection['checkpoint_sha256']
    ep = selection['best_epoch']
    validation = json.loads((directory/'validation'/f'epoch_{ep:03d}.json').read_text())
    prediction = directory/'validation'/f'epoch_{ep:03d}.csv'
    assert sha256(prediction) == validation['predictions_sha256']
    assert validation['checkpoint_sha256'] == selection['checkpoint_sha256']
    assert validation['images'] == 107 and validation['metrics'] == selection['metrics']
    return meta, selection, prediction, validation

def freeze():
    records = {}
    for run in RUNS:
        meta, selection, _, _ = verified(run)
        records[run] = dict(model=meta['model'], epoch=selection['best_epoch'],
            checkpoint=selection['checkpoint'], checkpoint_sha256=selection['checkpoint_sha256'],
            thresholds={m: selection['metrics'][m]['thr'] for m in ['iou50','iou75','legacy']},
            selection_sha256=sha256(ROOT/'runs'/run/'selection.json'))
    record = dict(runs=records, split_md5=V2_SPLIT_MD5, selection_source='validation only',
        test_used_for_selection=False, previous_test_exposure=True,
        real_normal_product_images=0, resolution=512, export_threshold=.001)
    path = REPORT/'frozen_selection.json'
    if path.exists():
        assert json.loads(path.read_text()) == record, 'Frozen selection changed'
    else:
        atomic_json(path, record)
    return record

def predictor_for(meta, selection, device):
    checkpoint = ROOT/selection['checkpoint']
    if meta['model'] == 'dfine':
        from xray_dfine import DFinePredictor
        return DFinePredictor(checkpoint, device, 512)
    from xray_predict_controlled import Predictor
    return Predictor(meta['model'], checkpoint, device, 512)
