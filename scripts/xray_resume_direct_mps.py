"""Explicitly approved Colab checkpoint continuation on this Mac's MPS GPU.

Keep the source CUDA binding. Allow only the device and wheel platform suffixes
to differ; all data, model, source hashes and base package versions must match.
"""
import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
os.environ['MPLBACKEND'] = 'Agg'
import torch
import xray_train_direct as training
from xray_config import ROOT
from xray_recovery import (RecoveryStore, atomic_json, checkpoint_epoch,
                           dataset_digest, run_lock, sha256)

NAME = 'v2_direct20_rfdetr_cuda_20261006'
RUN = ROOT / 'runs' / NAME
REPORT = ROOT / 'reports/direct_training_20261006'


def verify_source():
    original = json.loads((RUN / 'recovery/binding.json').read_text())
    config = original['config']
    assert config['model'] == 'rfdetr' and config['device'] == 'cuda'
    assert config['epochs'] == 20 and config['batch'] == 4 and config['resolution'] == 512
    for name, digest in config['code'].items():
        assert sha256(ROOT / 'scripts' / name) == digest, f'Source differs: {name}'
    actual_versions = {n: importlib.metadata.version(n) for n in config['versions']}
    for name, version in actual_versions.items():
        expected = config['versions'][name]
        if name in {'torch', 'torchvision'}:
            assert version.split('+')[0] == expected.split('+')[0], (name, version, expected)
        else:
            assert version == expected, (name, version, expected)
    assert dataset_digest(ROOT / 'data/xray_direct_training_20261006') == config['dataset_content_sha256']
    assert sha256(ROOT / 'weights/rf-detr-small.pth') == config['pretrained_sha256']
    saved = RecoveryStore(RUN / 'recovery', config).latest()
    assert saved and checkpoint_epoch(saved['checkpoint'], 'rfdetr') == saved['epoch']
    assert torch.backends.mps.is_available(), 'MPS GPU not available'
    return original, actual_versions, saved


class MPSContinuationStore(RecoveryStore):
    def __init__(self, directory, config):
        original = json.loads((Path(directory) / 'binding.json').read_text())['config']
        assert config['device'] == 'mps'
        for name, expected in original['versions'].items():
            actual = config['versions'][name]
            if name in {'torch', 'torchvision'}:
                assert actual.split('+')[0] == expected.split('+')[0]
            else:
                assert actual == expected
        checked = dict(config, device=original['device'], versions=original['versions'])
        super().__init__(directory, checked)
        saved = self.latest()
        assert saved, 'Full checkpoint required'
        atomic_json(Path(directory).parent / 'mps_continuation.json', dict(
            approved_by='user 2026-10-07 Mac GPU continuation',
            original_device=original['device'], execution_device='mps',
            original_versions=original['versions'], actual_versions=config['versions'],
            resume_epoch=saved['epoch'], original_fingerprint=self.fingerprint,
            launcher_sha256=sha256(Path(__file__)), numeric_equivalence=False,
            rng_policy='Restore Python/NumPy/CPU; restore MPS state when present; omit CUDA RNG',
            started_unix=time.time()))
        print('MPS_FULL_CHECKPOINT_VERIFIED', saved['epoch'], flush=True)


def preflight():
    original, versions, saved = verify_source()
    # A validation image is used only to prove GPU loading. No metric is tuned.
    from PIL import Image
    from xray_predict_controlled import Predictor
    image = next((ROOT / 'data/xray_direct_training_20261006/images/val').glob('*.png'))
    predictor = Predictor('rfdetr', saved['checkpoint'], 'mps', 512)
    with Image.open(image) as im:
        boxes = predictor(im)
    torch.mps.synchronize()
    proof = dict(status='passed', epoch=saved['epoch'], device='mps',
                 original_fingerprint=original['fingerprint'], versions=versions,
                 checkpoint_sha256=sha256(saved['checkpoint']), boxes=len(boxes),
                 mps_allocated_bytes=torch.mps.current_allocated_memory(),
                 checked_unix=time.time(), scope='full-state integrity and MPS inference; training not started')
    assert proof['mps_allocated_bytes'] > 0
    atomic_json(REPORT / 'mps_preflight.json', proof)
    print(json.dumps(proof), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--approved-mps-continuation', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        preflight()
        return
    if not args.approved_mps_continuation:
        parser.error('Explicit approved migration argument required')
    original, versions, saved = verify_source()
    stop = json.loads((REPORT / 'colab_stop_receipt.json').read_text())
    assert stop['status'] == 'stopped' and stop['run'] == NAME and not stop['active_processes']
    proof = json.loads((REPORT / 'mps_preflight.json').read_text())
    assert proof['status'] == 'passed' and proof['original_fingerprint'] == original['fingerprint']

    training.RecoveryStore = MPSContinuationStore
    original_restore = training.restore_rng

    def restore_mps(state, generator=None):
        original_restore({k: v for k, v in state.items() if k != 'cuda'}, generator)

    training.restore_rng = restore_mps
    import rfdetr.training as rf_training
    from pytorch_lightning import Callback
    builder = rf_training.build_trainer

    class MigrationProgress(Callback):
        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            if batch_idx == 0 or (batch_idx + 1) % 100 == 0:
                device = str(next(module.parameters()).device)
                assert device.startswith('mps'), 'Training unexpectedly left GPU'
                progress = dict(device=device, epoch=int(trainer.current_epoch) + 1,
                                batch=int(batch_idx) + 1, global_step=int(trainer.global_step),
                                restored_from_epoch=saved['epoch'],
                                mps_allocated_bytes=torch.mps.current_allocated_memory(),
                                updated_unix=time.time())
                atomic_json(RUN / 'mps_batch_progress.json', progress)
                print('MPS_BATCH_PROGRESS', json.dumps(progress), flush=True)

    def traced_builder(*a, **kw):
        trainer = builder(*a, **kw)
        trainer.callbacks.append(MigrationProgress())
        return trainer

    rf_training.build_trainer = traced_builder
    a = argparse.Namespace(model='rfdetr', data='data/xray_direct_training_20261006',
                           name=NAME, epochs=20, batch=4, resolution=512, device='mps',
                           yolo_weights='weights/yolo11s.pt', resume=True,
                           epoch_validation=True, stop_after_epoch=0)
    with run_lock(RUN):
        try:
            atomic_json(REPORT / 'mps_queue.json', dict(status='running', phase='training',
                        run=NAME, device='mps', pid=os.getpid(), updated_unix=time.time()))
            training.run(a)
            atomic_json(REPORT / 'mps_queue.json', dict(status='running', phase='test_evaluation',
                        run=NAME, device='mps', pid=os.getpid(), updated_unix=time.time()))
            subprocess.run([sys.executable, 'scripts/xray_evaluate_direct.py', '--run', NAME,
                            '--device', 'mps'], cwd=ROOT, check=True)
            atomic_json(REPORT / 'mps_queue.json', dict(status='complete', run=NAME,
                        device='mps', test_evaluated=True, updated_unix=time.time()))
        except BaseException as error:
            atomic_json(REPORT / 'mps_queue.json', dict(status='failed', run=NAME,
                        device='mps', error=repr(error), updated_unix=time.time()))
            raise


if __name__ == '__main__':
    main()
