"""User-approved CUDA to CPU continuation; original source/binding stay intact."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
os.environ['MPLBACKEND']='Agg'
import xray_train_direct as training
from xray_recovery import RecoveryStore, atomic_json, run_lock, sha256
from xray_config import ROOT

class CPUContinuationStore(RecoveryStore):
    def __init__(self, directory, config):
        original=json.loads((Path(directory)/'binding.json').read_text())['config']
        assert config['device']=='cpu' and original['device']=='cuda'
        # Only the explicitly approved execution device differs. All original
        # data, package and source hashes still pass RecoveryStore's fingerprint.
        checked=dict(config,device='cuda')
        super().__init__(directory,checked)
        saved=self.latest()
        if saved is None:raise RuntimeError('Verified full checkpoint required')
        atomic_json(Path(directory).parent/'cpu_continuation.json',dict(
            approved_by='user 2026-10-07 CPU continuation request',
            original_device='cuda',execution_device='cpu',resume_epoch=saved['epoch'],
            original_fingerprint=self.fingerprint,launcher_sha256=sha256(Path(__file__)),
            rng_policy='Restore CPU Python NumPy; CUDA RNG retained in previous checkpoint only',
            numeric_equivalence=False,started_unix=time.time()))
        print('CPU_CONTINUATION_VERIFIED',saved['epoch'],self.fingerprint,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--approved-cpu-continuation',action='store_true',required=True);p.parse_args()
    storage=(ROOT/'runs').resolve().parent
    assert str(storage)=='/content/drive/MyDrive/xray_direct_training_20261006'
    name='v2_direct20_rfdetr_cuda_20261006'
    state=storage/'queue_rfdetr.json'
    training.RecoveryStore=CPUContinuationStore
    original_restore=training.restore_rng
    def restore_cpu(state,generator=None):
        original_restore({k:v for k,v in state.items() if k not in ('cuda','mps')},generator)
    training.restore_rng=restore_cpu
    a=argparse.Namespace(model='rfdetr',data='data/xray_direct_training_20261006',name=name,
        epochs=20,batch=4,resolution=512,device='cpu',yolo_weights='weights/yolo11s.pt',
        resume=True,epoch_validation=True,stop_after_epoch=0)
    with run_lock(storage/'queue_rfdetr'):
        try:
            atomic_json(state,dict(status='running',phase='training',run=name,device='cpu',updated_unix=time.time()))
            with run_lock(ROOT/'runs'/name):training.run(a)
            atomic_json(state,dict(status='running',phase='test_evaluation',run=name,device='cpu',updated_unix=time.time()))
            with (storage/(name+'_test_evaluation.log')).open('a') as log:
                subprocess.run([sys.executable,'scripts/xray_evaluate_direct.py','--run',name,'--device','cpu'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            atomic_json(state,dict(status='complete',run=name,device='cpu',test_evaluated=True,updated_unix=time.time()))
        except BaseException as e:
            atomic_json(state,dict(status='failed',run=name,device='cpu',error=repr(e),updated_unix=time.time()));raise

if __name__=='__main__':main()
