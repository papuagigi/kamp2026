"""Resume the approved Colab RF-DETR run, then evaluate its selected checkpoint."""
import json
import subprocess
import sys
import time
from pathlib import Path
from xray_config import ROOT
from xray_recovery import atomic_json, run_lock


def main():
    storage = (ROOT/'runs').resolve().parent
    assert str(storage).startswith('/content/drive/MyDrive/xray_direct_training_20261006')
    name = 'v2_direct20_rfdetr_cuda_20261006'
    state = storage/'queue_rfdetr.json'
    with run_lock(storage/'queue_rfdetr'):
        phases = [
            ('training',[sys.executable,'scripts/xray_train_direct.py','--model','rfdetr',
              '--name',name,'--data','data/xray_direct_training_20261006','--epochs','20',
              '--batch','4','--resolution','512','--device','cuda','--epoch-validation','--resume']),
            ('test_evaluation',[sys.executable,'scripts/xray_evaluate_direct.py','--run',name,'--device','cuda'])]
        for phase, cmd in phases:
            atomic_json(state,dict(status='running',phase=phase,run=name,updated_unix=time.time()))
            with (storage/(name+'_'+phase+'.log')).open('a') as log:
                proc = subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if proc.returncode:
                atomic_json(state,dict(status='failed',phase=phase,run=name,returncode=proc.returncode,updated_unix=time.time()))
                raise SystemExit(proc.returncode)
        atomic_json(state,dict(status='complete',run=name,test_evaluated=True,updated_unix=time.time()))


if __name__ == '__main__':
    main()
