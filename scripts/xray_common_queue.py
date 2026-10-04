"""Run the agreed common-epoch comparison; one training job per device, no test use."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from xray_config import ROOT
from xray_recovery import atomic_json, run_lock


def main():
    p=argparse.ArgumentParser();p.add_argument('--host',choices=['mac','colab'],required=True)
    p.add_argument('--storage',type=Path);a=p.parse_args()
    if a.storage:
        target=a.storage/'runs';target.mkdir(parents=True,exist_ok=True)
        link=ROOT/'runs'
        if not link.exists():link.symlink_to(target,target_is_directory=True)
        if link.resolve()!=target.resolve():raise RuntimeError('Runs must point to approved persistent storage')
    jobs=[('yolo','mps',8),('rfdetr','mps',4)] if a.host=='mac' else [('faster','cuda',4),('dfine','cuda',4)]
    status=(a.storage if a.storage else ROOT/'reports/common_epoch_20261004')
    status.mkdir(parents=True,exist_ok=True)
    with run_lock(status/('queue_'+a.host)):
        for model,device,batch in jobs:
            name=f'v2_common20_{model}_{device}_20261004'
            script='xray_dfine.py' if model=='dfine' else 'xray_train_resumable.py'
            cmd=[sys.executable,'scripts/'+script,'--name',name,'--epochs','20','--batch',str(batch),
                 '--resolution','512','--device',device,'--resume']
            if model!='dfine':cmd+=['--model',model,'--epoch-validation']
            atomic_json(status/('queue_'+a.host+'.json'),dict(status='running',current=name,command=cmd))
            with (status/(name+'.log')).open('a') as log:
                proc=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if proc.returncode:
                atomic_json(status/('queue_'+a.host+'.json'),dict(status='failed',current=name,returncode=proc.returncode))
                raise SystemExit(proc.returncode)
        atomic_json(status/('queue_'+a.host+'.json'),dict(status='complete',jobs=[j[0] for j in jobs],test_evaluated=False))

if __name__=='__main__':main()
