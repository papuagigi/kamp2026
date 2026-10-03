"""Run the CUDA job after the user has mounted Drive in the notebook."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import torch
from xray_config import ROOT

def main():
    p=argparse.ArgumentParser();p.add_argument('--storage',required=True);a=p.parse_args()
    storage=Path(a.storage)
    if not Path('/content/drive/MyDrive').is_dir():raise RuntimeError('Mount Drive with user approval first')
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU is required; select T4 runtime')
    storage.mkdir(parents=True,exist_ok=True);(storage/'runs').mkdir(exist_ok=True)
    runs=ROOT/'runs'
    if runs.is_symlink():
        assert runs.resolve()==(storage/'runs').resolve()
    elif runs.exists():
        raise RuntimeError('Local runs exists; preserve it and inspect before linking Drive')
    else:runs.symlink_to(storage/'runs',target_is_directory=True)
    # Recover saved predictions after a runtime replacement.
    if (storage/'reports').exists():shutil.copytree(storage/'reports',ROOT/'reports',dirs_exist_ok=True)
    state=dict(status='running',started_unix=time.time(),gpu=torch.cuda.get_device_name(0),torch=torch.__version__,jobs=[])
    def save():
        tmp=storage/'cloud_status.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(storage/'cloud_status.json')
    save();print(json.dumps(state),flush=True)
    name='v2_mix2453_faster_cuda'
    commands=[
        [sys.executable,'-u','scripts/xray_train_resumable.py','--model','faster','--name',name,'--epochs','20','--batch','4','--device','cuda','--resume'],
        [sys.executable,'-u','scripts/xray_evaluate_run.py','--run',name,'--device','cuda']]
    try:
        for phase,cmd in zip(['train','evaluate'],commands):
            job=dict(phase=phase,status='running',started_unix=time.time());state['jobs'].append(job);save()
            # Stream progress into notebook and a persistent Drive log.
            with (storage/f'{name}_{phase}.log').open('a') as log:
                proc=subprocess.Popen(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                try:
                    for line in proc.stdout:print(line,end='',flush=True);log.write(line);log.flush()
                    code=proc.wait()
                except BaseException:
                    proc.terminate();proc.wait();raise
            job.update(status='complete' if code==0 else 'failed',returncode=code,finished_unix=time.time());save()
            shutil.copytree(ROOT/'reports',storage/'reports',dirs_exist_ok=True)
            if code:raise RuntimeError(f'{phase} failed; see persistent log')
        state['status']='complete'
    except BaseException as exc:
        state.update(status='failed',error=repr(exc));raise
    finally:
        state['finished_unix']=time.time();save()
    print('COMPLETE. Checkpoints, logs and scores saved to',storage,flush=True)

if __name__=='__main__':main()
