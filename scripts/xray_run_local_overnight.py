"""Sequential local MPS training queue; logs survive this chat process."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from xray_config import ROOT

def main():
    report=ROOT/'reports/overnight_20261003';report.mkdir(parents=True,exist_ok=True)
    logs=ROOT/'logs';logs.mkdir(exist_ok=True)
    state=dict(pid=os.getpid(),status='running',started_unix=time.time(),jobs=[])
    status=report/'local_queue.json'
    def save():
        tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(status)
    save()
    for model,batch in [('yolo',8),('rfdetr',4)]:
        name=f'v2_mix2453_{model}_mps'
        commands=[
            [sys.executable,'-u','scripts/xray_train_resumable.py','--model',model,'--name',name,'--epochs','20','--batch',str(batch),'--device','mps','--resume'],
            [sys.executable,'-u','scripts/xray_evaluate_run.py','--run',name,'--device','mps']]
        for phase,cmd in zip(['train','evaluate'],commands):
            job=dict(model=model,phase=phase,command=cmd,status='running',started_unix=time.time())
            state['jobs'].append(job);save()
            with (logs/f'{name}_{phase}.log').open('a') as log:
                child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                job['pid']=child.pid;save();code=child.wait()
            job.update(status='complete' if code==0 else 'failed',returncode=code,finished_unix=time.time());save()
            if code:
                # Preserve the failed job and still attempt the other model.
                break
    state.update(status='complete' if all(j['status']=='complete' for j in state['jobs']) else 'partial_failure',finished_unix=time.time());save()

if __name__=='__main__':main()
