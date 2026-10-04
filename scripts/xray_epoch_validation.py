"""Validate a saved epoch with the project's canonical scorer; never read test images."""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
from xray_config import ROOT, DATA
from xray_recovery import atomic_json, atomic_copy, sha256

POLICY = dict(primary='iou50_f1', ties=['iou50_recall','iou75_ap','earlier_epoch'],
              tolerance=1e-9, threshold='validation_best_f1_frozen_for_test',
              epoch_budget=20, test_used_for_selection=False)


def better(candidate, incumbent):
    if incumbent is None:return True
    for mode, key in [('iou50','f1'),('iou50','recall'),('iou75','ap')]:
        a=candidate['metrics'][mode][key];b=incumbent['metrics'][mode][key]
        a=-1 if a is None else a;b=-1 if b is None else b
        if a > b+POLICY['tolerance']:return True
        if b > a+POLICY['tolerance']:return False
    return candidate['epoch'] < incumbent['epoch']


def update_selection(run, through_epoch):
    run=Path(run);records=[];best=None
    for ep in range(1,through_epoch+1):
        p=run/'validation'/f'epoch_{ep:03d}.json'
        if not p.is_file():raise RuntimeError(f'Missing common validation: {p}')
        r=json.loads(p.read_text());records.append(r)
        if better(r,best):best=r
    if best is None:raise RuntimeError('No validated checkpoints')
    source=ROOT/best['checkpoint']
    if sha256(source)!=best['checkpoint_sha256']:raise RuntimeError('Selected weights changed')
    dest=run/'selected'/('best'+source.suffix)
    if not dest.exists() or sha256(dest)!=best['checkpoint_sha256']:atomic_copy(source,dest)
    summary=dict(policy=POLICY,completed_epochs=through_epoch,best_epoch=best['epoch'],
                 checkpoint=str(dest.relative_to(ROOT)),checkpoint_sha256=best['checkpoint_sha256'],
                 metrics=best['metrics'],selected_from='validation only',
                 history=[dict(epoch=r['epoch'],metrics=r['metrics'],seconds=r['seconds']) for r in records])
    atomic_json(run/'selection.json',summary)
    return summary


def validate_epoch(model, checkpoint, directory, run, epoch, resolution, device):
    subprocess.run([sys.executable,str(ROOT/'scripts/xray_epoch_validation.py'),
        '--model',model,'--checkpoint',str(checkpoint),'--data',str(directory),'--run',str(run),
        '--epoch',str(epoch),'--resolution',str(resolution),'--device',str(device)],check=True,cwd=ROOT)


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True,choices=['yolo','faster','rfdetr','dfine'])
    for n in ['checkpoint','data','run','device']:p.add_argument('--'+n,required=True)
    p.add_argument('--epoch',type=int,required=True);p.add_argument('--resolution',type=int,default=512)
    a=p.parse_args();run=Path(a.run);checkpoint=Path(a.checkpoint);directory=Path(a.data)
    paths=sorted((directory/'images/val').glob('*.png'))
    if not paths:raise ValueError('No validation images')
    # Verify the immutable official validation inputs, including smoke-test subsets.
    for path in paths:
        for folder,suffix in [('images','.png'),('labels','.txt')]:
            source=directory/folder/'val'/(path.stem+suffix)
            official=DATA/folder/'val'/(path.stem+suffix)
            if not official.is_file() or sha256(source)!=sha256(official):
                raise ValueError(f'Validation input differs from fixed split: {source}')
    digest=sha256(checkpoint);dest=run/'validation';dest.mkdir(exist_ok=True,parents=True)
    result_path=dest/f'epoch_{a.epoch:03d}.json';pred=dest/f'epoch_{a.epoch:03d}.csv'
    if result_path.exists():
        old=json.loads(result_path.read_text())
        if old['checkpoint_sha256']==digest and pred.exists() and sha256(pred)==old['predictions_sha256']:
            update_selection(run,a.epoch);print('Common validation already verified',a.epoch,flush=True);return
    from PIL import Image
    from xray_model_io import seed_all
    from xray_predict_controlled import Predictor
    from xray_eval import evaluate
    seed_all();start=time.time()
    if a.model=='dfine':
        from xray_dfine import DFinePredictor
        predictor=DFinePredictor(checkpoint,a.device,a.resolution)
    else:predictor=Predictor(a.model,checkpoint,a.device,a.resolution)
    tmp=pred.with_suffix('.csv.pending')
    with tmp.open('w') as f:
        writer=csv.writer(f);writer.writerow(['stem','cx','cy','w','h','score'])
        for path in paths:
            with Image.open(path) as im:boxes=predictor(im)
            for x1,y1,x2,y2,score in boxes:
                writer.writerow([path.stem,(x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1,score])
    os.replace(tmp,pred)
    stems={p.stem for p in paths};metrics={}
    for mode in ['iou50','iou75','legacy']:
        metrics[mode],_,_=evaluate(pred,'val',stems=stems,matching=mode)
        if metrics[mode]['n_images']!=len(paths):raise RuntimeError('Validation image count mismatch')
    r=dict(epoch=a.epoch,model=a.model,checkpoint=str(checkpoint.relative_to(ROOT)),
           checkpoint_sha256=digest,predictions_sha256=sha256(pred),metrics=metrics,
           seconds=time.time()-start,device=a.device,policy=POLICY,images=len(paths))
    atomic_json(result_path,r);s=update_selection(run,a.epoch)
    print(json.dumps(dict(common_validation_epoch=a.epoch,f1=metrics['iou50']['f1'],
                          recall=metrics['iou50']['recall'],best_epoch=s['best_epoch'],seconds=r['seconds'])),flush=True)

if __name__=='__main__':main()
