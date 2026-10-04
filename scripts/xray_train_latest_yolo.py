"""Controlled YOLO11/26 comparison, separate from immutable legacy runs.

Uses the existing 20-epoch selection and recovery utilities. No test access.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
from xray_config import ROOT, V2_SPLIT_MD5, select_device
from xray_epoch_validation import POLICY
from xray_latest_validation import GUARD
from xray_model_io import seed_all
from xray_recovery import (RecoveryStore, atomic_copy, atomic_json, capture_rng,
                           dataset_digest, restore_rng, run_lock, sha256)

WEIGHTS = {
    'yolo11n': '0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1',
    'yolo26n': '9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef',
}


def train(a):
    import yaml
    from ultralytics import YOLO
    seed_all()
    device = select_device(a.device)
    directory = ROOT / 'data/xray_combined_20261003'
    out = ROOT / 'runs' / a.name
    weight = ROOT / 'weights' / (a.architecture + '.pt')
    assert sha256(weight) == WEIGHTS[a.architecture]
    assert hashlib.md5((ROOT/'data/xray_v2/split.csv').read_bytes()).hexdigest() == V2_SPLIT_MD5
    config = dict(model='yolo', architecture=a.architecture, epochs=20, batch=8,
                  resolution=512, device=device, seed=0, epoch_validation=True,
                  dataset_content_sha256=dataset_digest(directory), split_md5=V2_SPLIT_MD5,
                  pretrained_sha256=WEIGHTS[a.architecture], selection=POLICY, geometry_guard=GUARD,
                  versions={k:importlib.metadata.version(k) for k in ['torch','torchvision','ultralytics']},
                  code={k:sha256(ROOT/'scripts'/k) for k in [
                      'xray_train_latest_yolo.py','xray_latest_validation.py','xray_epoch_validation.py','xray_eval.py',
                      'xray_recovery.py','xray_predict_controlled.py','xray_model_io.py']})
    with run_lock(out):
        store = RecoveryStore(out/'recovery', config)
        saved = store.latest()
        if saved and not a.resume:
            raise FileExistsError('Verified state exists; use --resume')
        completed = saved['epoch'] if saved else 0
        if completed >= 20:
            assert (out/'selection.json').is_file()
            print(json.dumps(dict(status='complete', architecture=a.architecture, epoch=completed)),flush=True)
            return
        old = json.loads((out/'execution.json').read_text()) if (out/'execution.json').exists() else {}
        started = time.time()
        train_images=len(list((directory/'images/train').glob('*.png')))
        validation_images=len(list((directory/'images/val').glob('*.png')))
        assert (train_images,validation_images)==(2453,107)
        meta = dict(config, status='running', name=a.name, train_images=train_images,
                    validation_images=validation_images, no_test=True, attempt_started_unix=started,
                    started_unix=old.get('started_unix',started), resume_epoch=completed,
                    controlled_recipe='same AdamW recipe as YOLOv8n; not default-recipe optimum')
        atomic_json(out/'execution.json',meta)
        runtime_data = yaml.safe_load((directory/'data.yaml').read_text())
        runtime_data['path'] = str(directory)
        runtime_path = out/'runtime_data.yaml'
        runtime_path.write_text(yaml.safe_dump(runtime_data))
        resume_path = out/'weights/resume.pt'
        if saved:
            atomic_copy(saved['checkpoint'],resume_path)
        model = YOLO(str(resume_path if saved else weight))
        meta['end2end'] = bool(getattr(model.model,'end2end',False))
        atomic_json(out/'execution.json',meta)

        def restore(trainer):
            if saved:
                restore_rng(saved['rng'],getattr(trainer.train_loader,'generator',None))

        def preserve(trainer):
            nonlocal completed
            completed = int(trainer.epoch)+1
            atomic_copy(trainer.last,resume_path)
            subprocess.run([sys.executable,str(ROOT/'scripts/xray_latest_validation.py'),
                '--checkpoint',str(out/'weights'/f'epoch{trainer.epoch}.pt'),
                '--data',str(directory),'--run',str(out),'--epoch',str(completed),
                '--device',device],check=True,cwd=ROOT)
            store.publish(resume_path,completed,capture_rng(device,getattr(trainer.train_loader,'generator',None)))
            atomic_json(out/'progress.json',dict(epoch=completed,epochs=20,verified=True,updated_unix=time.time()))
            if a.stop_after_epoch and completed >= a.stop_after_epoch:
                trainer.stop=True

        model.add_callback('on_train_start',restore)
        model.add_callback('on_model_save',preserve)
        try:
            if saved:
                model.train(resume=True,device=device)
            else:
                model.train(data=str(runtime_path),epochs=20,imgsz=512,device=device,batch=8,workers=0,
                    project=str(ROOT/'runs'),name=a.name,exist_ok=True,verbose=False,plots=False,seed=0,
                    deterministic=True,patience=1000,optimizer='AdamW',lr0=.001,lrf=.01,weight_decay=.0005,
                    warmup_epochs=1.,amp=False,mosaic=0.,mixup=0.,copy_paste=0.,
                    degrees=0.,translate=0.,scale=0.,shear=0.,perspective=0.,flipud=0.,fliplr=0.,
                    hsv_h=0.,hsv_s=0.,hsv_v=0.,close_mosaic=0,save=True,save_period=1)
            meta.update(status='complete' if completed==20 else 'paused',completed_epochs=completed)
        except BaseException as exc:
            meta.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=repr(exc))
            raise
        finally:
            meta['seconds']=old.get('seconds',0)+time.time()-started
            atomic_json(out/'execution.json',meta)
        print(json.dumps(meta,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--architecture',choices=WEIGHTS,required=True)
    p.add_argument('--name',required=True)
    p.add_argument('--device',default='mps')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--stop-after-epoch',type=int,default=0)
    a=p.parse_args()
    if Path(a.name).name!=a.name or a.name in ('.','..'):p.error('Single folder name required')
    if a.stop_after_epoch<0 or a.stop_after_epoch>20:p.error('Invalid stopping epoch')
    train(a)
