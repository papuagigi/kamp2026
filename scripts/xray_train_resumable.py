"""Full-epoch detector training with restart checkpoints; no test-set access."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time
import importlib.metadata
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')
import numpy as np
import torch
from xray_config import ROOT, select_device, V2_SPLIT_MD5
from xray_epoch_validation import validate_epoch, POLICY
from xray_model_io import XrayDataset, collate, faster_model, seed_all, sync
from xray_recovery import (RecoveryStore, atomic_json, atomic_torch, atomic_copy,
    capture_rng, restore_rng, run_lock, dataset_digest, sha256)

def save_json(path, value):
    atomic_json(path, value)

def save_torch(path, value):
    atomic_torch(path, value)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',choices=['yolo','faster','rfdetr'],required=True)
    p.add_argument('--data',default='data/xray_combined_20261003')
    p.add_argument('--name',required=True)
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--batch',type=int,default=4)
    p.add_argument('--resolution',type=int,default=512)
    p.add_argument('--device',default='auto')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--epoch-validation',action='store_true')
    p.add_argument('--stop-after-epoch',type=int,default=0,
                   help='Preparation check only: pause after this completed epoch')
    a=p.parse_args()
    if Path(a.name).name != a.name or a.name in {'.','..'}:p.error('name must be a single folder name')
    if a.epochs<1 or a.batch<1 or a.stop_after_epoch<0:p.error('Invalid epoch/batch count')
    with run_lock(ROOT/'runs'/a.name):
        run(a)

def run(a):
    seed_all();device=select_device(a.device)
    directory=ROOT/a.data;out=ROOT/'runs'/a.name
    assert (directory/'data.yaml').is_file()
    assert hashlib.md5((ROOT/'data/xray_v2/split.csv').read_bytes()).hexdigest()==V2_SPLIT_MD5
    dataset_sha=hashlib.sha256((directory/'provenance.json').read_bytes()).hexdigest()
    prior={}
    if (out/'execution.json').exists():
        prior=json.loads((out/'execution.json').read_text())
        for key,val in dict(model=a.model,epochs=a.epochs,batch=a.batch,resolution=a.resolution,dataset_sha256=dataset_sha).items():
            if prior.get(key)!=val:raise ValueError(f'Resume configuration differs: {key}')
        if prior['status']=='complete' and not (out/'recovery/binding.json').exists():
            print('Already complete (legacy run, unchanged):',a.name);return
        if not a.resume:raise FileExistsError('Use --resume for the existing run')
    out.mkdir(parents=True,exist_ok=True)
    package_names=['torch','torchvision']+{'yolo':['ultralytics'],
        'faster':[], 'rfdetr':['rfdetr','pytorch-lightning']}[a.model]
    recovery_config=dict(model=a.model,epochs=a.epochs,batch=a.batch,resolution=a.resolution,
        dataset_content_sha256=dataset_digest(directory),split_md5=V2_SPLIT_MD5,seed=0,device=device,epoch_validation=a.epoch_validation,selection_policy=POLICY,
        versions={n:importlib.metadata.version(n) for n in package_names},
        code={n:sha256(ROOT/'scripts'/n) for n in ['xray_train_resumable.py','xray_model_io.py','xray_recovery.py','xray_epoch_validation.py','xray_eval.py','xray_predict_controlled.py']})
    # A legacy interrupted run must be inspected/migrated explicitly, never restarted silently.
    if prior and not (out/'recovery/binding.json').exists():
        raise RuntimeError('Legacy incomplete run: inspect its checkpoint before migrating to the new recovery format')
    store=RecoveryStore(out/'recovery',recovery_config)
    saved=store.latest() if a.resume else None
    if not a.resume and store.latest() is not None:
        raise FileExistsError('Verified checkpoint exists; use --resume')
    completed=saved['epoch'] if saved else 0
    start=time.time()
    meta=dict(vars(a),device_used=device,seed=0,dataset=a.data,dataset_sha256=dataset_sha,
        split_md5=V2_SPLIT_MD5,status='running',started_unix=prior.get('started_unix',start),
        train_images=len(list((directory/'images/train').glob('*.png'))),
        augmentation='precomputed only; automatic random transforms disabled',
        checkpoint_selection=POLICY if a.epoch_validation else 'fixed final epoch; no test-based selection',attempt_started_unix=start)
    meta.update(recovery_schema=1,recovery_fingerprint=store.fingerprint,
                recovery_epoch=completed,loss_window='unfinished epoch since last verified commit')
    if saved and completed>=a.epochs:
        # A process may die after its final commit and before marking execution complete.
        meta.update(status='complete',completed_epochs=completed,checkpoint=str(saved['checkpoint'].relative_to(ROOT)),
                    seconds=prior.get('seconds',0),recovered_final_commit=True)
        save_json(out/'execution.json',meta);print(json.dumps(meta),flush=True);return
    save_json(out/'execution.json',meta)
    try:
        if a.model=='yolo':
            import yaml
            from ultralytics import YOLO
            config=yaml.safe_load((directory/'data.yaml').read_text());config['path']=str(directory)
            yp=out/'runtime_data.yaml';yp.write_text(yaml.safe_dump(config))
            resume_path=out/'weights/resume.pt'
            if saved:atomic_copy(saved['checkpoint'],resume_path)
            resume=saved is not None
            model=YOLO(str(resume_path if resume else ROOT/'weights/yolov8n.pt'))
            def restore_yolo_rng(trainer):
                if saved:restore_rng(saved['rng'],getattr(trainer.train_loader,'generator',None))
            def preserve_resume(trainer):
                nonlocal completed
                # Ultralytics strips the final last.pt after training; keep full state.
                completed=int(trainer.epoch)+1
                atomic_copy(trainer.last,resume_path)
                if a.epoch_validation:
                    validate_epoch('yolo',out/'weights'/f'epoch{trainer.epoch}.pt',directory,out,completed,a.resolution,device)
                store.publish(resume_path,completed,capture_rng(device,getattr(trainer.train_loader,'generator',None)))
                save_json(out/'progress.json',dict(epoch=completed,epochs=a.epochs,updated_unix=time.time(),verified=True))
                if a.stop_after_epoch and completed>=a.stop_after_epoch:trainer.stop=True
            model.add_callback('on_train_start',restore_yolo_rng)
            model.add_callback('on_model_save',preserve_resume)
            if resume:
                model.train(resume=True,device=device)
            else:
                model.train(data=str(yp),epochs=a.epochs,imgsz=a.resolution,device=device,batch=a.batch,workers=0,
                    project=str(ROOT/'runs'),name=a.name,exist_ok=True,verbose=False,plots=False,seed=0,
                    deterministic=True,patience=1000,optimizer='AdamW',lr0=.001,lrf=.01,weight_decay=.0005,
                    warmup_epochs=min(1.,a.epochs/10),amp=False,mosaic=0.,mixup=0.,copy_paste=0.,
                    degrees=0.,translate=0.,scale=0.,shear=0.,perspective=0.,flipud=0.,fliplr=0.,
                    hsv_h=0.,hsv_s=0.,hsv_v=0.,close_mosaic=0,save=True,save_period=1)
            checkpoint=out/'weights/last.pt'
            meta['resume_semantics']='Ultralytics native epoch resume; final full state retained in weights/resume.pt'
        elif a.model=='faster':
            from torch.utils.data import DataLoader
            gen=torch.Generator().manual_seed(0)
            ds=XrayDataset(directory)
            loader=DataLoader(ds,batch_size=a.batch,shuffle=True,num_workers=0,collate_fn=collate,generator=gen)
            model=faster_model(resolution=a.resolution).to(device)
            opt=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.0001)
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs)
            history=[];steps=0;first=0;checkpoint=out/'last.pt'
            if saved:
                ck=torch.load(saved['checkpoint'],map_location='cpu',weights_only=True)
                model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler'])
                first=ck['epoch'];steps=ck['steps'];history=ck['history']
                torch.set_rng_state(ck['torch_rng']);gen.set_state(ck['loader_rng']);random.setstate(ck['python_rng'])
                ns=ck['numpy_rng'];np.random.set_state((ns[0],np.array(ns[1],dtype=np.uint32),ns[2],ns[3],ns[4]))
                if str(device).startswith('cuda') and ck.get('cuda_rng'):torch.cuda.set_rng_state_all(ck['cuda_rng'])
                restore_rng(saved['rng'],gen)
                print('Resuming after epoch',first,flush=True)
            for ep in range(first,a.epochs):
                model.train();losses=[];t=time.time()
                for ims,targets in loader:
                    ims=[im.to(device) for im in ims];targets=[{k:v.to(device) for k,v in x.items()} for x in targets]
                    opt.zero_grad(set_to_none=True);ld=model(ims,targets);loss=sum(ld.values())
                    if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss: {ld}')
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);opt.step()
                    losses.append(float(loss.detach().cpu()));steps+=1
                    if steps<=2 or steps%100==0:print(json.dumps(dict(epoch=ep+1,step=steps,loss=losses[-1],seconds=time.time()-t)),flush=True)
                scheduler.step();sync(device)
                record=dict(epoch=ep+1,loss=sum(losses)/len(losses),seconds=time.time()-t,steps=steps)
                history.append(record);ns=np.random.get_state()
                save_torch(checkpoint,dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                    resolution=a.resolution,epoch=ep+1,optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),
                    seed=0,steps=steps,history=history,torch_rng=torch.get_rng_state(),loader_rng=gen.get_state(),
                    python_rng=random.getstate(),numpy_rng=(ns[0],ns[1].tolist(),ns[2],ns[3],ns[4]),
                    cuda_rng=torch.cuda.get_rng_state_all() if str(device).startswith('cuda') else None))
                completed=ep+1
                save_torch(out/'epochs'/f'epoch_{completed:03d}.pt',
                    dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},epoch=completed,resolution=a.resolution))
                if a.epoch_validation:
                    validate_epoch('faster',out/'epochs'/f'epoch_{completed:03d}.pt',directory,out,completed,a.resolution,device)
                store.publish(checkpoint,completed,capture_rng(device,gen))
                save_json(out/'history.json',history);save_json(out/'progress.json',record)
                print(json.dumps(record),flush=True)
                if a.stop_after_epoch and completed>=a.stop_after_epoch:break
            meta.update(optimizer_steps=steps,train_image_exposures=len(ds)*completed,
                resume_semantics='epoch-boundary model optimizer scheduler and RNG states')
        else:
            from rfdetr import RFDETRSmall
            import rfdetr.training as rf_training
            from pytorch_lightning.callbacks import Checkpoint
            model=RFDETRSmall(pretrain_weights=str(ROOT/'weights/rf-detr-small.pth'),device=device,
                resolution=a.resolution,num_classes=1,amp=False,fused_optimizer=False)
            checkpoint=out/'last.ckpt'
            if saved:atomic_copy(saved['checkpoint'],checkpoint)
            resume=str(checkpoint) if saved else ''
            class RecoveryCallback(Checkpoint):
                def on_train_start(self, trainer, module):
                    if saved:restore_rng(saved['rng'])
                def on_train_epoch_end(self, trainer, module):
                    nonlocal completed
                    completed=int(trainer.current_epoch)+1
                    native=out/f'checkpoint_{trainer.current_epoch}.ckpt'
                    # Appended after the native ModelCheckpoint callbacks have written the archive.
                    if not native.is_file():raise RuntimeError('Native RF-DETR epoch archive missing')
                    atomic_copy(native,checkpoint)
                    if a.epoch_validation:
                        validate_epoch('rfdetr',native,directory,out,completed,a.resolution,device)
                    store.publish(checkpoint,completed,capture_rng(device))
                    save_json(out/'progress.json',dict(epoch=completed,epochs=a.epochs,verified=True))
                    if a.stop_after_epoch and completed>=a.stop_after_epoch:trainer.should_stop=True
            original_builder=rf_training.build_trainer
            def recovery_builder(*args,**kwargs):
                trainer=original_builder(*args,**kwargs)
                trainer.callbacks.append(RecoveryCallback())
                return trainer
            rf_training.build_trainer=recovery_builder
            try:
                model.train(dataset_dir=str(directory),dataset_file='yolo',output_dir=str(out),epochs=a.epochs,
                    batch_size=a.batch,grad_accum_steps=1,device=device,num_workers=0,seed=0,
                    resolution=a.resolution,multi_scale=False,expanded_scales=False,aug_config={},
                    lr=.0001,lr_encoder=.000015,weight_decay=.0001,use_ema=False,amp_dtype=None,
                    tensorboard=False,wandb=False,run_test=False,early_stopping=False,
                    checkpoint_interval=1,progress_bar=None,compute_val_loss=False,resume=resume)
            finally:
                rf_training.build_trainer=original_builder
            if not checkpoint.exists():raise RuntimeError('RF-DETR last.ckpt missing')
            meta['target_classes']=['defect'];meta['resume_semantics']='Lightning full checkpoint resume; last.ckpt each epoch'
        meta.update(status='complete' if completed==a.epochs else 'paused',
                    checkpoint=str(checkpoint.relative_to(ROOT)),completed_epochs=completed)
    except BaseException as exc:
        meta.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=repr(exc));raise
    finally:
        meta['seconds']=prior.get('seconds',0)+time.time()-start
        save_json(out/'execution.json',meta)
    print(json.dumps(meta,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
