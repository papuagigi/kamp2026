"""Full-epoch detector training with restart checkpoints; no test-set access."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')
import numpy as np
import torch
from xray_config import ROOT, select_device, V2_SPLIT_MD5
from xray_model_io import XrayDataset, collate, faster_model, seed_all, sync

def save_json(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2))
    temp.replace(path)

def save_torch(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    torch.save(value,temp)
    temp.replace(path)

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
    a=p.parse_args();seed_all();device=select_device(a.device)
    directory=ROOT/a.data;out=ROOT/'runs'/a.name
    assert (directory/'data.yaml').is_file()
    assert hashlib.md5((ROOT/'data/xray_v2/split.csv').read_bytes()).hexdigest()==V2_SPLIT_MD5
    dataset_sha=hashlib.sha256((directory/'provenance.json').read_bytes()).hexdigest()
    prior={}
    if (out/'execution.json').exists():
        prior=json.loads((out/'execution.json').read_text())
        for key,val in dict(model=a.model,epochs=a.epochs,batch=a.batch,resolution=a.resolution,dataset_sha256=dataset_sha).items():
            if prior.get(key)!=val:raise ValueError(f'Resume configuration differs: {key}')
        if prior['status']=='complete':print('Already complete:',a.name);return
        if not a.resume:raise FileExistsError('Use --resume for the existing run')
    out.mkdir(parents=True,exist_ok=True)
    start=time.time()
    meta=dict(vars(a),device_used=device,seed=0,dataset=a.data,dataset_sha256=dataset_sha,
        split_md5=V2_SPLIT_MD5,status='running',started_unix=prior.get('started_unix',start),
        train_images=len(list((directory/'images/train').glob('*.png'))),
        augmentation='precomputed only; automatic random transforms disabled',
        checkpoint_selection='fixed final epoch; no test-based selection',attempt_started_unix=start)
    save_json(out/'execution.json',meta)
    try:
        if a.model=='yolo':
            import yaml
            from ultralytics import YOLO
            config=yaml.safe_load((directory/'data.yaml').read_text());config['path']=str(directory)
            yp=out/'runtime_data.yaml';yp.write_text(yaml.safe_dump(config))
            resume_path=out/'weights/resume.pt'
            resume=a.resume and resume_path.is_file()
            model=YOLO(str(resume_path if resume else ROOT/'weights/yolov8n.pt'))
            def preserve_resume(trainer):
                # Ultralytics strips the final last.pt after training; keep full state.
                tmp=resume_path.with_suffix('.tmp')
                shutil.copyfile(trainer.last,tmp);tmp.replace(resume_path)
                save_json(out/'progress.json',dict(epoch=int(trainer.epoch)+1,epochs=a.epochs,updated_unix=time.time()))
            model.add_callback('on_model_save',preserve_resume)
            if resume:
                model.train(resume=True,device=device)
            else:
                model.train(data=str(yp),epochs=a.epochs,imgsz=a.resolution,device=device,batch=a.batch,workers=0,
                    project=str(ROOT/'runs'),name=a.name,exist_ok=True,verbose=False,plots=False,seed=0,
                    deterministic=True,patience=1000,optimizer='AdamW',lr0=.001,lrf=.01,weight_decay=.0005,
                    warmup_epochs=min(1.,a.epochs/10),amp=False,mosaic=0.,mixup=0.,copy_paste=0.,
                    degrees=0.,translate=0.,scale=0.,shear=0.,perspective=0.,flipud=0.,fliplr=0.,
                    hsv_h=0.,hsv_s=0.,hsv_v=0.,close_mosaic=0,save=True,save_period=5)
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
            if a.resume and checkpoint.exists():
                ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
                model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler'])
                first=ck['epoch'];steps=ck['steps'];history=ck['history']
                torch.set_rng_state(ck['torch_rng']);gen.set_state(ck['loader_rng']);random.setstate(ck['python_rng'])
                ns=ck['numpy_rng'];np.random.set_state((ns[0],np.array(ns[1],dtype=np.uint32),ns[2],ns[3],ns[4]))
                if str(device).startswith('cuda') and ck.get('cuda_rng'):torch.cuda.set_rng_state_all(ck['cuda_rng'])
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
                save_json(out/'history.json',history);save_json(out/'progress.json',record)
                print(json.dumps(record),flush=True)
            meta.update(optimizer_steps=steps,train_image_exposures=len(ds)*a.epochs,
                resume_semantics='epoch-boundary model optimizer scheduler and RNG states')
        else:
            from rfdetr import RFDETRSmall
            model=RFDETRSmall(pretrain_weights=str(ROOT/'weights/rf-detr-small.pth'),device=device,
                resolution=a.resolution,num_classes=1,amp=False,fused_optimizer=False)
            checkpoint=out/'last.ckpt'
            resume=str(checkpoint) if a.resume and checkpoint.exists() else ''
            model.train(dataset_dir=str(directory),dataset_file='yolo',output_dir=str(out),epochs=a.epochs,
                batch_size=a.batch,grad_accum_steps=1,device=device,num_workers=0,seed=0,
                resolution=a.resolution,multi_scale=False,expanded_scales=False,aug_config={},
                lr=.0001,lr_encoder=.000015,weight_decay=.0001,use_ema=False,amp_dtype=None,
                tensorboard=False,wandb=False,run_test=False,early_stopping=False,
                checkpoint_interval=5,progress_bar=None,compute_val_loss=False,resume=resume)
            if not checkpoint.exists():raise RuntimeError('RF-DETR last.ckpt missing')
            meta['target_classes']=['defect'];meta['resume_semantics']='Lightning full checkpoint resume; last.ckpt each epoch'
        meta.update(status='complete',checkpoint=str(checkpoint.relative_to(ROOT)),completed_epochs=a.epochs)
    except BaseException as exc:
        meta.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=repr(exc));raise
    finally:
        meta['seconds']=prior.get('seconds',0)+time.time()-start
        save_json(out/'execution.json',meta)
    print(json.dumps(meta,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
