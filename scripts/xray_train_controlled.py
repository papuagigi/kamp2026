"""Train one model on one isolated dataset; predictions and scoring are separate."""
import argparse
import json
import os
from pathlib import Path
import time
import math

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')
from xray_config import ROOT,select_device,V2_SPLIT_MD5
from xray_model_io import PREPARED,XrayDataset,collate,faster_model,seed_all,sync


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['yolo','faster','rfdetr'],required=True)
    p.add_argument('--variant',default='official');p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--batch',type=int,default=4);p.add_argument('--resolution',type=int,default=512)
    p.add_argument('--name',required=True);p.add_argument('--device',default='auto');p.add_argument('--limit',type=int)
    p.add_argument('--steps',type=int,help='YOLO-only exact optimizer steps, no accumulation, step-based LR; for data ablations')
    a=p.parse_args();seed_all();device=select_device(a.device)
    directory=PREPARED/'variants'/a.variant;out=ROOT/'runs'/a.name
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);start=time.time()
    if a.limit:
        import os
        import yaml
        small=out/'smoke_dataset'
        for split in ['train','val']:
            for image in sorted((directory/'images'/split).glob('*.png'))[:a.limit]:
                for kind,suffix in [('images','.png'),('labels','.txt')]:
                    src=directory/kind/split/(image.stem+suffix);dst=small/kind/split/src.name
                    dst.parent.mkdir(parents=True,exist_ok=True);dst.symlink_to(os.path.relpath(src,dst.parent))
        (small/'data.yaml').write_text(yaml.safe_dump({'path':str(small),'train':'images/train','val':'images/val','names':{0:'defect'}}))
        directory=small
    meta=dict(vars(a),device_used=device,seed=0,dataset=str(directory.relative_to(ROOT)),
              split_md5=V2_SPLIT_MD5,augmentation='precomputed only; automatic random transforms disabled',status='running')
    (out/'execution.json').write_text(json.dumps(meta,indent=2))
    try:
        if a.model=='yolo':
            import yaml
            from ultralytics import YOLO
            config=yaml.safe_load((directory/'data.yaml').read_text());config['path']=str(directory)
            yp=out/'runtime_data.yaml';yp.write_text(yaml.safe_dump(config))
            model=YOLO(str(ROOT/'weights/yolov8n.pt'))
            control={};train_epochs=a.epochs;counter={'steps':0}
            if a.steps:
                batches=math.ceil(len(list((directory/'images/train').glob('*.png')))/a.batch)
                train_epochs=math.ceil(a.steps/batches)
                def batch_start(trainer):
                    lr=.001*(.01+.99*(1-counter['steps']/max(1,a.steps-1)))
                    for pg in trainer.optimizer.param_groups:pg['lr']=lr
                def batch_end(trainer):
                    counter['steps']+=1
                    if counter['steps']>=a.steps:trainer.stop=True
                model.add_callback('on_train_batch_start',batch_start);model.add_callback('on_train_batch_end',batch_end)
                control['nbs']=a.batch
            model.train(data=str(yp),epochs=train_epochs,imgsz=a.resolution,device=device,batch=a.batch,workers=0,
                project=str(ROOT/'runs'),name=a.name,exist_ok=True,verbose=False,plots=False,seed=0,
                deterministic=True,patience=1000,optimizer='AdamW',lr0=.001,lrf=.01,weight_decay=.0005,
                warmup_epochs=0 if a.steps else min(1.,a.epochs/10),amp=False,mosaic=0.,mixup=0.,copy_paste=0.,
                degrees=0.,translate=0.,scale=0.,shear=0.,perspective=0.,flipud=0.,fliplr=0.,
                hsv_h=0.,hsv_s=0.,hsv_v=0.,close_mosaic=0,save=True,**control)
            if a.steps:
                assert counter['steps']==a.steps
                assert model.trainer.ema.updates==a.steps, 'Actual optimizer/EMA updates differ from budget'
                n_images=len(list((directory/'images/train').glob('*.png')))
                exposures=sum(min(a.batch,n_images-(i%batches)*a.batch) for i in range(a.steps))
                meta.update(train_image_exposures=exposures,optimizer_steps=counter['steps'],learning_rate='linear 0.001 to 0.00001 per update',gradient_accumulation=1)
            checkpoint=out/'weights/last.pt'
        elif a.model=='faster':
            import torch
            from torch.utils.data import DataLoader
            ds=XrayDataset(directory,limit=a.limit)
            loader=DataLoader(ds,batch_size=a.batch,shuffle=True,num_workers=0,collate_fn=collate,
                              generator=torch.Generator().manual_seed(0))
            model=faster_model(resolution=a.resolution).to(device)
            opt=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.0001)
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs)
            history=[];steps=0
            for ep in range(a.epochs):
                model.train();losses=[];t=time.time()
                for ims,targets in loader:
                    ims=[im.to(device) for im in ims];targets=[{k:v.to(device) for k,v in x.items()} for x in targets]
                    opt.zero_grad(set_to_none=True);ld=model(ims,targets);loss=sum(ld.values())
                    if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite training loss: {ld}')
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);opt.step()
                    losses.append(float(loss.detach().cpu()));steps+=1
                    if steps<=2 or steps%10==0:print(json.dumps(dict(epoch=ep+1,step=steps,loss=losses[-1],seconds=time.time()-t)),flush=True)
                scheduler.step();sync(device)
                record=dict(epoch=ep+1,loss=sum(losses)/len(losses),seconds=time.time()-t,steps=steps)
                history.append(record);print(json.dumps(record),flush=True)
                (out/'history.json').write_text(json.dumps(history,indent=2))
                checkpoint=out/'last.pt'
                torch.save({'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'resolution':a.resolution,'epoch':ep+1,
                            'optimizer':opt.state_dict(),'seed':0,'steps':steps},checkpoint)
        else:
            from rfdetr import RFDETRSmall
            model=RFDETRSmall(pretrain_weights=str(ROOT/'weights/rf-detr-small.pth'),device=device,
                              resolution=a.resolution,num_classes=1,amp=False,fused_optimizer=False)
            assert model.model_config.num_classes==1
            model.train(dataset_dir=str(directory),dataset_file='yolo',output_dir=str(out),epochs=a.epochs,
                batch_size=a.batch,grad_accum_steps=1,device=device,num_workers=0,seed=0,
                resolution=a.resolution,multi_scale=False,expanded_scales=False,aug_config={},
                lr=.0001,lr_encoder=.000015,weight_decay=.0001,use_ema=False,amp_dtype=None,
                tensorboard=False,wandb=False,run_test=False,early_stopping=False,
                checkpoint_interval=max(1,a.epochs),progress_bar=None,compute_val_loss=False)
            candidates=list(out.glob(f'checkpoint_{a.epochs-1}.ckpt'))+list(out.glob('last.ckpt'))+list(out.glob('last.pth'))
            if not candidates:raise RuntimeError(f'Cannot locate last checkpoint in {out}')
            checkpoint=candidates[0]
            meta['target_classes']=['defect']
        meta.update(status='complete',seconds=time.time()-start,checkpoint=str(checkpoint.relative_to(ROOT)),
                    train_images=len(list((directory/'images/train').glob('*.png'))))
    except BaseException as exc:
        meta.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',seconds=time.time()-start,error=repr(exc));raise
    finally:(out/'execution.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta,indent=2),flush=True)


if __name__=='__main__':main()
