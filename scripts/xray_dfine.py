"""D-FINE-S adapter using the pinned official implementation and common evaluation."""
import argparse
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')
import numpy as np
import torch
from PIL import Image
from xray_config import ROOT, V2_SPLIT_MD5
from xray_model_io import seed_all
from xray_recovery import RecoveryStore, atomic_json, atomic_torch, capture_rng, restore_rng, run_lock, dataset_digest, sha256
from xray_epoch_validation import validate_epoch, POLICY

VENDOR=ROOT/'reports/common_epoch_20261004/vendor/D-FINE'

def vendor_import():
    if not (VENDOR/'src/core/yaml_config.py').is_file():raise FileNotFoundError('Pinned D-FINE source missing')
    sys.path.insert(0,str(VENDOR))
    from src.core import YAMLConfig, yaml_config, yaml_utils
    # Official load_config has a mutable default; isolate separate config loads.
    yaml_config.load_config=lambda path: yaml_utils.load_config(path,{})
    return YAMLConfig


def annotations(data,split,out):
    images=[];anns=[]
    for i,p in enumerate(sorted((data/'images'/split).glob('*.png'))):
        with Image.open(p) as im:w,h=im.size
        images.append(dict(id=i,file_name=p.name,width=w,height=h))
        for line in (data/'labels'/split/(p.stem+'.txt')).read_text().splitlines():
            if not line.strip():continue
            c,x,y,bw,bh=map(float,line.split());assert c==0
            box=[(x-bw/2)*w,(y-bh/2)*h,bw*w,bh*h]
            anns.append(dict(id=len(anns),image_id=i,category_id=0,bbox=box,area=box[2]*box[3],iscrowd=0))
    atomic_json(out,dict(images=images,annotations=anns,categories=[dict(id=0,name='defect')]))
    return len(images)


def make_config(data,out,resolution,batch,epochs):
    import yaml
    YAMLConfig=vendor_import();base=VENDOR/'configs/dfine/dfine_hgnetv2_s_coco.yml'
    cfg=YAMLConfig(str(base));c=cfg.yaml_cfg
    c.pop('__include__',None)
    c.update(num_classes=1,remap_mscoco_category=False,eval_spatial_size=[resolution]*2,
        epochs=epochs,output_dir=str(out),use_amp=False,use_ema=False,sync_bn=False,use_wandb=False)
    c['HGNetv2']['pretrained']=False
    # Fixed precomputed inputs: no crop, flip, photometric or multiscale augmentation here.
    for split,key in [('train','train_dataloader'),('val','val_dataloader')]:
        ann=out/(split+'_coco.json');annotations(data,split,ann)
        d=c[key];d.update(total_batch_size=batch,num_workers=0,drop_last=False)
        d['dataset'].update(img_folder=str(data/'images'/split),ann_file=str(ann))
        ops=[dict(type='Resize',size=[resolution,resolution]),dict(type='ConvertPILImage',dtype='float32',scale=True)]
        if split=='train':ops.append(dict(type='ConvertBoxes',fmt='cxcywh',normalize=True))
        d['dataset']['transforms']=dict(type='Compose',ops=ops)
        d['collate_fn']=dict(type='BatchImageCollateFunction',base_size=resolution,base_size_repeat=None,stop_epoch=epochs+1)
    # Fine-tuning budget: AdamW 1e-4 (backbone 5e-5), cosine 20 epochs, no staged reload.
    c['optimizer']['lr']=1e-4
    for group in c['optimizer']['params']:
        if 'lr' in group:group['lr']=5e-5
    c['lr_scheduler']=dict(type='CosineAnnealingLR',T_max=epochs,eta_min=1e-6)
    c.pop('lr_warmup_scheduler',None)
    path=out/'dfine_config.yml';path.write_text(yaml.safe_dump(c,sort_keys=False))
    return path


class DFinePredictor:
    def __init__(self,checkpoint,device,resolution=512):
        from torchvision import transforms as T
        ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
        self.device=device;YAMLConfig=vendor_import()
        cfg=YAMLConfig(str(ROOT/ck['xray_config']))
        cfg.yaml_cfg['HGNetv2']['pretrained']=False
        self.model=cfg.model;self.model.load_state_dict(ck['model'])
        self.model.to(device).eval();self.post=cfg.postprocessor.to(device).eval()
        self.transform=T.Compose([T.Resize((resolution,resolution)),T.ToTensor()])
    @torch.inference_mode()
    def __call__(self,image,threshold=.001):
        im=image.convert('RGB');w,h=im.size
        r=self.post(self.model(self.transform(im).unsqueeze(0).to(self.device)),
                    torch.tensor([[w,h]],device=self.device))[0]
        keep=(r['labels']==0)&(r['scores']>=threshold)
        return np.column_stack([r['boxes'][keep].cpu().numpy(),r['scores'][keep].cpu().numpy()])


def run(a):
    seed_all();out=ROOT/'runs'/a.name;out.mkdir(parents=True,exist_ok=True);data=ROOT/a.data
    YAMLConfig=vendor_import()
    from src.solver import TASKS
    from src.solver.det_engine import train_one_epoch
    import hashlib
    assert hashlib.md5((ROOT/'data/xray_v2/split.csv').read_bytes()).hexdigest()==V2_SPLIT_MD5
    binding=dict(model='dfine',epochs=a.epochs,batch=a.batch,resolution=a.resolution,device=a.device,
        dataset_content_sha256=dataset_digest(data),split_md5=V2_SPLIT_MD5,seed=0,policy=POLICY,
        source={str(p.relative_to(VENDOR)):sha256(p) for p in sorted(VENDOR.rglob('*.py'))},
        code={n:sha256(ROOT/'scripts'/n) for n in ['xray_dfine.py','xray_epoch_validation.py','xray_recovery.py','xray_eval.py']},
        versions={n:importlib.metadata.version(n) for n in ['torch','torchvision','faster-coco-eval']})
    store=RecoveryStore(out/'recovery',binding);saved=store.latest()
    if saved and not a.resume:raise FileExistsError('Use --resume')
    cfg_path=make_config(data,out,a.resolution,a.batch,a.epochs)
    cfg=YAMLConfig(str(cfg_path),device=a.device)
    solver=TASKS[cfg.yaml_cfg['task']](cfg)
    # Load only matching COCO weights; keep the new single-class head initialization.
    pretrained=torch.load(ROOT/'weights/dfine_s_coco.pth',map_location='cpu',weights_only=True)
    state=pretrained['ema']['module'] if 'ema' in pretrained else pretrained['model']
    current=cfg.model.state_dict();matched={k:v for k,v in state.items() if k in current and v.shape==current[k].shape}
    cfg.model.load_state_dict(matched,strict=False)
    solver.train();first=0
    if saved:
        solver.load_state_dict(torch.load(saved['checkpoint'],map_location='cpu',weights_only=True))
        restore_rng(saved['rng']);first=saved['epoch']
    prior=json.loads((out/'execution.json').read_text()) if (out/'execution.json').exists() else {}
    meta=dict(model='dfine',epochs=a.epochs,batch=a.batch,resolution=a.resolution,seed=0,dataset=a.data,
        device_used=a.device,status='running',completed_epochs=first,checkpoint_selection=POLICY,
        pretrained_matching_tensors=len(matched),train_images=len(solver.train_dataloader.dataset),
        optimizer='AdamW',lr=1e-4,lr_backbone=5e-5,weight_decay=1e-4,
        augmentation='precomputed only',pretrained_sha256=sha256(ROOT/'weights/dfine_s_coco.pth'),
        recovery_fingerprint=store.fingerprint)
    start=time.time();atomic_json(out/'execution.json',meta)
    try:
        for epoch in range(first,a.epochs):
            solver.train_dataloader.set_epoch(epoch);t=time.time()
            stats=train_one_epoch(solver.model,solver.criterion,solver.train_dataloader,solver.optimizer,
                solver.device,epoch,use_wandb=False,epochs=a.epochs,max_norm=.1,print_freq=100,
                ema=None,scaler=None,lr_warmup_scheduler=None,writer=None,output_dir=None)
            solver.lr_scheduler.step();solver.last_epoch=epoch
            ck=solver.state_dict();ck['xray_config']=str(cfg_path.relative_to(ROOT));ck['resolution']=a.resolution
            archive=out/'epochs'/f'epoch_{epoch+1:03d}.pth';atomic_torch(archive,ck)
            validate_epoch('dfine',archive,data,out,epoch+1,a.resolution,a.device)
            store.publish(archive,epoch+1,capture_rng(a.device))
            meta['completed_epochs']=epoch+1;meta['checkpoint']=str(archive.relative_to(ROOT))
            atomic_json(out/'progress.json',dict(epoch=epoch+1,epochs=a.epochs,seconds=time.time()-t,stats=stats,verified=True))
            atomic_json(out/'execution.json',meta)
            if a.stop_after_epoch and epoch+1>=a.stop_after_epoch:break
        meta['status']='complete' if meta['completed_epochs']==a.epochs else 'paused'
    except BaseException as e:
        meta.update(status='failed',error=repr(e));raise
    finally:
        meta['seconds']=prior.get('seconds',0)+time.time()-start;atomic_json(out/'execution.json',meta)
    print(json.dumps(meta,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/xray_combined_20261003');p.add_argument('--name',required=True)
    p.add_argument('--epochs',type=int,default=20);p.add_argument('--batch',type=int,default=4)
    p.add_argument('--resolution',type=int,default=512);p.add_argument('--device',default='cuda')
    p.add_argument('--resume',action='store_true');p.add_argument('--stop-after-epoch',type=int,default=0)
    a=p.parse_args()
    if Path(a.name).name!=a.name or a.name in {'.','..'}:p.error('Name must be a folder name')
    with run_lock(ROOT/'runs'/a.name):run(a)

if __name__=='__main__':main()
