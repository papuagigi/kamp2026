"""Export one trained detector's predictions to the existing common CSV format."""
import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from xray_config import ROOT,DATA,select_device
from xray_model_io import faster_model,sync,seed_all


class Predictor:
    def __init__(self,kind,checkpoint,device='auto',resolution=512):
        self.kind=kind;self.device=select_device(device);self.resolution=resolution
        if kind=='yolo':
            from ultralytics import YOLO
            self.model=YOLO(str(checkpoint))
        elif kind=='faster':
            ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
            self.model=faster_model(False,ck.get('resolution',resolution))
            self.model.load_state_dict(ck['model']);self.model.to(self.device).eval()
        else:
            from rfdetr import RFDETRSmall
            self.model=RFDETRSmall.from_checkpoint(str(checkpoint),device=self.device,resolution=resolution,amp=False,fused_optimizer=False)

    @torch.inference_mode()
    def __call__(self,image,threshold=.001):
        image=image.convert('RGB')
        if self.kind=='yolo':
            r=self.model.predict(image,imgsz=self.resolution,device=self.device,conf=threshold,iou=.5,max_det=50,verbose=False)[0]
            return np.column_stack([r.boxes.xyxy.cpu().numpy(),r.boxes.conf.cpu().numpy()])
        if self.kind=='faster':
            r=self.model([pil_to_tensor(image).float().to(self.device)/255])[0]
            keep=(r['labels']==1)&(r['scores']>=threshold)
            return np.column_stack([r['boxes'][keep].cpu().numpy(),r['scores'][keep].cpu().numpy()])
        r=self.model.predict(image,threshold=threshold)
        keep=r.class_id==0
        return np.column_stack([r.xyxy[keep],r.confidence[keep]])


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--split',default='val')
    p.add_argument('--images');p.add_argument('--out');p.add_argument('--device',default='auto')
    a=p.parse_args();seed_all();run=ROOT/'runs'/a.run;meta=json.loads((run/'execution.json').read_text())
    if meta['status']!='complete':raise RuntimeError('Training run is not complete')
    checkpoint=ROOT/meta['checkpoint'];device=select_device(a.device)
    predictor=Predictor(meta['model'],checkpoint,device,meta['resolution'])
    directory=ROOT/a.images if a.images else DATA/'images'/a.split
    paths=sorted(directory.glob('*.png'));assert paths
    out=ROOT/a.out if a.out else ROOT/'reports'/f'preds_{a.run}_{a.split}.csv'
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True);timings=[];start=time.time()
    with out.open('w') as f:
        w=csv.writer(f);w.writerow(['stem','cx','cy','w','h','score'])
        for i,path in enumerate(paths):
            t=time.time();im=Image.open(path);res=predictor(im);sync(device);timings.append(time.time()-t)
            for x1,y1,x2,y2,score in res:w.writerow([path.stem,(x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1,score])
            if (i+1)%100==0:print(f'{i+1}/{len(paths)} images',flush=True)
    (out.with_suffix('.run.json')).write_text(json.dumps(dict(model=meta['model'],run=a.run,device=device,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),images=len(paths),seconds=time.time()-start,
        mean_seconds=float(np.mean(timings[1:])),p95_seconds=float(np.percentile(timings[1:],95)),
        latency_scope='image open/convert + inference + postprocess; excludes color masking; first image warmup excluded',
        image_stems=[p.stem for p in paths],resolution=meta['resolution']),indent=2))
    print('Saved',str(out.relative_to(ROOT)),flush=True)


if __name__=='__main__':main()
