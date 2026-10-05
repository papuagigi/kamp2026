"""Label-free candidate and complete-coverage role inference, no product release."""
import argparse
import csv
import hashlib
import json
import os
import time
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')
import numpy as np
import pandas as pd
from PIL import Image
from xray_config import ROOT, DATA, V2_SPLIT_MD5
from xray_predict_controlled import Predictor
from xray_model_io import seed_all, sync
from xray_review_geometry import around, tiles, remap, suppress
from xray_recovery import atomic_json, sha256

BASE=ROOT/'reports/review_roles_20261005'
FROZEN=ROOT/'reports/common_epoch_20261004/evaluation/frozen_selection.json'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model', choices=['rfdetr','dfine'],default='rfdetr')
    p.add_argument('--checkpoint')
    p.add_argument('--name',required=True)
    p.add_argument('--role',choices=['candidate','miss','both'],default='both')
    args=p.parse_args();seed_all()
    assert hashlib.md5((DATA/'split.csv').read_bytes()).hexdigest()==V2_SPLIT_MD5
    frozen=json.loads(FROZEN.read_text())['runs']
    run='v2_common20_rfdetr_mps_20261004' if args.model=='rfdetr' else 'v2_common20_dfine_cuda_20261004'
    checkpoint=ROOT/(args.checkpoint or frozen[run]['checkpoint'])
    out=BASE/args.name
    if (out/'execution.json').exists():raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    policy=dict(first='YOLOv8n',review=args.model,checkpoint=str(checkpoint.relative_to(ROOT)),
        checkpoint_sha256=sha256(checkpoint),first_threshold=frozen['v2_common20_yolo_mps_20261004']['thresholds']['iou50'],
        review_frozen_threshold=frozen[run]['thresholds']['iou50'],candidate_side=96,tile_side=256,tile_overlap=64,
        role=args.role,split_md5=V2_SPLIT_MD5,split='val',no_test=True,no_training=True,
        gate='Preserve all first alarms. No auto-PASS. Candidate disagreement is not TP/FP truth.',
        forced_miss_branch='Run complete-coverage mode on all validation images separately from routed results.')
    atomic_json(out/'policy.json',policy)
    first_path=ROOT/'runs/v2_common20_yolo_mps_20261004/validation/epoch_020.csv'
    first=pd.read_csv(first_path);first=first[first.score>=policy['first_threshold']]
    policy['first_predictions_sha256']=sha256(first_path)
    atomic_json(out/'policy.json',policy)
    if args.model=='dfine':
        from xray_dfine import DFinePredictor
        predictor=DFinePredictor(checkpoint,'mps',512)
    else:predictor=Predictor('rfdetr',checkpoint,'mps',512)
    paths=sorted((DATA/'images/val').glob('*.png'))
    with Image.open(paths[0]) as warm:
        for _ in range(3):predictor(warm)
    timings=[];stems=[];routes=[];frames={'candidate':[], 'miss':[]}
    roles=['candidate','miss'] if args.role=='both' else [args.role]
    for i,path in enumerate(paths):
        im=Image.open(path).convert('RGB');a=first[first.stem==path.stem]
        routes.append(dict(stem=path.stem,first_boxes=len(a),route='candidate' if len(a) else 'miss'))
        for role in roles:
            windows=(list(dict.fromkeys(around(r.cx,r.cy,im.width,im.height) for r in a.itertuples()))
                     if role=='candidate' else tiles(im.width,im.height))
            sync('mps');start=time.perf_counter();found=[]
            for window in windows:found.extend(remap(predictor(im.crop(window)),window))
            boxes=suppress(found);sync('mps')
            timings.append(dict(stem=path.stem,role=role,calls=len(windows),seconds=time.perf_counter()-start))
            for x1,y1,x2,y2,score in boxes:
                frames[role].append([path.stem,(x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1,score])
        stems.append(path.stem)
        if (i+1)%25==0:print(args.name,i+1,len(paths),flush=True)
    for role in roles:
        with (out/f'{role}.csv').open('w') as f:
            w=csv.writer(f);w.writerow(['stem','cx','cy','w','h','score']);w.writerows(frames[role])
    atomic_json(out/'execution.json',dict(status='complete',images=len(paths),routes=routes,timings=timings,
        means_ms={role:float(np.mean([x['seconds'] for x in timings if x['role']==role])*1000) for role in roles},
        stems=stems,prediction_hashes={role:sha256(out/f'{role}.csv') for role in roles},
        scope='Review crop/tiling, inference, remapping, NMS and synchronization; excludes first detector and I/O. One pass, exploratory timing.'))
    print('Saved',str(out.relative_to(ROOT)),flush=True)

if __name__=='__main__':main()
