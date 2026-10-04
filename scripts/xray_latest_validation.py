"""Validate latest YOLO with transparent invalid-geometry quarantine.

Keep raw model predictions; reject negative-width/height boxes without moving
their coordinates. The existing xray_eval remains the only metric calculator.
"""
import argparse
import csv
import json
import os
import time
from pathlib import Path
import numpy as np
from PIL import Image
from xray_config import ROOT, DATA
from xray_epoch_validation import POLICY, update_selection
from xray_eval import evaluate, PRED_COLUMNS
from xray_model_io import seed_all
from xray_predict_controlled import Predictor
from xray_recovery import atomic_json, sha256

GUARD=dict(version='negative_geometry_quarantine_v1',rule='w < 0 or h < 0',
           action='preserve raw row and quarantine; no coordinate repair',
           zero_area='retained, consistent with existing evaluator',
           nonfinite='fatal error; never silently dropped')


def partition(rows):
    valid=[];invalid=[]
    for r in rows:
        if not np.isfinite(np.asarray(r[1:],dtype=float)).all():raise ValueError('Nonfinite prediction')
        (invalid if r[3]<0 or r[4]<0 else valid).append(r)
    return valid,invalid


def main():
    p=argparse.ArgumentParser()
    for k in ['checkpoint','data','run','device']:p.add_argument('--'+k,required=True)
    p.add_argument('--epoch',type=int,required=True)
    a=p.parse_args();seed_all();start=time.time()
    ck=Path(a.checkpoint);run=Path(a.run);directory=Path(a.data)
    paths=sorted((directory/'images/val').glob('*.png'));assert len(paths)==107
    for path in paths:
        for kind,suffix in [('images','.png'),('labels','.txt')]:
            assert sha256(directory/kind/'val'/(path.stem+suffix))==sha256(DATA/kind/'val'/(path.stem+suffix))
    out=run/'validation';out.mkdir(exist_ok=True)
    guard=out/'geometry_policy.json'
    if guard.exists():assert json.loads(guard.read_text())==GUARD
    else:atomic_json(guard,GUARD)
    pred=out/f'epoch_{a.epoch:03d}.csv';raw=out/f'epoch_{a.epoch:03d}_raw.csv'
    bad=out/f'epoch_{a.epoch:03d}_invalid.csv'
    record=out/f'epoch_{a.epoch:03d}.json'
    if record.exists():
        r=json.loads(record.read_text())
        assert r['checkpoint_sha256']==sha256(ck) and r['predictions_sha256']==sha256(pred)
        update_selection(run,a.epoch);return
    predictor=Predictor('yolo',ck,a.device,512);rows=[]
    for path in paths:
        with Image.open(path) as im:
            for x1,y1,x2,y2,score in predictor(im):
                rows.append([path.stem,(x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1,score])
    valid,invalid=partition(rows)
    for path,values in [(raw,rows),(pred,valid),(bad,invalid)]:
        pending=path.with_suffix('.pending')
        with pending.open('w') as f:
            writer=csv.writer(f);writer.writerow(PRED_COLUMNS);writer.writerows(values)
        os.replace(pending,path)
    metrics={}
    for mode in ['iou50','iou75','legacy']:
        metrics[mode],_,_=evaluate(pred,'val',matching=mode)
    r=dict(epoch=a.epoch,model='yolo',checkpoint=str(ck.relative_to(ROOT)),
           checkpoint_sha256=sha256(ck),predictions_sha256=sha256(pred),raw_sha256=sha256(raw),
           invalid_sha256=sha256(bad),geometry_policy=GUARD,raw_predictions=len(rows),
           invalid_predictions=len(invalid),invalid_images=len({x[0] for x in invalid}),
           highest_invalid_score=max((float(x[5]) for x in invalid),default=None),
           metrics=metrics,seconds=time.time()-start,device=a.device,policy=POLICY,images=len(paths))
    atomic_json(record,r);s=update_selection(run,a.epoch)
    print(json.dumps(dict(epoch=a.epoch,best_epoch=s['best_epoch'],f1=metrics['iou50']['f1'],
                         invalid_predictions=len(invalid),seconds=r['seconds'])),flush=True)


if __name__=='__main__':main()
