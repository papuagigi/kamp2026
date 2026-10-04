"""Export frozen YOLO predictions on original and flipped validation inputs."""
import csv
import hashlib
import json
import time

import pandas as pd
import torch
from PIL import Image, ImageOps

from xray_cascade_audit import OUT, save, sha
from xray_config import DATA, ROOT, V2_SPLIT_MD5
from xray_model_io import seed_all, sync
from xray_predict_controlled import Predictor
from xray_selected_common import RUNS, REPORT


def main():
    assert hashlib.md5((DATA/'split.csv').read_bytes()).hexdigest() == V2_SPLIT_MD5
    frozen=json.loads((REPORT/'frozen_selection.json').read_text())['runs'][RUNS[0]]
    checkpoint=ROOT/frozen['checkpoint']
    assert sha(checkpoint)==frozen['checkpoint_sha256']
    directory=OUT/'consistency_native_csv';directory.mkdir(exist_ok=True)
    policy=dict(version='flip_consistency_v2_native_csv',split='val',run=RUNS[0],device='mps',seed=0,
        checkpoint_sha256=sha(checkpoint),threshold=frozen['thresholds']['iou50'],
        export_threshold=.001,resolution=512,box_correspondence_iou=.5,
        export_precision='Preserve NumPy scalars, arithmetic and csv.writer formatting from xray_benchmark_selected.py; no Python float widening before export',
        route='No accepted original box OR unmatched box between original and restored horizontal-flip output. One-to-one xray_eval.match; GT not used to route.',
        test_access=False,new_training=False,development_validation_reuse=True,
        previous_test_exposure=True)
    p=directory/'policy.json'
    if p.exists():assert json.loads(p.read_text())==policy
    else:save(p,policy)
    if (directory/'execution.json').exists():
        old=json.loads((directory/'execution.json').read_text())
        assert old['status']=='complete' and old['policy_sha256']==sha(p)
        for name,digest in old['csv_sha256'].items():assert sha(directory/name)==digest
        print('Verified existing completed consistency inference');return
    if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
    seed_all();torch.set_num_threads(6)
    predictor=Predictor('yolo',checkpoint,'mps',512)
    man=pd.read_csv(DATA/'manifest.csv');man=man[man.split=='val'].sort_values('stem')
    assert len(man)==107
    for r in [man.iloc[0],man.iloc[len(man)//2],man.iloc[-1]]:
        with Image.open(DATA/'images/val'/f'{r.stem}.png') as im:predictor(im)
    sync('mps');torch.set_num_threads(6)
    module=predictor.model.predictor.model.model
    devices=sorted({str(x.device) for x in module.parameters()})
    assert all(x.startswith('mps') for x in devices)
    record=dict(status='running',policy_sha256=sha(p),device=devices,images=107)
    save(directory/'execution.json',record)
    rows={'original':[], 'hflip':[]};timing=[]
    for r in man.itertuples():
        with Image.open(DATA/'images/val'/f'{r.stem}.png') as source:
            original=source.convert('RGB')
        times={}
        for mode in rows:
            sync('mps');start=time.perf_counter()
            im=original if mode=='original' else ImageOps.mirror(original)
            pred=predictor(im,threshold=.001);sync('mps')
            times[mode]=time.perf_counter()-start
            for x1,y1,x2,y2,score in pred:
                cx=(x1+x2)/2
                if mode=='hflip':cx=original.width-cx
                rows[mode].append([r.stem,cx,(y1+y2)/2,x2-x1,y2-y1,score])
        timing.append(dict(stem=r.stem,**times))
    paths=[]
    for mode,values in rows.items():
        path=directory/f'{mode}_val.csv'
        with path.open('w') as f:
            w=csv.writer(f);w.writerow(['stem','cx','cy','w','h','score']);w.writerows(values)
        paths.append(path)
    record.update(status='complete',csv_sha256={p.name:sha(p) for p in paths},timings=timing,
        latency_scope='warmed YOLO, MPS synchronization; includes mirroring for hflip; excludes image read and factory processing; one repeat')
    save(directory/'execution.json',record)
    print(json.dumps({'status':'complete','images':107,'passes':2,'device':devices,
        'original_seconds':sum(t['original'] for t in timing), 'hflip_seconds':sum(t['hflip'] for t in timing)},indent=2))


if __name__=='__main__':main()
