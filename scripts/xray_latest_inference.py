"""Infer completed latest YOLO checkpoints; validation timing or frozen test.

No threshold/model selection occurs here. CSV scoring remains in xray_eval.
"""
import argparse
import json
import os
import time
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')
import numpy as np
import pandas as pd
import torch
from PIL import Image
from xray_config import ROOT,DATA
from xray_model_io import seed_all,sync
from xray_predict_controlled import Predictor
from xray_cascade_runtime import canonical_predictions
from xray_secondary_review_live import spec
from xray_eval import PRED_COLUMNS
from xray_recovery import atomic_json,sha256,run_lock


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True)
    p.add_argument('--split',choices=['val','test'],default='val')
    p.add_argument('--repeats',type=int,default=3);a=p.parse_args()
    if a.repeats<1:p.error('repeats must be positive')
    s=spec(a.run);assert sha256(ROOT/s['checkpoint'])==s['sha256']
    out=ROOT/'reports/design_review_20261004/inference'/a.run/a.split
    with run_lock(out):
        policy=dict(spec=s,split=a.split,repeats=a.repeats,device='mps',resolution=512,
                    seed=0,split_sha256=sha256(DATA/'split.csv'))
        if a.split=='test':
            frozen=ROOT/'reports/design_review_20261004/proposal_frozen.json'
            proposal=json.loads(frozen.read_text())
            assert proposal['no_test_based_selection'] and a.run in proposal['test_runs']
            assert proposal['specs'][a.run]==s
            policy['frozen_proposal_sha256']=sha256(frozen)
        pp=out/'policy.json'
        if pp.exists():assert json.loads(pp.read_text())==policy
        else:atomic_json(pp,policy)
        ep=out/'execution.json'
        if ep.exists():
            e=json.loads(ep.read_text());assert e['status']=='complete','Prior incomplete run preserved'
            assert e['policy_sha256']==sha256(pp)
            for f,h in e['csv_sha256'].items():assert sha256(out/f)==h
            print('Verified complete',a.run,a.split);return
        assert torch.backends.mps.is_available();seed_all()
        model=Predictor('yolo',ROOT/s['checkpoint'],'mps',512)
        paths=sorted((DATA/'images'/a.split).glob('*.png'))
        assert len(paths)==(107 if a.split=='val' else 97)
        for path in [paths[0],paths[len(paths)//2],paths[-1]]:
            with Image.open(path) as im:model(im.convert('RGB'))
        sync('mps');calls=[];files=[]
        atomic_json(ep,dict(status='running',policy_sha256=sha256(pp)))
        for repeat in range(a.repeats):
            frames=[]
            for path in paths:
                with Image.open(path) as im:im=im.convert('RGB')
                sync('mps');start=time.perf_counter();raw=model(im);sync('mps')
                elapsed=time.perf_counter()-start
                assert np.isfinite(raw).all()
                frames.append(canonical_predictions(path.stem,raw))
                calls.append(dict(stem=path.stem,repeat=repeat,seconds=elapsed))
            rawdf=pd.concat(frames,ignore_index=True);bad=(rawdf.w<0)|(rawdf.h<0)
            for suffix,df in [('raw',rawdf),('invalid',rawdf[bad]),('preds',rawdf[~bad])]:
                dest=out/f'{suffix}_repeat{repeat}.csv';df[PRED_COLUMNS].to_csv(dest,index=False);files.append(dest)
            print(json.dumps(dict(run=a.run,split=a.split,repeat=repeat,invalid=int(bad.sum()))),flush=True)
        ms=np.array([x['seconds'] for x in calls])*1000
        atomic_json(ep,dict(status='complete',policy_sha256=sha256(pp),calls=calls,
            csv_sha256={f.name:sha256(f) for f in files},mean_ms=float(ms.mean()),
            p95_ms=float(np.quantile(ms,.95)),
            latency_scope='warm model inference and synchronization, excludes PNG decode, camera, masking, queue'))


if __name__=='__main__':main()
