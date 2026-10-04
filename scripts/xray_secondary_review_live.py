"""Independent secondary inspection, all-image or flip-gated validation only.

Explicit outputs are preserved. This is an offline prototype, not factory control.
"""
import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')
import numpy as np
import pandas as pd
import torch
from PIL import Image,ImageOps
from xray_config import ROOT,DATA
from xray_model_io import seed_all,sync
from xray_predict_controlled import Predictor
from xray_cascade_runtime import canonical_predictions,agreement
from xray_secondary_review_runtime import review_required,review_outcome
from xray_selected_common import REPORT,RUNS
from xray_eval import PRED_COLUMNS
from xray_recovery import atomic_json,sha256,run_lock


def spec(run):
    frozen=json.loads((REPORT/'frozen_selection.json').read_text())['runs']
    if run in frozen:
        s=frozen[run]
        return dict(run=run,model=s['model'],checkpoint=s['checkpoint'],
                    sha256=s['checkpoint_sha256'],threshold=s['thresholds']['iou50'])
    directory=ROOT/'runs'/run
    execution=json.loads((directory/'execution.json').read_text())
    assert execution['status']=='complete' and execution['model']=='yolo'
    s=json.loads((directory/'selection.json').read_text())
    assert s['completed_epochs']==20
    return dict(run=run,model='yolo',checkpoint=s['checkpoint'],sha256=s['checkpoint_sha256'],
                threshold=s['metrics']['iou50']['thr'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--first-run',default=RUNS[0])
    p.add_argument('--second',choices=['faster','rfdetr','dfine'],required=True)
    p.add_argument('--scope',choices=['all','flip'],default='all')
    p.add_argument('--name',required=True);p.add_argument('--repeats',type=int,default=3)
    a=p.parse_args()
    if Path(a.name).name!=a.name or a.name in ('.','..') or a.repeats<1:p.error('Invalid name/repeats')
    out=ROOT/'reports/design_review_20261004'/a.name
    with run_lock(out):
        first=spec(a.first_run);second=spec(RUNS[{'faster':1,'rfdetr':2,'dfine':3}[a.second]])
        assert first['model']=='yolo'
        for s in [first,second]:assert sha256(ROOT/s['checkpoint'])==s['sha256']
        policy=dict(first=first,second=second,scope=a.scope,repeats=a.repeats,device='mps',
                    no_test=True,no_pass=True,validation_reused=True,seed=0,resolution=512,
                    invalid_geometry='preserve raw outputs; negative or nonfinite -> reinspection error',
                    deadline=None,deadline_reason='No factory limit supplied; offline accuracy comparison',
                    on_conflict='retain both and refer',agreement_is_truth=False,
                    split_sha256=sha256(DATA/'split.csv'))
        pp=out/'policy.json'
        if pp.exists():assert json.loads(pp.read_text())==policy
        else:atomic_json(pp,policy)
        ep=out/'execution.json'
        if ep.exists():
            old=json.loads(ep.read_text())
            if old['status']=='complete':
                assert old['policy_sha256']==sha256(pp)
                for f,h in old['csv_sha256'].items():assert sha256(out/f)==h
                print('Completed run verified',a.name);return
            raise RuntimeError('Incomplete prior run preserved; inspect before restarting')
        assert torch.backends.mps.is_available();seed_all()
        start=time.perf_counter()
        first_model=Predictor('yolo',ROOT/first['checkpoint'],'mps',512)
        if a.second=='dfine':
            from xray_dfine import DFinePredictor
            second_model=DFinePredictor(ROOT/second['checkpoint'],'mps',512)
        else:second_model=Predictor(a.second,ROOT/second['checkpoint'],'mps',512)
        loading=time.perf_counter()-start
        paths=sorted((DATA/'images/val').glob('*.png'));assert len(paths)==107
        for path in [paths[0],paths[53],paths[-1]]:
            with Image.open(path) as im:first_model(im);second_model(im)
        sync('mps')
        record=dict(status='running',policy_sha256=sha256(pp),loading_seconds=loading,calls=[])
        atomic_json(ep,record);csvs=[]
        for repeat in range(a.repeats):
            tables={k:[] for k in ['first','flip','second']}
            for path in paths:
                sync('mps');start=time.perf_counter();error=None;b=None;flip=None;invalid=[]
                orig=pd.DataFrame(columns=PRED_COLUMNS);referred=a.scope=='all'
                def predict(model,kind,im,threshold,mirror=False):
                    raw=model(im)
                    if not np.isfinite(raw).all():raise ValueError('Nonfinite predictions')
                    table=canonical_predictions(path.stem,raw,im.width if mirror else None)
                    tables[kind].append(table)
                    bad=(table.w<0)|(table.h<0)
                    if bad.any():
                        invalid.extend([dict(model=kind,**r) for r in table[bad].to_dict(orient='records')])
                    return table[~bad & (table.score>=threshold)]
                try:
                    with Image.open(path) as im:im=im.convert('RGB')
                    orig=predict(first_model,'first',im,first['threshold'])
                    if a.scope=='flip':flip=predict(first_model,'flip',ImageOps.mirror(im),first['threshold'],True)
                    referred=review_required(a.scope,orig,flip)
                    if invalid:referred=True
                    if referred:b=predict(second_model,'second',im,second['threshold'])
                    if invalid:error='Invalid box geometry preserved'
                    sync('mps')
                except Exception as exc:error=type(exc).__name__+': '+str(exc)
                elapsed=time.perf_counter()-start
                state=review_outcome(orig,b,referred,elapsed,error=error)
                record['calls'].append(dict(repeat=repeat,stem=path.stem,referred=referred,seconds=elapsed,
                    state=state,error=error,invalid=invalid,first_predictions=orig.to_dict(orient='records'),
                    second_predictions=None if b is None else b.to_dict(orient='records'),
                    comparison=None if b is None else agreement(orig,b)))
            for kind,frames in tables.items():
                dest=out/f'{kind}_repeat{repeat}_val.csv'
                (pd.concat(frames,ignore_index=True) if frames else pd.DataFrame(columns=PRED_COLUMNS)).to_csv(dest,index=False)
                csvs.append(dest)
            atomic_json(ep,record)
            calls=[x for x in record['calls'] if x['repeat']==repeat]
            print(json.dumps(dict(repeat=repeat,states=dict(Counter(x['state'] for x in calls)),mean_ms=float(np.mean([x['seconds'] for x in calls])*1000))),flush=True)
        times=np.array([x['seconds'] for x in record['calls']])*1000
        record.update(status='complete',csv_sha256={p.name:sha256(p) for p in csvs},
            mean_ms=float(times.mean()),p95_ms=float(np.quantile(times,.95)),
            latency_scope='warm resident PNG read, inference, correspondence, synchronization; excludes masking, camera, equipment, queue, telemetry I/O',
            no_factory_deadline_or_hard_watchdog=True,operational_f1=None)
        atomic_json(ep,record)


if __name__=='__main__':main()
