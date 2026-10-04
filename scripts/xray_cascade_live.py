"""Run resident YOLO + conditional RF-DETR on validation images, without labels."""
import os
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')
import hashlib
import json
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
from PIL import Image,ImageOps

from xray_cascade_audit import save,sha
from xray_cascade_runtime import canonical_predictions,agreement,should_refer,outcome
from xray_config import ROOT,DATA,V2_SPLIT_MD5
from xray_eval import PRED_COLUMNS
from xray_model_io import seed_all,sync
from xray_predict_controlled import Predictor
from xray_selected_common import REPORT,RUNS

OUT=REPORT/'cascade_live'


def main():
    assert hashlib.md5((DATA/'split.csv').read_bytes()).hexdigest()==V2_SPLIT_MD5
    OUT.mkdir(exist_ok=True)
    frozen=json.loads((REPORT/'frozen_selection.json').read_text())['runs']
    runs=[RUNS[0],RUNS[2]]
    for r in runs:assert sha(ROOT/frozen[r]['checkpoint'])==frozen[r]['checkpoint_sha256']
    policy=dict(version='live_cascade_v1',split='val',device='mps',seed=0,repeats=3,
        first=runs[0],second=runs[1],thresholds={r:frozen[r]['thresholds']['iou50'] for r in runs},
        checkpoints={r:frozen[r]['checkpoint_sha256'] for r in runs},resolution=512,export_threshold=.001,
        route='No original box or unmatched original/undo-horizontal-flip predictions, one-to-one IoU50.',
        second_input='Full original image; no ROI-only inference or tiled inference in this prototype.',
        conflict='Retain both outputs and mark for reinspection; never clear first alarm.',
        deadline_seconds=1.0,deadline_role='Development soft deadline checked after synchronous calls; NOT a factory requirement or a hard watchdog.',
        no_pass=True,no_training=True,no_test=True,validation_reused=True,
        precision='Native NumPy CSV serialization then common pandas float parsing, as existing evaluation.',
        split_md5=V2_SPLIT_MD5)
    pp=OUT/'policy.json'
    if pp.exists():assert json.loads(pp.read_text())==policy
    else:save(pp,policy)
    ep=OUT/'execution.json'
    if ep.exists():
        old=json.loads(ep.read_text())
        if old['status']=='complete':
            assert old['policy_sha256']==sha(pp)
            for f,h in old['csv_sha256'].items():assert sha(OUT/f)==h
            print('Verified existing completed live cascade');return
        raise RuntimeError('Existing incomplete execution requires explicit recovery inspection')
    if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
    seed_all();torch.set_num_threads(6)
    start=time.perf_counter()
    first=Predictor('yolo',ROOT/frozen[runs[0]]['checkpoint'],'mps',512)
    second=Predictor('rfdetr',ROOT/frozen[runs[1]]['checkpoint'],'mps',512)
    loading=time.perf_counter()-start
    paths=sorted((DATA/'images/val').glob('*.png'));assert len(paths)==107
    for path in [paths[0],paths[len(paths)//2],paths[-1]]:
        with Image.open(path) as im:first(im);second(im)
    sync('mps');torch.set_num_threads(6)
    def memory():return dict(allocated_bytes=torch.mps.current_allocated_memory(),driver_bytes=torch.mps.driver_allocated_memory())
    record=dict(status='running',policy_sha256=sha(pp),loading_seconds=loading,
        resident_after_warmup=memory(),device='mps',calls=[],completed_repeats=0)
    save(ep,record)
    csv_paths=[]
    for repeat in range(policy['repeats']):
        tables={k:[] for k in ['original','hflip','second']}
        for path in paths:
            sync('mps');start=time.perf_counter()
            a=pd.DataFrame(columns=PRED_COLUMNS);b=None;c=None;referred=False;error=None;parts={}
            try:
                with Image.open(path) as im:im=im.convert('RGB')
                t=time.perf_counter();raw=first(im);sync('mps')
                ta=canonical_predictions(path.stem,raw);a=ta[ta.score>=policy['thresholds'][runs[0]]]
                parts['original_seconds']=time.perf_counter()-t;tables['original'].append(ta)
                t=time.perf_counter();raw=first(ImageOps.mirror(im));sync('mps')
                tb=canonical_predictions(path.stem,raw,im.width);b=tb[tb.score>=policy['thresholds'][runs[0]]]
                parts['flip_seconds']=time.perf_counter()-t;tables['hflip'].append(tb)
                referred=should_refer(a,b)
                if referred:
                    t=time.perf_counter();raw=second(im);sync('mps')
                    tc=canonical_predictions(path.stem,raw);c=tc[tc.score>=policy['thresholds'][runs[1]]]
                    parts['second_seconds']=time.perf_counter()-t;tables['second'].append(tc)
                    pair=agreement(a,c)
                else:pair=None
                error=None
            except Exception as exc:
                error=type(exc).__name__+': '+str(exc);pair=None
            elapsed=time.perf_counter()-start
            state=outcome(a,c,referred,elapsed,policy['deadline_seconds'],error)
            record['calls'].append(dict(repeat=repeat,stem=path.stem,referred=referred,state=state,
                seconds=elapsed,parts=parts,error=error,first_boxes=len(a),second_boxes=len(c) if c is not None else None,
                comparison=pair,first_predictions=a.to_dict(orient='records'),
                second_predictions=c.to_dict(orient='records') if c is not None else None,memory=memory()))
        for kind,frames in tables.items():
            dest=OUT/f'{kind}_repeat{repeat}_val.csv'
            (pd.concat(frames,ignore_index=True) if frames else pd.DataFrame(columns=PRED_COLUMNS)).to_csv(dest,index=False)
            csv_paths.append(dest)
        record['completed_repeats']=repeat+1;save(ep,record)
        current=[r for r in record['calls'] if r['repeat']==repeat]
        print(json.dumps({'repeat':repeat,'images':len(current),'referred':sum(r['referred'] for r in current),
                          'states':dict(Counter(r['state'] for r in current)),
                          'mean_ms':np.mean([r['seconds'] for r in current])*1000}),flush=True)
    record.update(status='complete',csv_sha256={p.name:sha(p) for p in csv_paths},
        latency_scope='Warm resident models, PNG read, conversion, original+flip forward, canonical score conversion, routing, conditional RF full-image forward, disagreement comparison, synchronization. Excludes model load, warmup, telemetry write, color removal, camera, factory I/O and queue.',
        memory_scope='MPS allocator and driver snapshots at warmup and each image completion; not exact peak RSS or peak GPU usage.')
    save(ep,record)


if __name__=='__main__':main()
