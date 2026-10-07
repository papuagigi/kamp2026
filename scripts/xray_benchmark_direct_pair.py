"""Measure the two frozen direct-training models on the same MPS full images."""
import os
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')
import csv
import json
import platform
import time

import numpy as np
import torch
from PIL import Image

from xray_config import ROOT, DATA
from xray_recovery import sha256
from xray_predict_controlled import Predictor
from xray_model_io import seed_all, sync
from xray_cascade_runtime import canonical_predictions, agreement
from xray_eval import evaluate
from xray_rescore_direct import RUNS, OUT


def main():
    assert torch.backends.mps.is_available()
    seed_all();torch.set_num_threads(6);OUT.mkdir(parents=True,exist_ok=True)
    specs={};models={}
    paths=sorted((DATA/'images/val').glob('*.png'));assert len(paths)==107
    for name,run in RUNS.items():
        r=ROOT/'runs'/run;meta=json.loads((r/'execution.json').read_text());s=json.loads((r/'selection.json').read_text())
        assert sha256(ROOT/s['checkpoint'])==s['checkpoint_sha256']
        specs[name]=dict(run=run,checkpoint=s['checkpoint'],checkpoint_sha256=s['checkpoint_sha256'],epoch=s['best_epoch'],threshold=s['metrics']['iou50']['thr'])
        models[name]=Predictor(meta['model'],ROOT/s['checkpoint'],'mps',512)
        for p in [paths[0],paths[len(paths)//2],paths[-1]]:
            with Image.open(p) as im:models[name](im)
        sync('mps');torch.set_num_threads(6)
        obj=models[name].model
        module=obj.predictor.model.model if meta['model']=='yolo' else obj
        while not isinstance(module,torch.nn.Module):module=module.model
        dev=sorted({str(p.device) for p in module.parameters()});dt=sorted({str(p.dtype) for p in module.parameters() if p.is_floating_point()})
        assert all(x.startswith('mps') for x in dev) and dt==['torch.float32']
        specs[name].update(parameter_devices=dev,parameter_dtypes=dt)
    calls=[];exports={n:[] for n in RUNS};names=list(RUNS)
    for repeat in range(3):
        for p in paths:
            sync('mps');begin=time.perf_counter()
            with Image.open(p) as im:image=im.copy()
            preds={};lat={}
            for name in names:
                sync('mps');start=time.perf_counter();raw=models[name](image);sync('mps')
                df=canonical_predictions(p.stem,raw);preds[name]=df[df.score>=specs[name]['threshold']]
                lat[name]=(time.perf_counter()-start)*1000
                if repeat==0:exports[name].extend(df.to_dict('records'))
            a=agreement(preds[names[0]],preds[names[1]])
            total=(time.perf_counter()-begin)*1000
            calls.append(dict(repeat=repeat,stem=p.stem,pair_ms=total,
                yolo_ms=lat[names[0]],rf_ms=lat[names[1]],yolo_boxes=len(preds[names[0]]),rf_boxes=len(preds[names[1]]),**a))
        print('validation pair pass',repeat+1,'mean_ms',np.mean([c['pair_ms'] for c in calls if c['repeat']==repeat]),flush=True)
    import pandas as pd
    pd.DataFrame(calls).to_csv(OUT/'pair_live_calls.csv',index=False)
    validation={}
    for name in names:
        dest=OUT/f'{name}_live_val.csv';pd.DataFrame(exports[name]).to_csv(dest,index=False)
        m,_,_=evaluate(dest,'val',thr=specs[name]['threshold'],matching='iou50')
        validation[name]=dict(metrics=m,csv_sha256=sha256(dest))
    summaries={}
    for key in ['yolo_ms','rf_ms','pair_ms']:
        x=np.array([c[key] for c in calls]);summaries[key]=dict(mean=float(x.mean()),median=float(np.median(x)),p95=float(np.percentile(x,95)),calls=len(x))
    counts=[dict(repeat=k,agree=sum(c['agree'] for c in calls if c['repeat']==k),disagree=sum(not c['agree'] for c in calls if c['repeat']==k)) for k in range(3)]
    result=dict(status='complete',device='mps',platform=platform.platform(),torch=torch.__version__,threads=torch.get_num_threads(),
        batch=1,resolution_setting=512,precision='FP32',fallback_enabled=os.environ['PYTORCH_ENABLE_MPS_FALLBACK'],
        specs=specs,images=107,repeats=3,warmup_calls_per_model=3,timing=summaries,agreement=counts,live_validation=validation,
        timing_scope='Full-image sequential inference for each model including native preprocessing/postprocessing and threshold/CSV-format conversion. Pair includes one image read and result matching. No acquisition, color-square removal, device communication, human review or checkpoint load.',
        limitations='One Mac session. Not factory throughput; two separate models are not statistically independent. Native resize differs. Thresholds unchanged.',
        calls_sha256=sha256(OUT/'pair_live_calls.csv'))
    (OUT/'pair_benchmark.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(timing=summaries,agreement=counts),indent=2),flush=True)


if __name__=='__main__':main()
