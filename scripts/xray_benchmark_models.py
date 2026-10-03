"""Benchmark all final models sequentially after training, using the same images/device."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from xray_config import ROOT,DATA,select_device
from xray_model_io import seed_all,sync
from xray_predict_controlled import Predictor

REPORT=ROOT/'reports/roadmap_20261002'


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='auto');a=p.parse_args();seed_all();device=select_device(a.device)
    final=json.loads((REPORT/'final_comparison.json').read_text());man=pd.read_csv(DATA/'manifest.csv')
    paths=[DATA/'images/val'/f'{s}.png' for s in sorted(man.loc[man.split=='val','stem'])]
    results={}
    for row in final:
        meta=json.loads((ROOT/'runs'/row['run']/'execution.json').read_text())
        predictor=Predictor(meta['model'],ROOT/meta['checkpoint'],device,meta['resolution'])
        for path in paths[:3]:
            with Image.open(path) as im:predictor(im)
        sync(device);times=[]
        for path in paths:
            start=time.perf_counter()
            with Image.open(path) as im:predictor(im)
            sync(device);times.append(time.perf_counter()-start)
        results[row['run']]={'device':device,'images':len(paths),'mean_seconds':float(np.mean(times)),
            'p95_seconds':float(np.percentile(times,95)),'warmup_images':3,'individual_seconds':times,
            'scope':'image open/convert + model inference + postprocess; excludes masking and factory communication',
            'image_stems':[p.stem for p in paths],'training_complete_before_measurement':True}
        print(row['run'],results[row['run']]['mean_seconds'],flush=True)
        del predictor
        if device=='mps':torch.mps.empty_cache()
    (REPORT/'inference_benchmark.json').write_text(json.dumps(results,indent=2))

if __name__=='__main__':main()
