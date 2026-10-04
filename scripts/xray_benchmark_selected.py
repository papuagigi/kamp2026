"""Time four validation-selected detectors on the same device and validation images; no training."""
import os
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image

from xray_config import DATA, ROOT, V2_SPLIT_MD5, select_device
from xray_model_io import seed_all, sync
from xray_selected_common import RUNS, REPORT, verified, freeze, predictor_for
from xray_eval import evaluate




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='mps', choices=['mps', 'cpu'])
    parser.add_argument('--limit', type=int)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--out', default='reports/common_epoch_20261004/evaluation/diagnostics/benchmark_mps.json')
    args = parser.parse_args()
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable in this process. Do not silently substitute CPU.')
    freeze()
    seed_all()
    device = select_device(args.device)
    assert hashlib.md5((DATA / 'split.csv').read_bytes()).hexdigest() == V2_SPLIT_MD5
    man = pd.read_csv(DATA / 'manifest.csv')
    paths = [DATA / 'images/val' / f'{s}.png' for s in sorted(man.loc[man.split == 'val', 'stem'])]
    if args.limit:
        paths = [paths[int(i)] for i in np.linspace(0, len(paths) - 1, args.limit)]
    out = ROOT / args.out
    if out.exists() and not args.resume:
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {'device': device, 'platform': platform.platform(), 'torch_threads': 6,
              'processor': subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string']).decode().strip(),
              'versions': {name: importlib.metadata.version(name) for name in ['torch', 'torchvision', 'ultralytics', 'rfdetr']},
              'split_md5': V2_SPLIT_MD5, 'batch': 1, 'resolution_setting': 512, 'export_threshold': .001,
              'precision': 'FP32, no AMP, existing Predictor adapters, no extra compilation/optimization',
              'scope': 'PNG read, RGB conversion, native resize, model forward, native postprocess, transfer to CPU numpy, frozen score filter, device synchronization',
              'excluded': 'checkpoint loading, 3 warmup calls/model, color-square removal, acquisition, factory communication',
              'resize_caveat': 'Same original images and resolution setting; native resize/padding differs by model. Not equal FLOPs or equal network tensor dimensions.',
              'fallback_enabled': os.environ['PYTORCH_ENABLE_MPS_FALLBACK'],
              'warmup_calls': 3, 'repeats': args.repeats, 'images_per_repeat': len(paths),
              'image_stems': [p.stem for p in paths], 'models': {},
              'started_unix': time.time(), 'status': 'running',
              'limits': 'One Mac session, filesystem cache and thermal/background load not fully controlled; no factory throughput or production latency claim.'}
    if args.resume:
        previous=json.loads(out.read_text())
        for key in ['device','platform','torch_threads','processor','versions','split_md5','batch',
                    'resolution_setting','export_threshold','precision','scope','warmup_calls',
                    'repeats','images_per_repeat','image_stems']:
            assert previous[key]==report[key], ('Resume setting mismatch', key)
        report=previous
        report.setdefault('resumed_unix',[]).append(time.time())
    out.write_text(json.dumps(report, indent=2))
    for run in RUNS:
        metadata, selection, _, _ = verified(run)
        checkpoint = ROOT / selection['checkpoint']
        threshold = selection['metrics']['iou50']['thr']
        if run in report['models']:
            saved=report['models'][run]
            assert saved['checkpoint_sha256']==selection['checkpoint_sha256']
            assert saved['threshold']==threshold and saved['calls']==len(paths)*args.repeats
            assert hashlib.sha256((out.parent/f'{run}_benchmark_validation.csv').read_bytes()).hexdigest()==saved['validation_csv_sha256']
            print('Verified completed benchmark',run,flush=True)
            continue
        print('Loading', run, device, flush=True)
        predictor = predictor_for(metadata, selection, device)
        warmup = [paths[0], paths[len(paths)//2], paths[-1]]
        for path in warmup:
            with Image.open(path) as im:
                predictor(im, threshold=.001)
        sync(device)
        torch.set_num_threads(6)
        # Ultralytics uses an AutoBackend copy for prediction; the source
        # checkpoint object can remain on CPU after an MPS prediction.
        module = predictor.model.predictor.model.model if metadata['model'] == 'yolo' else predictor.model
        while not isinstance(module, torch.nn.Module):
            module = module.model
        devices = sorted({str(p.device) for p in module.parameters()})
        dtypes = sorted({str(p.dtype) for p in module.parameters()})
        assert all(d.startswith(device) for d in devices), devices
        float_dtypes=sorted({str(p.dtype) for p in module.parameters() if p.is_floating_point()})
        assert float_dtypes == ['torch.float32'], float_dtypes
        integer_parameters=[dict(name=n,dtype=str(p.dtype),requires_grad=p.requires_grad)
                            for n,p in module.named_parameters() if not p.is_floating_point()]
        assert not any(p['requires_grad'] for p in integer_parameters), integer_parameters
        timings, kept_counts, pass_means, validation_rows = [], [], [], []
        for repeat in range(args.repeats):
            elapsed = []
            for path in paths:
                sync(device)
                start = time.perf_counter()
                with Image.open(path) as im:
                    predictions = predictor(im, threshold=.001)
                kept = predictions[predictions[:, 4] >= threshold]
                sync(device)
                elapsed.append(time.perf_counter() - start)
                kept_counts.append(len(kept))
                if repeat == 0:
                    for x1,y1,x2,y2,score in predictions:
                        validation_rows.append([path.stem,(x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1,score])
            timings.extend(elapsed)
            pass_means.append(float(np.mean(elapsed)))
            print(run, 'pass', repeat + 1, 'mean_seconds', pass_means[-1], flush=True)
        validation_csv=out.parent/f'{run}_benchmark_validation.csv'
        with validation_csv.open('w') as f:
            writer=csv.writer(f); writer.writerow(['stem','cx','cy','w','h','score']); writer.writerows(validation_rows)
        val_metric,_,_=evaluate(validation_csv,'val',thr=threshold,matching='iou50',stems={p.stem for p in paths})
        report['models'][run] = {'checkpoint': selection['checkpoint'], 'selected_epoch': selection['best_epoch'],
            'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'parameter_devices': devices, 'parameter_dtypes': dtypes, 'threshold': threshold,
            'non_floating_parameters': integer_parameters,
            'calls': len(timings), 'mean_seconds': float(np.mean(timings)),
            'median_seconds': float(np.median(timings)), 'p95_seconds': float(np.percentile(timings, 95)),
            'serial_images_per_second': float(1 / np.mean(timings)),
            'pass_means_seconds': pass_means, 'individual_seconds': timings, 'kept_counts': kept_counts,
            'torch_threads': torch.get_num_threads(),
            'validation_on_benchmark_device': val_metric,
            'training_validation_device': json.loads((ROOT/'runs'/run/'validation'/f"epoch_{selection['best_epoch']:03d}.json").read_text())['device'],
            'validation_csv_sha256': hashlib.sha256(validation_csv.read_bytes()).hexdigest()}
        out.write_text(json.dumps(report, indent=2))
        del predictor, module, predictions, kept
        gc.collect()
        if device == 'mps':
            torch.mps.empty_cache()
    report.update(status='complete', completed_unix=time.time())
    out.write_text(json.dumps(report, indent=2))
    print('Saved', out.relative_to(ROOT), flush=True)


if __name__ == '__main__':
    main()
