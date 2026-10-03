"""Time completed detectors on the same device and validation images; no training."""
import os
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import argparse
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
from xray_predict_controlled import Predictor

RUNS = ['v2_mix2453_yolo_mps', 'v2_mix2453_faster_cuda', 'v2_mix2453_rfdetr_mps']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='mps', choices=['mps', 'cpu'])
    parser.add_argument('--limit', type=int)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--out', default='reports/overnight_20261003/diagnostics/benchmark_mps.json')
    args = parser.parse_args()
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable in this process. Do not silently substitute CPU.')
    seed_all()
    device = select_device(args.device)
    assert hashlib.md5((DATA / 'split.csv').read_bytes()).hexdigest() == V2_SPLIT_MD5
    man = pd.read_csv(DATA / 'manifest.csv')
    paths = [DATA / 'images/val' / f'{s}.png' for s in sorted(man.loc[man.split == 'val', 'stem'])]
    if args.limit:
        paths = [paths[int(i)] for i in np.linspace(0, len(paths) - 1, args.limit)]
    out = ROOT / args.out
    if out.exists():
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
    out.write_text(json.dumps(report, indent=2))
    for run in RUNS:
        metadata = json.loads((ROOT / 'runs' / run / 'execution.json').read_text())
        assert metadata['status'] == 'complete' and metadata['resolution'] == 512
        checkpoint = ROOT / metadata['checkpoint']
        evaluation = json.loads((ROOT / 'reports/overnight_20261003' / f'{run}_evaluation.json').read_text())
        threshold = evaluation['metrics']['iou50']['val']['thr']
        print('Loading', run, device, flush=True)
        predictor = Predictor(metadata['model'], checkpoint, device, 512)
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
        assert dtypes == ['torch.float32'], dtypes
        timings, kept_counts, pass_means = [], [], []
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
            timings.extend(elapsed)
            pass_means.append(float(np.mean(elapsed)))
            print(run, 'pass', repeat + 1, 'mean_seconds', pass_means[-1], flush=True)
        report['models'][run] = {'checkpoint': metadata['checkpoint'],
            'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'parameter_devices': devices, 'parameter_dtypes': dtypes, 'threshold': threshold,
            'calls': len(timings), 'mean_seconds': float(np.mean(timings)),
            'median_seconds': float(np.median(timings)), 'p95_seconds': float(np.percentile(timings, 95)),
            'serial_images_per_second': float(1 / np.mean(timings)),
            'pass_means_seconds': pass_means, 'individual_seconds': timings, 'kept_counts': kept_counts,
            'torch_threads': torch.get_num_threads()}
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
