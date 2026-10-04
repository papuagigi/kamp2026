"""Read trusted local training logs/checkpoints; no training, inference or scoring."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/training_history_20261004'
RUNS = {'yolo': 'v2_mix2453_yolo_mps', 'faster': 'v2_mix2453_faster_cuda', 'rfdetr': 'v2_mix2453_rfdetr_mps'}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summaries, curves, sources = {}, [], []
    for model, run in RUNS.items():
        folder = ROOT / 'runs' / run
        execution = json.loads((folder / 'execution.json').read_text())
        assert execution['completed_epochs'] == 20
        assert execution['checkpoint_selection'] == 'fixed final epoch; no test-based selection'
        if model == 'yolo':
            path = folder / 'results.csv'
            d = pd.read_csv(path).rename(columns=str.strip)
            table = pd.DataFrame({'epoch': d.epoch, 'train_loss': d['train/box_loss'],
                                  'val_loss': d['val/box_loss'], 'native_map_50_95': d['metrics/mAP50-95(B)'],
                                  'native_map_50': d['metrics/mAP50(B)']})
            loss_definition = 'Ultralytics box_loss component, not total model loss'
        elif model == 'rfdetr':
            path = folder / 'metrics.csv'
            raw = pd.read_csv(path)
            # Lightning writes train and validation epoch aggregates in separate rows.
            assert raw.dropna(subset=['val/mAP_50_95']).epoch.is_unique
            assert raw.dropna(subset=['train/loss']).epoch.is_unique
            d = raw.groupby('epoch').last(numeric_only=True).reset_index()
            table = pd.DataFrame({'epoch': d.epoch + 1, 'train_loss': d['train/loss'],
                                  'val_loss': np.nan, 'native_map_50_95': d['val/mAP_50_95'],
                                  'native_map_50': d['val/mAP_50'], 'native_map_75': d['val/mAP_75']})
            loss_definition = 'RF-DETR logged train/loss; compute_val_loss=False'
        else:
            path = folder / 'history.json'
            d = pd.DataFrame(json.loads(path.read_text()))
            table = pd.DataFrame({'epoch': d.epoch, 'train_loss': d.loss, 'val_loss': np.nan,
                                  'native_map_50_95': np.nan, 'native_map_50': np.nan})
            loss_definition = 'Mean training total loss per epoch; validation was not run per epoch'
        assert table.epoch.tolist() == list(range(1, 21))
        assert table.train_loss.notna().all()
        table.insert(0, 'model', model)
        curves.append(table)
        first, last = table.iloc[0], table.iloc[-1]
        def row(r):
            return {k: (None if pd.isna(r[k]) else int(r[k]) if k == 'epoch' else float(r[k]))
                    for k in ['epoch', 'train_loss', 'val_loss', 'native_map_50_95', 'native_map_50']}
        best = table.loc[table.native_map_50_95.idxmax()] if table.native_map_50_95.notna().any() else None
        summary = {'run': run, 'loss_definition': loss_definition, 'first': row(first), 'last': row(last),
                   'native_best': row(best) if best is not None else None,
                   'per_epoch_validation': best is not None,
                   'checkpoint_selection_used': execution['checkpoint_selection']}
        if model == 'yolo':
            p = folder / 'weights/best.pt'
            # Our own Ultralytics checkpoints contain a Python model object.
            ck = torch.load(p, map_location='cpu', weights_only=False)
            saved = ck['train_metrics']
            keys = ['metrics/mAP50-95(B)', 'metrics/mAP50(B)', 'val/box_loss']
            candidates = d[np.logical_and.reduce([np.isclose(d[k], saved[k], atol=1e-7, rtol=0) for k in keys])]
            assert len(candidates) == 1
            summary['best_checkpoint'] = {'path': str(p.relative_to(ROOT)), 'saved_epoch_field': int(ck['epoch']),
                                          'epoch_inferred_from_unique_metrics': int(candidates.iloc[0].epoch),
                                          'note': 'Ultralytics stripped epoch to -1; unique stored metric tuple identifies epoch.'}
            del ck
        elif model == 'rfdetr':
            p = folder / 'checkpoint_best_regular.pth'
            ck = torch.load(p, map_location='cpu', weights_only=True)
            summary['best_checkpoint'] = {'path': str(p.relative_to(ROOT)), 'epoch_one_based': int(ck['epoch']) + 1}
            assert summary['best_checkpoint']['epoch_one_based'] == summary['native_best']['epoch']
            del ck
        else:
            p = folder / 'last.pt'
            ck = torch.load(p, map_location='cpu', weights_only=True)
            assert ck['epoch'] == 20
            saved = sorted(str(q.relative_to(folder)) for q in folder.rglob('*') if q.suffix in {'.pt', '.pth', '.ckpt'})
            summary['saved_checkpoints'] = saved
            summary['last_checkpoint_epoch'] = int(ck['epoch'])
            summary['best_checkpoint'] = None
            del ck
        for source in [path, folder / 'execution.json', p]:
            sources.append({'path': str(source.relative_to(ROOT)), 'sha256': digest(source)})
        summaries[model] = summary
    table = pd.concat(curves, ignore_index=True)
    table.to_csv(OUT / 'epoch_curves.csv', index=False)
    report = {'scope': 'Read saved full-mixture training histories; no new predictions, threshold search, scoring or training',
              'models': summaries, 'sources': sources,
              'limitations': ['Native metrics from different libraries are not the project common scorer.',
                             'A lower late validation metric is compatible with overfitting but does not prove its cause.',
                             'Faster R-CNN has no epochwise validation record or earlier local checkpoint.',
                             'Loss definitions differ across models; compare time trends within each model only.',
                             'Best-checkpoint review is post hoc after earlier test access; no new test was used here.']}
    (OUT / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
