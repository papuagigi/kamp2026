"""Read saved research evidence; do not train, infer, or rescore detections."""
import hashlib
import itertools
import json
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / 'reports/model_research_20261003'
    out.mkdir(parents=True, exist_ok=True)
    notebooks = []
    for p in sorted((ROOT / '제조AI데이터셋').rglob('*.ipynb')):
        cells = json.loads(p.read_text())['cells']
        commands = [dict(cell_one_based=i + 1, source=''.join(c.get('source', [])))
                    for i, c in enumerate(cells)
                    if c.get('cell_type') == 'code'
                    and any(s in ''.join(c.get('source', []))
                            for s in ['!python train.py', '!python detect.py', '!python test.py'])]
        notebooks.append(dict(path=str(p.relative_to(ROOT)),
                              sha256=hashlib.sha256(p.read_bytes()).hexdigest(), commands=commands))
    run = ROOT / 'runs/v2_mix2453_rfdetr_mps'
    metrics = pd.read_csv(run / 'metrics.csv').dropna(subset=['val/mAP_50_95'])
    best = metrics.loc[metrics['val/mAP_50_95'].idxmax()]
    final = metrics.iloc[-1]
    def row(r):
        return dict(epoch_one_based=int(r['epoch']) + 1,
                    native_map_50_95=float(r['val/mAP_50_95']),
                    native_map_50=float(r['val/mAP_50']), native_map_75=float(r['val/mAP_75']))
    checkpoints = []
    for name in ['checkpoint_best_regular.pth', 'last.ckpt']:
        p = run / name
        checkpoint = torch.load(p, map_location='cpu', weights_only=True)
        checkpoints.append(dict(path=str(p.relative_to(ROOT)),
                                epoch_one_based=int(checkpoint['epoch']) + 1))
        del checkpoint
    details_path = ROOT / 'reports/overnight_20261003/diagnostics/ground_truth_detail.csv'
    details = pd.read_csv(details_path)
    val = details[details.split == 'val']
    misses = {k: set(zip(x.loc[~x.detected, 'stem'], x.loc[~x.detected, 'gt_index']))
              for k, x in val.groupby('run')}
    overlaps = []
    for n in [2, 3]:
        for names in itertools.combinations(misses, n):
            overlap = set.intersection(*(misses[name] for name in names))
            overlaps.append(dict(models=list(names), shared_fn=len(overlap)))
    result = dict(original_notebooks=notebooks, rfdetr_native_best=row(best),
                  rfdetr_native_last=row(final), checkpoints=checkpoints,
                  validation_missed_gt={k: len(v) for k, v in misses.items()},
                  validation_shared_misses=overlaps,
                  limitations=['Native RF-DETR mAP is not project F1 or all-points AP.',
                               'Overlap is a ground-truth-assisted diagnostic, not measured ensemble performance.',
                               'No training, inference, threshold search, or scoring was performed.'])
    (out / 'research_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
