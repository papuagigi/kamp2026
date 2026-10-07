"""Evaluate a completed direct-label run using its frozen validation selection."""
import argparse
import csv
import json
import os
import time
from PIL import Image
from xray_config import ROOT, DATA, V2_SPLIT_MD5
from xray_recovery import sha256, atomic_json, atomic_copy, run_lock
from xray_eval import evaluate
from xray_predict_controlled import Predictor
from xray_model_io import seed_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--device', required=True)
    args = parser.parse_args()
    run = ROOT / 'runs' / args.run
    with run_lock(run / 'evaluation'):
        meta = json.loads((run / 'execution.json').read_text())
        selection = json.loads((run / 'selection.json').read_text())
        assert meta['status'] == 'complete' and meta['completed_epochs'] == meta['epochs']
        assert selection['completed_epochs'] == meta['epochs']
        checkpoint = ROOT / selection['checkpoint']
        assert sha256(checkpoint) == selection['checkpoint_sha256']
        epoch = selection['best_epoch']
        validation = json.loads((run / 'validation' / f'epoch_{epoch:03d}.json').read_text())
        assert validation['checkpoint_sha256'] == selection['checkpoint_sha256']
        val_csv = run / 'validation' / f'epoch_{epoch:03d}.csv'
        assert sha256(val_csv) == validation['predictions_sha256']
        result = dict(run=args.run, selected_epoch=epoch,
                      checkpoint_sha256=selection['checkpoint_sha256'], metrics={},
                      split_md5=V2_SPLIT_MD5, previous_test_exposure=True,
                      test_used_for_selection=False, real_normal_test_images=0)
        # Persist selection before accessing test pixels or predictions.
        frozen = dict(checkpoint_sha256=selection['checkpoint_sha256'], epoch=epoch,
                      thresholds={m: selection['metrics'][m]['thr'] for m in ['iou50','iou75','legacy']})
        frozen_path = run / 'evaluation_frozen.json'
        if frozen_path.exists():
            assert json.loads(frozen_path.read_text()) == frozen
        else:
            atomic_json(frozen_path, frozen)
        for split, count in [('val',107), ('test',97)]:
            dest = run / f'predictions_{split}.csv'
            proof = dest.with_suffix('.json')
            paths = sorted((DATA / 'images' / split).glob('*.png'))
            assert len(paths) == count
            if not dest.exists():
                start = time.time()
                if split == 'val':
                    atomic_copy(val_csv, dest)
                else:
                    seed_all()
                    predictor = Predictor(meta['model'], checkpoint, args.device, meta['resolution'])
                    tmp = dest.with_suffix('.pending')
                    with tmp.open('w') as f:
                        writer = csv.writer(f)
                        writer.writerow(['stem','cx','cy','w','h','score'])
                        for path in paths:
                            with Image.open(path) as im:
                                boxes = predictor(im)
                            for x1,y1,x2,y2,score in boxes:
                                writer.writerow([path.stem,(x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1,score])
                    os.replace(tmp, dest)
                atomic_json(proof, dict(checkpoint_sha256=frozen['checkpoint_sha256'],
                    predictions_sha256=sha256(dest), images=count,
                    device=validation['device'] if split=='val' else args.device,
                    source='selected epoch saved validation' if split=='val' else 'new test inference',
                    seconds=time.time()-start))
            info = json.loads(proof.read_text())
            assert info['checkpoint_sha256'] == frozen['checkpoint_sha256']
            assert info['predictions_sha256'] == sha256(dest) and info['images'] == count
            for mode in ['iou50','iou75','legacy']:
                threshold = frozen['thresholds'][mode]
                metric, _, _ = evaluate(dest, split, thr=1.000001 if threshold is None else threshold, matching=mode)
                assert metric['n_images'] == count
                if split == 'val':
                    for key in ['tp','fp','fn','f1','precision','recall','ap']:
                        assert metric[key] == selection['metrics'][mode][key]
                result['metrics'].setdefault(mode, {})[split] = metric
        atomic_json(run / 'evaluation.json', result)
        print(json.dumps(dict(run=args.run,selected_epoch=epoch,
            test=result['metrics']['iou50']['test']),ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
