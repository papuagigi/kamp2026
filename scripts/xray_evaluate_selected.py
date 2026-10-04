"""Infer the fixed test split with a validation-selected checkpoint; never tune it."""
import argparse
import csv
import json
import os
import time
from xray_selected_common import RUNS, REPORT, verified, freeze, predictor_for
from xray_config import ROOT, DATA, V2_SPLIT_MD5
from xray_recovery import sha256, atomic_json, atomic_copy
from xray_eval import evaluate

def main():
    p=argparse.ArgumentParser(); p.add_argument('--run',required=True,choices=RUNS)
    p.add_argument('--device',default='mps'); a=p.parse_args()
    freeze()
    meta, selection, validation_csv, validation = verified(a.run)
    result=dict(run=a.run, execution=meta, selected_epoch=selection['best_epoch'],
        selected_checkpoint=selection['checkpoint'], checkpoint_sha256=selection['checkpoint_sha256'],
        metrics={}, normal_real_product_images=0, test_used_for_selection=False, previous_test_exposure=True)
    for split,n in [('val',107),('test',97)]:
        dest=ROOT/'reports'/f'preds_{a.run}_{split}.csv'
        provenance=dest.with_suffix('.run.json')
        paths=sorted((DATA/'images'/split).glob('*.png')); assert len(paths)==n
        if split=='val':
            if dest.exists(): assert sha256(dest)==sha256(validation_csv)
            else: atomic_copy(validation_csv,dest)
            atomic_json(provenance,dict(run=a.run,images=n,device=validation['device'],
                checkpoint_sha256=selection['checkpoint_sha256'], predictions_sha256=sha256(dest),
                source='saved selected-epoch validation; re-scored without new inference',
                image_stems=[x.stem for x in paths]))
        elif not dest.exists():
            from PIL import Image
            from xray_model_io import seed_all, sync
            import torch
            if a.device=='mps' and not torch.backends.mps.is_available(): raise RuntimeError('MPS unavailable')
            seed_all(); predictor=predictor_for(meta,selection,a.device); start=time.time()
            tmp=dest.with_suffix('.csv.pending')
            with tmp.open('w') as f:
                writer=csv.writer(f); writer.writerow(['stem','cx','cy','w','h','score'])
                for i,path in enumerate(paths):
                    with Image.open(path) as im: boxes=predictor(im,threshold=.001)
                    for x1,y1,x2,y2,score in boxes:
                        writer.writerow([path.stem,(x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1,score])
                    if (i+1)%25==0: print(a.run, i+1,n,flush=True)
            sync(a.device)
            atomic_json(provenance,dict(run=a.run,images=n,device=a.device,
                checkpoint_sha256=selection['checkpoint_sha256'],predictions_sha256=sha256(tmp),
                image_stems=[x.stem for x in paths],resolution=512,seconds=time.time()-start,
                source='new selected checkpoint test inference',export_threshold=.001))
            os.replace(tmp,dest)
        info=json.loads(provenance.read_text())
        assert info['checkpoint_sha256']==selection['checkpoint_sha256'] and info['images']==n
        assert info['predictions_sha256']==sha256(dest)
        assert info['image_stems']==[x.stem for x in paths]
        for mode in ['iou50','iou75','legacy']:
            threshold=selection['metrics'][mode]['thr']
            threshold=1.000001 if threshold is None else threshold
            metric,_,_=evaluate(dest,split,thr=threshold,matching=mode)
            assert metric['n_images']==n
            if split=='val':
                for key in ['tp','fp','fn','f1','recall','precision','ap']:
                    assert metric[key]==selection['metrics'][mode][key],(mode,key)
            metric['threshold_policy']='selected_epoch_validation_best_f1_frozen_for_test'
            result['metrics'].setdefault(mode,{})[split]=metric
        print(a.run,split,json.dumps(result['metrics']['iou50'][split]),flush=True)
    atomic_json(REPORT/f'{a.run}_evaluation.json',result)

if __name__=='__main__': main()
