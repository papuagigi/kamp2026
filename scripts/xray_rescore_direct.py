"""Re-score immutable direct-training outputs with the existing common evaluator."""
import json
import math
import pandas as pd
from xray_config import ROOT
from xray_recovery import sha256
from xray_eval import evaluate
from xray_epoch_validation import better

RUNS={'YOLO11s':'v2_direct20_yolo11s_mps_20261006','RF-DETR-S':'v2_direct20_rfdetr_cuda_20261006'}
OUT=ROOT/'reports/direct_analysis_20261007'


def main():
    OUT.mkdir(parents=True,exist_ok=True);checks={};summary=[]
    for name,run in RUNS.items():
        r=ROOT/'runs'/run;e=json.loads((r/'evaluation.json').read_text());s=json.loads((r/'selection.json').read_text())
        frozen=json.loads((r/'evaluation_frozen.json').read_text())
        assert sha256(ROOT/s['checkpoint'])==s['checkpoint_sha256']==e['checkpoint_sha256']==frozen['checkpoint_sha256']
        records=[json.loads((r/f'validation/epoch_{i:03d}.json').read_text()) for i in range(1,21)]
        best=None
        for rec in records:
            assert sha256(r/f"validation/epoch_{rec['epoch']:03d}.csv")==rec['predictions_sha256']
            if best is None or better(rec,best):best=rec
        assert best['epoch']==s['best_epoch']==e['selected_epoch']==frozen['epoch']
        result={}
        for split in ['val','test']:
            p=r/f'predictions_{split}.csv';proof=json.loads(p.with_suffix('.json').read_text())
            assert sha256(p)==proof['predictions_sha256']
            for mode in ['iou50','iou75','legacy']:
                threshold=frozen['thresholds'][mode]
                m,curve,g=evaluate(p,split,thr=threshold,matching=mode)
                old=e['metrics'][mode][split]
                for k in ['tp','fp','fn','f1','precision','recall','ap','thr','n_images','n_gt']:
                    assert m[k]==old[k] or (m[k] is not None and old[k] is not None and math.isclose(m[k],old[k],abs_tol=1e-12)),(name,split,mode,k)
                assert int(g.detected.sum())==m['tp'] and len(g)-int(g.detected.sum())==m['fn']
                result.setdefault(mode,{})[split]=m
                g.to_csv(OUT/f'{name}_{split}_{mode}_gt.csv',index=False)
                summary.append(dict(model=name,split=split,matching=mode,selected_epoch=s['best_epoch'],**{k:m[k] for k in ['thr','tp','fp','fn','precision','recall','f1','ap','n_images','n_gt']}))
                print(name,split,mode,'verified',m['f1'],flush=True)
        checks[name]=dict(status='passed',selected_epoch=s['best_epoch'],all_20_validation_hashes_match=True,
            selection_reproduced=True,checkpoint_sha256=s['checkpoint_sha256'],predictions_and_metrics=result)
    pd.DataFrame(summary).to_csv(OUT/'rescore_summary.csv',index=False)
    (OUT/'rescore_audit.json').write_text(json.dumps(dict(models=checks,new_training=False,new_inference=False,thresholds_retuned=False),ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
