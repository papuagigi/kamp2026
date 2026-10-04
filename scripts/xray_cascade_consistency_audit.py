"""Evaluate a predeclared prediction-consistency referral, never using test."""
import json
import numpy as np
import pandas as pd

from xray_cascade_audit import OUT, NAMES, save, sha
from xray_eval import evaluate, load_gt, match, PRED_COLUMNS
from xray_selected_common import REPORT, RUNS


def main():
    directory=OUT/'consistency_native_csv'
    policy=json.loads((directory/'policy.json').read_text())
    record=json.loads((directory/'execution.json').read_text())
    assert record['status']=='complete' and record['policy_sha256']==sha(directory/'policy.json')
    for name,digest in record['csv_sha256'].items():assert sha(directory/name)==digest
    original=pd.read_csv(directory/'original_val.csv');flip=pd.read_csv(directory/'hflip_val.csv')
    first_kept=original[original.score>=policy['threshold']]
    flip_kept=flip[flip.score>=policy['threshold']]
    _,man=load_gt('val');routes=[]
    for stem in sorted(man.stem):
        a=first_kept[first_kept.stem==stem];b=flip_kept[flip_kept.stem==stem]
        matched,_=match(a,b,iou_thr=policy['box_correspondence_iou'],iou_only=True)
        pairs=int((matched.hit>=0).sum())
        routes.append(dict(stem=stem,original_boxes=len(a),flip_boxes=len(b),pairs=pairs,
            referred=(len(a)==0 or pairs!=len(a) or pairs!=len(b))))
    pd.DataFrame(routes).to_csv(directory/'routes.csv',index=False)
    selected={r['stem'] for r in routes if r['referred']}
    baseline,_,gt=evaluate(directory/'original_val.csv','val',thr=policy['threshold'],matching='iou50')
    previous=json.loads((OUT/'summary.json').read_text())['baseline']['YOLOv8n']
    stable={k:baseline[k]==previous[k] for k in ['tp','fp','fn','f1']}
    assert all(stable.values()), 'Original inference did not reproduce baseline; audit separately before comparison'
    frozen=json.loads((REPORT/'frozen_selection.json').read_text())
    source_policy=json.loads((OUT/'policy.json').read_text())
    mask=gt.stem.isin(selected).to_numpy();first=gt.detected.to_numpy(bool)
    results=[]
    for run in RUNS[1:]:
        path=REPORT/'diagnostics'/f'{run}_benchmark_validation.csv'
        from xray_config import ROOT
        assert sha(path)==source_policy['source_hashes'][str(path.relative_to(ROOT))]
        threshold=frozen['runs'][run]['thresholds']['iou50']
        raw=pd.read_csv(path); kept=raw[raw.score>=threshold]
        _,_,other=evaluate(path,'val',thr=threshold,matching='iou50')
        assert other[['stem','cx','cy','w','h']].equals(gt[['stem','cx','cy','w','h']])
        found=other.detected.to_numpy(bool)
        outpath=directory/f'{frozen["runs"][run]["model"]}_replacement_val.csv'
        output=pd.concat([first_kept[~first_kept.stem.isin(selected)],kept[kept.stem.isin(selected)]])
        output[PRED_COLUMNS].to_csv(outpath,index=False)
        result,_,newgt=evaluate(outpath,'val',thr=0,matching='iou50')
        assert np.array_equal(newgt.detected,np.where(mask,found,first))
        results.append(dict(second=NAMES[run],first_fn_recoverable=int((mask & ~first & found).sum()),
            first_tp_lost_if_replaced=int((mask & first & ~found).sum()),
            replacement={k:result[k] for k in ['tp','fp','fn','f1']},
            oracle_remaining_fn=int((~first & ~(mask & found)).sum())))
    summary=dict(policy_sha256=sha(directory/'policy.json'),validation_images=len(man),validation_gt=len(gt),
        baseline={k:baseline[k] for k in ['tp','fp','fn','f1']},new_original_regression=stable,
        referred_images=len(selected),first_fn_referred=int((mask & ~first).sum()),results=results,
        first_fn_route=[dict(stem=r.stem,referred=r.stem in selected) for r in gt[~first].itertuples()],
        limits=['Exploratory reused validation; not independent final performance.',
            'All FN use IoU50; physical containment and actual misses require separate review.',
            'Fresh first-model inference, cached second-model predictions; not a live two-model cascade.',
            'One-to-one prediction agreement does not prove correctness; shared errors can be stable.',
            'No real normal products; no production referral fraction or PASS safety estimate.'])
    save(directory/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
