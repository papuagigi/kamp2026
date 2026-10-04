"""Audit stable wrong predictions and accuracy-first secondary candidates.

Validation replay only. Ground truth diagnoses blind spots; it never routes images.
No union/oracle coverage is presented as an operational system F1.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from xray_config import ROOT
from xray_eval import evaluate, match
from xray_cascade_runtime import agreement
from xray_recovery import atomic_json, sha256
from xray_selected_common import REPORT, RUNS

OUT=ROOT/'reports/design_review_20261004/shared_error_audit'
NAMES=['YOLOv8n','Faster R-CNN','RF-DETR-S','D-FINE-S']


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source=REPORT/'cascade_audit/consistency_native_csv'
    frozen=json.loads((REPORT/'frozen_selection.json').read_text())
    sources={name:REPORT/'diagnostics'/f'{run}_benchmark_validation.csv'
             for name,run in zip(NAMES,RUNS)}
    sources['YOLOv8n']=source/'original_val.csv'
    sources['YOLOv8n flipped']=source/'hflip_val.csv'
    thresholds={name:frozen['runs'][run]['thresholds']['iou50'] for name,run in zip(NAMES,RUNS)}
    thresholds['YOLOv8n flipped']=thresholds['YOLOv8n']
    routes=pd.read_csv(source/'routes.csv')
    assert routes.stem.nunique()==107 and routes.referred.dtype==bool
    policy=dict(scope='reused validation diagnostic, no new inference or test',
                selection_priority='actual omission, wrong or duplicate indication, disagreement; latency secondary',
                matching='existing xray_eval iou50', no_threshold_change=True,
                prior_test_exposure=True, normal_products_available=False,
                hashes={str(p.relative_to(ROOT)):sha256(p) for p in [
                    *sources.values(),source/'routes.csv',REPORT/'frozen_selection.json']},
                thresholds=thresholds)
    policy_path=OUT/'policy.json'
    if policy_path.exists():assert json.loads(policy_path.read_text())==policy
    else:atomic_json(policy_path,policy)
    kept={};metrics={};found={};curves={};fps={};gt=None
    for name,path in sources.items():
        m,curve,g=evaluate(path,'val',thr=thresholds[name],matching='iou50')
        if gt is not None:
            assert g[['stem','cx','cy','w','h']].equals(gt[['stem','cx','cy','w','h']])
        else:gt=g.copy()
        raw=pd.read_csv(path)
        kept[name]=raw[raw.score>=thresholds[name]].copy()
        tagged,_=match(kept[name],g,iou_thr=.5,iou_only=True)
        fps[name]=tagged[tagged.hit<0].drop(columns=['hit'])
        metrics[name]={k:m[k] for k in ['tp','fp','fn','f1','precision','recall','thr','n_images','n_gt']}
        found[name]=g.detected.to_numpy(bool)
        curves[name]=curve
    stable=set(routes.loc[~routes.referred,'stem'])
    mask=gt.stem.isin(stable).to_numpy()
    both_wrong=~found['YOLOv8n'] & ~found['YOLOv8n flipped']
    cases=[]
    for i,row in gt.loc[both_wrong].iterrows():
        cases.append(dict(gt_index=int(i),stem=row.stem,cx=float(row.cx),cy=float(row.cy),
                          w=float(row.w),h=float(row.h),skipped_by_flip_gate=row.stem in stable,
                          detected_by={name:bool(found[name][i]) for name in NAMES[1:]}))
    fp_pairs,_=match(fps['YOLOv8n'],fps['YOLOv8n flipped'],iou_thr=.5,iou_only=True)
    paired_fp=fp_pairs[fp_pairs.hit>=0]
    stable_fp=paired_fp[paired_fp.stem.isin(stable)]
    fp_detail=stable_fp[['stem','cx','cy','w','h','score']].to_dict(orient='records')
    comparisons=[]
    for name in NAMES[1:]:
        conflicts=[]
        for stem in sorted(routes.stem):
            a=kept['YOLOv8n'][kept['YOLOv8n'].stem==stem]
            b=kept[name][kept[name].stem==stem]
            if not agreement(a,b)['agree']:conflicts.append(stem)
        curve=curves[name];operating=[]
        for target in [.99,1.]:
            feasible=curve[curve.recall>=target]
            row=feasible.iloc[0] if len(feasible) else None
            operating.append(dict(target_recall=target,threshold=None if row is None else float(row.thr),
                fp=None if row is None else int(row.fp),
                recall=None if row is None else float(row.recall),
                role='diagnostic validation frontier, not selected deployment threshold'))
        comparisons.append(dict(model=name,metrics=metrics[name],
            stable_first_fn_covered=int((mask & ~found['YOLOv8n'] & found[name]).sum()),
            first_fn_covered=int((~found['YOLOv8n'] & found[name]).sum()),
            shared_fn=int((~found['YOLOv8n'] & ~found[name]).sum()),
            first_correct_second_wrong=int((found['YOLOv8n'] & ~found[name]).sum()),
            all_review_conflict_images=len(conflicts),
            conflict_images_missed_by_flip_gate=len(set(conflicts)&stable),
            conflict_stems=conflicts,recall_frontier=operating))
    result=dict(policy_sha256=sha256(policy_path),images=107,official_targets=len(gt),
        stable_images=len(stable),referred_images=int(routes.referred.sum()),
        baseline=metrics['YOLOv8n'],flipped=metrics['YOLOv8n flipped'],
        fn_in_both_views=int(both_wrong.sum()),stable_fn_in_both_views=int((both_wrong&mask).sum()),
        corresponding_fp_in_both_views=len(paired_fp),stable_corresponding_fp=len(stable_fp),
        common_fn_cases=cases,stable_common_fp=fp_detail,secondary=comparisons,
        recommendation='Compare independent full-frame secondary inspection, including Faster R-CNN, on accuracy first. Do not use agreement as correctness or auto-PASS.',
        limits=['All FP/FN here are official IoU50 localization metrics, not independently verified physical absence.',
                'Shared model failures remain possible even with different architectures.',
                'Full secondary inspection changes resource use and requires measured latency and queue design.',
                'No normal products, so product false alarm, production review rate, PASS safety unavailable.',
                'Validation reused and test previously seen; prospective independent validation still needed.',
                'Performance alone cannot certify AI visual reviewer reliability.'])
    atomic_json(OUT/'summary.json',result)
    pd.DataFrame(cases).to_csv(OUT/'common_fn.csv',index=False)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
