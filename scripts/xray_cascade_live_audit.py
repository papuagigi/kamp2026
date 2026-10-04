"""Score live prototype exports with xray_eval and summarize measured telemetry."""
import json
from collections import Counter
import numpy as np
import pandas as pd
from xray_cascade_audit import save,sha
from xray_cascade_live import OUT
from xray_eval import PRED_COLUMNS,evaluate,match,load_gt
from xray_selected_common import REPORT


def main():
    p=json.loads((OUT/'policy.json').read_text());e=json.loads((OUT/'execution.json').read_text())
    assert e['status']=='complete' and e['policy_sha256']==sha(OUT/'policy.json')
    for f,h in e['csv_sha256'].items():assert sha(OUT/f)==h
    assert len(e['calls'])==321
    repeat=[];selected_stems=None
    for i in range(3):
        rows=[r for r in e['calls'] if r['repeat']==i]
        assert len(rows)==107 and len({r['stem'] for r in rows})==107
        routed={r['stem'] for r in rows if r['referred']}
        if selected_stems is None:selected_stems=routed
        else:assert routed==selected_stems
        first_path=OUT/f'original_repeat{i}_val.csv'
        baseline,_,gt=evaluate(first_path,'val',thr=p['thresholds'][p['first']],matching='iou50')
        first=pd.read_csv(first_path);second=pd.read_csv(OUT/f'second_repeat{i}_val.csv')
        assert set(second.stem)<=routed
        a=first[first.score>=p['thresholds'][p['first']]]
        b=second[second.score>=p['thresholds'][p['second']]]
        replacement=pd.concat([a[~a.stem.isin(routed)],b],ignore_index=True)
        dest=OUT/f'replacement_control_repeat{i}_val.csv';replacement[PRED_COLUMNS].to_csv(dest,index=False)
        control,_,_=evaluate(dest,'val',thr=0,matching='iou50')
        if i==0:
            baseline_gt=gt
            scored,_=match(a,gt,iou_thr=.5,iou_only=True)
            detail_stems=set(gt.loc[~gt.detected,'stem'])|set(scored.loc[scored.hit<0,'stem'])|routed
            save(OUT/'visual_scope.json',dict(stems=sorted(detail_stems),
                scope='All routed validation images and all baseline IoU50 FP/FN images. Targeted review, not all validation images.',
                missing_gt=gt.loc[~gt.detected,['stem','cx','cy','w','h']].to_dict(orient='records')))
        repeat.append(dict(repeat=i,referred=len(routed),states=dict(Counter(r['state'] for r in rows)),
            baseline={k:baseline[k] for k in ['tp','fp','fn','f1']},
            replacement_control={k:control[k] for k in ['tp','fp','fn','f1']}))
    def stats(rows):
        ms=np.array([r['seconds'] for r in rows])*1000
        return dict(n=len(ms),mean_ms=float(ms.mean()),median_ms=float(np.median(ms)),
                    p95_ms=float(np.quantile(ms,.95)),max_ms=float(ms.max()))
    calls=e['calls'];conflicts=[r for r in calls if r['state']=='REINSPECTION_DISAGREEMENT']
    memory=[e['resident_after_warmup']]+[r['memory'] for r in calls]
    summary=dict(policy_sha256=sha(OUT/'policy.json'),execution_sha256=sha(OUT/'execution.json'),
        device=e['device'],validation_images=107,validation_gt=len(baseline_gt),repeats=repeat,
        latency=dict(all=stats(calls),not_referred=stats([r for r in calls if not r['referred']]),
                     referred=stats([r for r in calls if r['referred']])),
        loading_seconds=e['loading_seconds'],resident_after_warmup=e['resident_after_warmup'],
        largest_observed_memory={k:max(m[k] for m in memory) for k in memory[0]},
        conflict_stems=sorted({r['stem'] for r in conflicts}),
        errors=sum(r['error'] is not None for r in calls),timeouts=sum(r['state']=='REINSPECTION_TIMEOUT' for r in calls),
        latency_scope=e['latency_scope'],memory_scope=e['memory_scope'],
        operational_f1=None,operational_f1_reason='Policy preserves two outputs and defers disagreement; replacement control is not operational adjudication.',
        limitations=['Exploratory reused validation, no new held-out test.',
            'All real validation products contain labeled targets; normal-product false alarms and referral fraction not estimable.',
            'Soft deadline checks after synchronous return; no hard interruption watchdog or factory deadline validation.',
            'Only full-frame second inference, no specialist fine-tuning or ROI/tiling.',
            'One Mac session and correlated capture groups. Not a production latency guarantee.'])
    save(OUT/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
