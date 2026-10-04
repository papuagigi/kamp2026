"""Verify live independent secondary review and re-score its actual exports."""
import argparse
import json
from collections import Counter
import numpy as np
from xray_config import ROOT
from xray_eval import evaluate
from xray_recovery import atomic_json,sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('name');a=p.parse_args()
    out=ROOT/'reports/design_review_20261004'/a.name
    policy=json.loads((out/'policy.json').read_text());execution=json.loads((out/'execution.json').read_text())
    assert execution['status']=='complete' and execution['policy_sha256']==sha256(out/'policy.json')
    for f,h in execution['csv_sha256'].items():assert sha256(out/f)==h
    rows=[]
    for i in range(policy['repeats']):
        calls=[x for x in execution['calls'] if x['repeat']==i]
        assert len(calls)==107 and len({x['stem'] for x in calls})==107
        m,_,g1=evaluate(out/f'first_repeat{i}_val.csv','val',thr=policy['first']['threshold'],matching='iou50')
        first={k:m[k] for k in ['tp','fp','fn','f1','recall','precision']}
        second=None;pair=None
        if policy['scope']=='all':
            m,_,g2=evaluate(out/f'second_repeat{i}_val.csv','val',thr=policy['second']['threshold'],matching='iou50')
            second={k:m[k] for k in first}
            assert g1[['stem','cx','cy','w','h']].equals(g2[['stem','cx','cy','w','h']])
            da=g1.detected.to_numpy(bool);db=g2.detected.to_numpy(bool)
            pair=dict(first_fn_covered=int((~da&db).sum()),shared_fn=int((~da&~db).sum()),first_correct_second_wrong=int((da&~db).sum()))
        rows.append(dict(repeat=i,first=first,second=second,pair=pair,
            states=dict(Counter(x['state'] for x in calls)),
            conflicts=[x['stem'] for x in calls if x['state']=='REINSPECTION_DISAGREEMENT'],
            referred=sum(x['referred'] for x in calls),errors=sum(x['error'] is not None for x in calls)))
    outdata=dict(policy=policy,execution_sha256=sha256(out/'execution.json'),repeats=rows,
        mean_ms=execution['mean_ms'],p95_ms=execution['p95_ms'],latency_scope=execution['latency_scope'],
        operational_f1=None,shared_fn_is_iou_not_physical=True,
        agreement_does_not_prove_correctness=True,normal_product_false_alarm_rate=None)
    atomic_json(out/'summary.json',outdata)
    print(json.dumps(outdata,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
