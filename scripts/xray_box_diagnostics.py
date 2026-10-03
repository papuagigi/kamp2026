"""Describe localization of matches from the common scorer; do not create a new AP metric."""
import argparse
import json
import numpy as np
import pandas as pd
from xray_config import ROOT
from xray_eval import evaluate,load_gt,match,iou


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
    csv=ROOT/'reports'/f'preds_{a.run}_val.csv';res=json.loads((csv.with_name(csv.stem+'_custom_ap_v2_eval.json')).read_text())
    gt,_=load_gt('val');pred=pd.read_csv(csv);matched,_=match(pred,gt);rows=[]
    for r in matched[(matched.score>=res['thr']) & (matched.hit>=0)].itertuples():
        g=gt.iloc[r.hit];rows.append(dict(stem=r.stem,score=r.score,center_error_px=float(np.hypot(r.cx-g.cx,r.cy-g.cy)),
            iou=float(iou((r.cx,r.cy,r.w,r.h),(g.cx,g.cy,g.w,g.h))),gt_index=r.hit))
    table=pd.DataFrame(rows);out=ROOT/'reports/roadmap_20261002/localization';out.mkdir(exist_ok=True,parents=True)
    table.to_csv(out/f'{a.run}.csv',index=False)
    summary={'run':a.run,'split':'val','threshold':res['thr'],'matched_boxes':len(table),'fn':res['fn'],'fp':res['fp'],
        'mean_center_error_px':float(table.center_error_px.mean()),'p95_center_error_px':float(table.center_error_px.quantile(.95)),
        'median_iou':float(table.iou.median()),'mean_iou':float(table.iou.mean()),
        'fraction_matched_iou_ge_05':float((table.iou>=.5).mean()),
        'scope':'descriptive statistics of true positives under custom_ap_v2; not COCO AP or a second F1 score'}
    (out/f'{a.run}.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
