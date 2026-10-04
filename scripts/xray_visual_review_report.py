"""Write a traceable report from visual judgments aggregated by xray_eval."""
import json
from pathlib import Path

import pandas as pd

from xray_eval import evaluate_visual_review
from xray_selected_common import REPORT, ROOT, RUNS

NAMES = dict(zip(RUNS, ['YOLOv8n','Faster R-CNN','RF-DETR-S','D-FINE-S']))


def main():
    folder=REPORT/'visual_review'
    result=evaluate_visual_review(folder)
    (folder/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    pd.DataFrame([{k:v for k,v in row.items() if not isinstance(v,dict)} for row in result['scores']]).to_csv(folder/'comparison.csv',index=False)
    rows={r['run']:r for r in result['scores'] if r['split']=='test'}
    lines=['| 모델 | AI 검토 TP | 중복·잘못된 표시 FP | 표시 누락 FN | AI 보조 F1 |',
           '|---|---:|---:|---:|---:|']
    for run in RUNS:
        r=rows[run];lines.append(f"| {NAMES[run]} | {r['tp']} | {r['fp']} | {r['fn']} | {r['f1']:.4f} |")
    table='\n'.join(lines)
    (folder/'table.md').write_text(table+'\n')
    print(table)
    print('Review scope: validation pilot 12 images/26 targets; test 97 images/225 targets; 899 accepted test boxes.')


if __name__=='__main__':main()
