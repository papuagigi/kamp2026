"""Create review-oriented product outputs from saved predictions; never certify normality."""
import json
from pathlib import Path
import pandas as pd
from xray_config import ROOT,DATA
from xray_eval import evaluate,match,load_gt

REPORT=ROOT/'reports/roadmap_20261002'
OUT=ROOT/'outputs/roadmap_20261002'


def main():
    decision=json.loads((REPORT/'model_selection.json').read_text());run=decision['recommended_run'];thr=decision['threshold']
    OUT.mkdir(parents=True,exist_ok=True)
    pred=ROOT/'reports'/f'preds_{run}_test.csv'
    rows=pd.read_csv(pred);inference=json.loads(pred.with_suffix('.run.json').read_text())
    man=pd.read_csv(DATA/'manifest.csv');man=man[man.split=='test']
    assert set(man.stem)==set(inference['image_stems'])
    rows[rows.score>=thr].to_csv(OUT/'test_detected_boxes.csv',index=False)
    products=[]
    for r in man.itertuples():
        pr=rows[rows.stem==r.stem];selected=pr[pr.score>=thr]
        state='이물의심_분리후재검사' if len(selected) else ('낮은점수_재검사' if len(pr) else '무검출_보류')
        products.append({'stem':r.stem,'anomaly_suspicion_score':float(pr.score.max()) if len(pr) else 0.,
                         'threshold':thr,'detected_boxes':len(selected),'decision':state,'probability_calibrated':False})
    pd.DataFrame(products).to_csv(OUT/'test_product_decisions.csv',index=False,encoding='utf-8-sig')
    scores,curve,gt=evaluate(pred,'test',thr=thr,matching='iou50')
    gt.to_csv(OUT/'test_gt_detection_details.csv',index=False,encoding='utf-8-sig')
    matched,_=match(rows,gt,iou_thr=.5,iou_only=True);matched[matched.score>=thr].to_csv(OUT/'test_prediction_matching.csv',index=False)
    conditions=[]
    for group,values in scores['recall_by_condition'].items():
        for condition,(recall,n) in values.items():conditions.append({'condition_group':group,'condition':condition,'recall':recall,'n_gt':n})
    pd.DataFrame(conditions).to_csv(OUT/'test_condition_recall.csv',index=False,encoding='utf-8-sig')
    # A separate operating-candidate threshold comes ONLY from validation; no test tuning.
    final=json.loads((REPORT/'final_comparison.json').read_text());validation=next(r['validation'] for r in final if r['run']==run)
    settings={'run':run,'comparison_threshold':thr,'comparison_source':'validation maximum F1',
        'recall99_threshold_candidate':validation['thr_recall99'],'recall99_validation_fp':validation['fp_at_recall99'],
        'automatic_pass_enabled':False,'reason':'Representative real normal-product data and line-latency validation unavailable',
        'product_score':'maximum emitted detection score, not calibrated defect probability',
        'no_detection_policy':'hold for review; empty detections are not evidence of normality',
        'test_images':len(products),'test_boxes_at_comparison_threshold':len(rows[rows.score>=thr]),
        'model_checkpoint_sha256':inference['checkpoint_sha256'],'metric_version':'iou50_v1'}
    (OUT/'operating_candidate.json').write_text(json.dumps(settings,ensure_ascii=False,indent=2))
    (OUT/'README.md').write_text('''# 예측 결과 읽기

- `test_detected_boxes.csv`: 검증 F1로 고른 임계값 이상 박스. 중심·크기는 픽셀 단위입니다.
- `test_product_decisions.csv`: 사진별 최고 점수와 재검사·보류 제안입니다. 점수는 보정된 불량 확률이 아닙니다.
- `test_gt_detection_details.csv`: 공식 이물별 탐지 여부와 크기·대비·제품 가장자리 거리입니다.
- `test_prediction_matching.csv`: 예측과 정답의 대응 기록입니다. hit=-1은 공통 채점 기준에서의 오탐입니다.
- `test_condition_recall.csv`: 조건별 재현율과 정답 개수입니다. 관찰한 연관성으로 해석합니다.
- `operating_candidate.json`: 비교 임계값, 검증 재현율 99% 후보, 체크포인트 확인값을 기록합니다.

공식 테스트 사진은 양성 시험편 중심입니다. 정상 생산품의 오경보율, 보정된 이상 확률, 자동 통과의 안전성은 검증되지 않았습니다. 이 결과는 검토용이며 공장 설비를 제어하지 않습니다.
''')
    print(json.dumps(settings,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
