"""Build inspectable tables from completed common-checkpoint evaluation artifacts."""
import json
import pandas as pd
from xray_selected_common import RUNS, REPORT
from xray_config import ROOT

NAMES=['YOLOv8n','Faster R-CNN','RF-DETR-S','D-FINE-S']

def main():
    diag=REPORT/'diagnostics'
    benchmark=json.loads((diag/'benchmark_mps.json').read_text());assert benchmark['status']=='complete'
    conditions=pd.read_csv(diag/'conditions.csv')
    condition_summary=json.loads((diag/'condition_summary.json').read_text())
    audit=json.loads((diag/'error_audit.json').read_text())
    outcomes={x['run']:x for x in condition_summary['image_outcomes'] if x['split']=='test'}
    errors={x['run']:x for x in audit['results'] if x['split']=='test'}
    rows=[];lines=['## 18. 공통 검증 선택 저장본의 테스트·실패 조건·속도 — 2026-10-04',
        '', '네 모델을 전체 혼합 2,453장으로 각각 20epoch 학습한 뒤, 검증 107장에서 같은 규칙으로 저장본과 탐지 임계값을 선택했다. 아래는 그 값을 고정하고 테스트 97장에 적용한 결과다. 테스트 결과로 저장본·탐지 임계값을 바꾸지 않았다. 이전 실험에서 같은 테스트를 열람한 이력이 있으므로 완전히 새로운 비공개 시험으로 표현하지 않는다.',
        '', '테스트 추론은 네 모델 모두 맥북 MPS에서 새로 실행했다. 검증 표는 각 모델을 선택할 때 저장한 원래 검증 결과다. Faster·D-FINE의 원래 검증은 CUDA, YOLO·RF-DETR은 MPS였다. 같은 MPS에서 검증을 다시 확인한 결과와 환경 차이는 18.5절에 별도로 적는다.',
        '', '### 18.1. 새 선택 저장본 비교', '',
        '| 모델 | 선택 epoch | 고정 탐지 임계값 | 검증 F1 | 테스트 정밀도 | 테스트 재현율 | 테스트 F1 | TP / FP / FN |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for run,name in zip(RUNS,NAMES):
        r=json.loads((REPORT/f'{run}_evaluation.json').read_text());v=r['metrics']['iou50']['val'];t=r['metrics']['iou50']['test'];b=benchmark['models'][run]
        rows.append(dict(model=name,run=run,epoch=r['selected_epoch'],threshold=v['thr'],validation_f1=v['f1'],
            **{k:t[k] for k in ['precision','recall','f1','tp','fp','fn','ap']},
            ap_iou75=r['metrics']['iou75']['test']['ap'],f1_iou75=r['metrics']['iou75']['test']['f1'],
            f1_legacy=r['metrics']['legacy']['test']['f1'],
            no_alarm_images=outcomes[run]['positive_images_without_any_alarm'],
            mean_ms=b['mean_seconds']*1000,p95_ms=b['p95_seconds']*1000))
        lines.append(f"| {name} | {r['selected_epoch']} | {v['thr']:.8f} | {v['f1']:.4f} | {t['precision']:.4f} | {t['recall']:.4f} | {t['f1']:.4f} | {t['tp']} / {t['fp']} / {t['fn']} |")
    lines += ['', '기준은 **IoU ≥ 0.5·점수 순 일대일 대응**이다. 테스트 사진에는 정답 이물이 총 225개 있다. TP는 정답 박스에 맞게 찾은 수, FP는 대응하지 못한 채택 예측 수, FN은 찾지 못한 정답 박스 수다. 가까운 위치를 예측해도 IoU가 0.5 미만이면 해당 예측은 FP, 정답은 FN으로 각각 센다.',
        '', 'Faster R-CNN의 임계값이 1에 가깝다고 더 정확하거나 더 안전하다는 뜻은 아니다. 모델별 점수 분포가 다르고, 이 점수는 보정된 제조 이상 확률이 아니다. 임계값을 반올림해 1.0으로 사용하면 판정이 달라지므로 저장된 값을 그대로 사용한다.',
        '', '| 모델 | 테스트 AP@0.5 | 테스트 AP@0.75 | IoU 0.75 F1 | 기존 완화 기준 F1 |', '|---|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['model']} | {r['ap']:.4f} | {r['ap_iou75']:.4f} | {r['f1_iou75']:.4f} | {r['f1_legacy']:.4f} |")
    lines += ['', 'AP는 기존 채점기의 all-points 보간 AP이며 COCO mAP가 아니다. 보조 F1의 임계값도 해당 매칭 규칙의 검증에서 선택한 값이다. 표의 차이는 이 데이터·학습 설정·seed 0에서 관찰한 결과이며, 아키텍처의 일반적 우열을 입증하지 않는다.',
        '', '### 18.2. 어떤 방식으로 실패했나', '',
        '| 모델 | 근처 박스가 있었지만 IoU 미달인 FN | 위치는 맞지만 점수가 임계값 미달인 FN | 저장 예측에 IoU 0.5 이상 후보가 없는 FN |', '|---|---:|---:|---:|']
    cats=['nearby_selected_box_iou_below_0.5','matching_box_below_threshold','no_matching_box_in_saved_predictions']
    for run,name in zip(RUNS,NAMES):lines.append('| '+name+' | '+' | '.join(str(errors[run]['fn_categories'].get(c,0)) for c in cats)+' |')
    lines += ['', '‘근처’는 오류를 설명하기 위한 분류다. 중심 거리 8픽셀 이내 또는 IoU 0.3 이상을 사용하며, 정답으로 인정하는 IoU 0.5 기준은 바꾸지 않는다. 분류 순서는 근처 채택 예측 → 임계값 미달의 위치 일치 예측 → 저장 예측에 위치 일치 후보 없음이다. 이 분류는 원인 확정이 아니다. 저장 대상인 탐지 점수 0.001 미만 후보까지 확인한 것도 아니다.',
        '', '### 18.3. 이물 위치와 사진 단위 경보를 구분한다', '',
        '| 모델 | 경보가 전혀 없는 이물 사진 / 97장 | 이물을 하나 이상 정확히 찾은 사진 | 모든 이물을 정확히 찾은 사진 |', '|---|---:|---:|---:|']
    for run,name in zip(RUNS,NAMES):
        r=outcomes[run];lines.append(f"| {name} | {r['positive_images_without_any_alarm']} | {r['positive_images_with_any_correct_localization']} | {r['positive_images_with_all_boxes_found']} |")
    lines += ['', '경보는 임계값 이상 예측이 하나 이상 있는 상태다. 경보가 있어도 모든 이물 위치를 정확히 찾았다는 뜻은 아니다. 위 수치는 이물 사진에서의 관측이며 실제 출하·분리 장치를 시험한 결과가 아니다. 실제 정상 제품이 0장이므로 정상 오경보율, 재검사 업무량, 자동 PASS 안전성은 판단할 수 없다.',
        '', '### 18.4. 실패 조건을 어떻게 나누었나', '',
        '조건 경계는 공식 학습 사진 296장의 정답 박스에서 계산한 25·75백분위수로 정했다. 테스트 결과에 맞춰 경계를 바꾸지 않았다. 아래 각 칸은 **FN / 해당 조건의 정답 박스 수**다. 한 이물이 여러 조건에 동시에 포함되므로 행을 합산하지 않는다.',
        '', '| 조건 | YOLOv8n | Faster R-CNN | RF-DETR-S | D-FINE-S |', '|---|---:|---:|---:|---:|']
    targets=[('작은 박스: 긴 변 ≤ 10픽셀','size','low_le_train_q25'),
        ('제품 경계에 상대적으로 가까움: 거리 ≤ 45.84픽셀','edge','low_le_train_q25'),
        ('낮은 대비: 지표 ≤ 42.5','contrast','low_le_train_q25'),
        ('상대적으로 길쭉한 박스: 긴 변/짧은 변 > 1.2614','box_aspect_ratio','high_gt_train_q75'),
        ('어두운 배경: 주변 밝기 중앙값 ≤ 98','background_gray','low_le_train_q25'),
        ('밝기 변화가 큰 배경: 주변 표준편차 > 14.2349','background_std','high_gt_train_q75')]
    for label,feature,group in targets:
        cells=[]
        for run in RUNS:
            r=conditions[(conditions.run==run)&(conditions.split=='test')&(conditions.feature==feature)&(conditions.group==group)].iloc[0]
            cells.append(f'{int(r.fn)} / {int(r.n_gt)}')
        lines.append('| '+label+' | '+' | '.join(cells)+' |')
    lines += ['', '크기는 실제 이물 지름이 아니라 TXT 박스의 긴 변이다. 경계 거리는 영상에서 근사한 제품 마스크 내부 거리이며 실제 막대 경계와 다를 수 있다. 대비는 기존 채점기의 국소 밝기 차이 지표다. 종횡비는 이물의 분할 윤곽이 아닌 TXT 박스 모양이다. 배경 밝기와 표준편차는 주변 영역에서 모든 정답 박스를 제외하고 계산했다.',
        '', '따라서 ‘가장자리·저대비에서 항상 실패한다’고 결론 내릴 수 없다. 상대적으로 길쭉한 박스 집단에 일부 오류가 모였지만, 오류 수가 적고 촬영 묶음 안의 사진들이 비슷하다. 사진·촬영 묶음별 표본 수도 `conditions.csv`에 함께 보관한다. 정답 박스를 알아야 계산할 수 있는 조건은 미지의 이물을 자동으로 재검사시키는 실시간 규칙으로 바로 사용할 수 없다.',
        '', '### 18.5. 같은 맥북 GPU의 처리 시간', '',
        f"장치: {benchmark['processor']}, MPS. 모델별 검증 107장 × 3회, batch 1, FP32, 입력 크기 설정 512다. 모델을 하나씩 순서대로 실행하고 매 호출 뒤 GPU 완료를 기다렸다. 모델마다 3회 예열은 제외했다.", '',
        '| 모델 | 평균 ms/장 | 중앙값 ms/장 | P95 ms/장 | 평균 기준 장/초 |', '|---|---:|---:|---:|---:|']
    for run,name in zip(RUNS,NAMES):
        b=benchmark['models'][run];lines.append(f"| {name} | {b['mean_seconds']*1000:.2f} | {b['median_seconds']*1000:.2f} | {b['p95_seconds']*1000:.2f} | {b['serial_images_per_second']:.2f} |")
    lines += ['', '측정 구간은 PNG 읽기·RGB 변환·모델별 크기 변환·추론·후처리·CPU 배열 변환·고정 임계값 필터·GPU 동기화다. 색 네모 제거, 모델 불러오기, 촬영, 통신, 제품 분리 시간은 제외했다. P95는 측정 호출의 95%가 그 시간 이내였다는 뜻이며 최대 지연 보장이 아니다.',
        '', '동일한 512 설정이어도 모델별 늘리기·여백 추가 방식이 다르다. 같은 연산량 비교는 아니다. 한 맥북 세션의 결과이며 파일 캐시·온도·백그라운드 다운로드 및 압축 해제 부하를 완전히 통제하지 않았다. 공장 처리량이나 다른 GPU의 속도로 일반화하지 않는다. MPS fallback을 허용했으며 모델 파라미터의 MPS·FP32 배치를 확인했다. 모든 연산이 GPU에서만 실행됐음을 뜻하지는 않는다.',
        '', '첫 속도 측정 반복에서 검증 예측도 별도로 보관했다. 기존 검증 임계값 그대로 재채점해 장치 이동에 따른 판정 차이를 확인한다. 이 기록과 채점은 시간 측정 구간에서 제외했다.',
        '', '| 모델 | 원래 검증 장치 | 원래 TP / FP / FN | MPS 재확인 TP / FP / FN |', '|---|---|---:|---:|']
    for run,name in zip(RUNS,NAMES):
        b=benchmark['models'][run];r=json.loads((REPORT/f'{run}_evaluation.json').read_text())['metrics']['iou50']['val'];v=b['validation_on_benchmark_device']
        lines.append(f"| {name} | {b['training_validation_device']} | {r['tp']} / {r['fp']} / {r['fn']} | {v['tp']} / {v['fp']} / {v['fn']} |")
    lines += ['', '**D-FINE-S는 검증 판정 1개가 달라졌다.** 같은 이물의 점수가 CUDA 기록에서는 0.72014374, MPS에서는 0.7201233이었다. 고정 임계값 0.72014374를 사이에 두고 판정이 달라졌다. IoU는 두 환경 모두 약 0.6288로 위치 기준을 넘었다. 이 비교만으로 하드웨어가 유일한 원인이라고 단정하지 않는다. 환경 이동 시 경계 점수의 판정이 달라질 수 있음을 확인한 결과다. 임계값은 다시 맞추지 않았으며, 테스트 표는 MPS에서 관측한 성능이다.',
        '', '### 18.6. 실제 사진으로 이해하기', '',
        '아래 선들은 설명용으로 덧그린 것이다. 파랑은 공식 TXT의 정답 박스, 주황은 고정 탐지 임계값을 통과한 예측 박스, 보라 점선은 임계값 미달 예측이다. 원본의 색 네모와 구분한다. 왼쪽 위에는 사진 전체, 위쪽 가운데에는 선 없는 실제 입력 확대를 배치했다.',
        '', '**예시 A — 경보가 있어도 위치 채점에서 실패할 수 있다.** 가운데 이물의 정답 박스는 9×5픽셀이다. 네 모델 모두 근처를 예측했지만 IoU가 약 0.397~0.472로 0.5에 미달했다. 따라서 각 모델에서 FP 1개와 FN 1개를 센다. 이 사진에서 AI가 아무 경보도 내지 않았다는 뜻은 아니다.',
        '', '![네 모델의 작은 이물 박스 겹침 실패](figures/common20_localization_example.png)',
        '', '**예시 B — 위치가 맞아도 점수가 낮으면 경보하지 않는다.** YOLOv8n은 IoU 약 0.785인 후보를 만들었지만 탐지 점수 0.624495가 고정 임계값 0.68106407보다 낮았다. 이 사진에는 다른 채택 예측도 없어 경보가 없었다. 다른 세 모델은 같은 이물에 경보했다. 이 사례를 본 뒤 임계값을 낮추면 테스트를 이용한 조정이 되므로 이번 비교에서는 바꾸지 않았다.',
        '', '![점수가 임계값에 미달해 경보하지 않은 실제 사례](figures/common20_threshold_example.png)',
        '', '검증·테스트에서 오류가 있었던 서로 다른 사진 20장의 비교 그림과 원본 해시는 `reports/common_epoch_20261004/evaluation/diagnostics/figures/` 및 `figure_manifest.json`에 보관했다. 테스트 오류 사진은 11장이다. 모델별 오류 사진을 합칠 때 같은 사진은 한 번만 센 수다.',
        '', '### 18.7. 보관 위치와 다음 단계', '',
        'Colab 두 모델의 148개 파일, 8,085,339,976바이트를 맥북의 기존 프로젝트 `runs/v2_common20_faster_cuda_20261004/`, `runs/v2_common20_dfine_cuda_20261004/`에 보관했다. 선택 저장본뿐 아니라 20개 epoch별 가중치·검증 예측·선택 기록·재개 상태를 포함한다. 실행 로그·큐 기록은 `reports/common_epoch_20261004/colab_backup/`에 있다. Drive 원본은 유지한다.',
        '', '백업 목록의 파일 크기와 로컬 SHA-256을 기록했다. 40개 epoch의 가중치·검증 예측 해시와 4개 재개 저장 묶음의 파일 해시도 기존 기록과 대조했다. 네 모델의 선택 규칙·저장본 해시를 다시 확인했다. 전체 제출 패키지를 깨끗한 새 환경에서 재현한 검사는 아직 다음 단계다.',
        '', '이 단계에서는 최종 모델과 운영 재검사 임계값을 확정하지 않았다. 후속 결정에서는 검증 선택 근거, 위치 미탐, 사진 단위 무경보, 속도, 정상 제품 자료의 부재를 함께 검토한다.',
        '', '실행 근거: `reports/common_epoch_20261004/evaluation/frozen_selection.json`, 각 `*_evaluation.json`, `diagnostics/error_audit.json`, `conditions.csv`, `condition_summary.json`, `benchmark_mps.json`. 테스트 원본 예측은 `reports/preds_v2_common20_*_test.csv`다.']
    pd.DataFrame(rows).to_csv(REPORT/'comparison.csv',index=False)
    (REPORT/'document_section.md').write_text('\n'.join(lines)+'\n')
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':main()
