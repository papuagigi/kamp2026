# 설명 그림 목록

[문서 처음으로](../README.md)

그림은 전처리·그림 생성 스크립트가 사용하는 파일 이름을 유지한다. 아래에서 현재 정리본에 쓰는 그림과 과거 실험 그림을 구분했다. 과거 그림의 점수·설정은 당시 조건이며 최신 결과로 그대로 인용하지 않는다.

## 현재 정리본의 그림

| 파일 | 읽는 방법 |
|---|---|
| [19 색 네모와 정답](19_색표시와_정답의_차이.png) | 빨간 장비 선·검은 이물·TXT를 그린 파란 박스를 구분 |
| [19 TXT 라벨과 사진 좌표](19_TXT라벨_사진과_좌표.png) | 실제 사진의 이물 3개와 공식 TXT 3줄 대응, 종류·중심·크기 해석 |
| [19 사진을 자르는 과정](19_사진을_자르는_과정.png) | 노란 범위 밖을 잘라내고 남은 사진을 확대한 결과 |
| [19 실제 생성한 전처리](19_실행한_전처리_사진예시.png) | 동일 학습 사진의 자르기·회전·이물 일부/전부 제거·다른 위치 합성 결과와 저장 라벨 |
| [19 사진 전체 회전](19_사진전체_회전.png) | 제품·막대·이물을 함께 돌리는 실제 설명용 변환 |
| [19 AI 라벨 검수 예시](19_AI라벨_검수예시.png) | 맞는 박스·누락·과한 크기. 실제 추론이 아닌 가상 검수 예시 |
| [19 촬영 묶음 분할](19_촬영묶음으로_나누기.png) | 비슷한 연속 사진을 같은 학습·검증·테스트 묶음에 두는 개념 도식 |
| [20 실제 예측과 정답](20_실제모델_예측과_정답.png) | 공식 검증 사진에서 저장된 실제 예측과 TXT를 겹친 비교, 같은 이물 확대 |
| [20 모델 세 가지](20_모델세가지_쉬운도식.png) | 세 구조의 탐지 과정과 비교 목적을 쉬운 말로 설명 |
| [16 동료 데이터 종류](16_types_actual_examples.png) | 전달받은 A~F의 실제 사진. 기존 그림을 재사용 |

## 과거 원본 진단과 전처리 그림

| 파일 | 당시 용도 |
|---|---|
| [04 원본 예시](04_xray_samples.png) | 초기 원본 관찰 |
| [04 표시 없는 예시](04_xray_no_marker_samples.png) | 초기 표시 유형 관찰. 현재의 노란 선/정상 구분과 함께 해석 |
| [04 마스킹 예시](04_xray_masking_demo.png) | 초기 표시 제거 예시 |
| [04 검사기 미표시](04_xray_machine_missed.png) | 초기 색 네모 대응 분석 |
| [06 사진 유형](06_image_types.png) | 원본 구성 진단 |
| [06 사진 읽기](06_raw_walkthrough.png) | 원본 속 요소 설명 |
| [06 이물 통계](06_object_stats.png) | 공식 정답의 크기·대비 진단 |
| [06 촬영 시간](06_time_pattern.png) | 시간·검사 루틴 진단 |
| [06 마스킹 비교](06_mask_methods.png) | 표시 제거 방법 비교 |
| [08 v1 v2 비교](08_v1_v2_compare.png) | 전처리 방식의 변화 |

## 과거 실험과 회의 그림

| 파일 | 당시 용도 |
|---|---|
| [06 조건별 재현율](06_recall_by_condition.png) | 과거 기준선 결과. 데이터·채점 버전 확인 필요 |
| [07 가이드북 표시 의존](07_guidebook_mark_dependence.png) | 가이드북 모델 진단 |
| [08 v2 조건별 재현율](08_v2_recall_by_condition.png) | 기존 v2 실험 결과 |
| [13 원본](13_raw_example.png) | 회의 자료의 원본 사진 |
| [13 v2 과정](13_v2_process.png) | 이전 전처리 설명 도식 |
| [13 묶음 분할](13_group_split.png) | 이전 분할 설명 도식 |
| [13 모델 구조](13_model_architectures.png) | 전문 용어를 포함한 이전 모델 도식 |

## 동료 자료 검사와 증강 조사 그림

| 파일 | 당시 용도 |
|---|---|
| [15 표시 제거 비교](15_marks_before_after.png) | 동료 자료의 원본·처리 후 비교 |
| [15 증강 예시](15_augmentation_examples.png) | 동료 자료의 증강 종류 |
| [15 라벨 누락](15_missing_validation_label.png) | 공식 TXT 누락 의심 사진의 확인 |
| [16 픽셀 확대](16_pixel_changes_zoom.png) | 합성·제거로 실제 바뀐 영역 |
| [17 사진 변형 예시](17_whole_image_augmentation_examples.png) | 이전 Crop·회전 시제품 설명 |
| [17 장비와 TXT](17_equipment_vs_txt_example.png) | 색 네모 크기와 공식 정답 박스 차이 |

문서 통합 때 만든 그림 6개와 이후 TXT 설명·실제 전처리·모델 예측 그림 3개, 기존 그림 24개를 합쳐 33개다. 기존 24개는 내용·경로를 변경하지 않았다. 이전 Markdown 문서들의 그림 참조는 원문 모음과 ZIP에서 추적할 수 있다.

- [색 네모·정답 박스·예측 박스](19_색네모_정답박스_예측박스.png): 동일한 실제 사진에서 원본 표시·TXT·YOLO 예측을 구분한다.
- [IoU와 중복 예측](19_IoU와_중복예측_설명.png): 겹친 면적 비율과 정답 하나를 중복 정답으로 세지 않는 규칙을 설명하는 개념도다.

## 2026-10-03 모델 구조 설명 도식

아래 5개는 원 논문과 공식 구현을 참고해 만든 한국어 설명 도식이다. 실제 모델의 특징 지도나 예측 결과가 아니다. 각 그림은 PNG와 같은 이름의 SVG로 보관한다. 그림 안의 S번호는 [모델 정리본 2.7절](../20_모델선정과_평가_정리본.md)의 출처 목록을 가리킨다.

| 그림 | 설명 |
|---|---|
| [세 모델의 처리 순서](20_구조비교_출처기반.png) | 한 단계 CNN 탐지, 두 단계 CNN 탐지, Transformer 탐지 흐름 |
| [YOLOv8n](20_YOLOv8n_구조설명.png) | Backbone, Neck, 탐지 Head, NMS |
| [Faster R-CNN](20_FasterRCNN_구조설명.png) | 공유 특징, RPN, RoIAlign, 영역별 판단 |
| [ResNet50과 FPN](20_ResNet50_FPN_역할.png) | 잔차 연결과 여러 해상도의 특징 결합 |
| [RF-DETR-S](20_RFDETRS_구조설명.png) | 패치, DINOv2 ViT, projector, query와 decoder |

생성: `scripts/xray_model_architecture_figures.py`. 기록: `reports/model_explanation_20261003/figure_manifest.json`. 기본 학습 데이터셋의 입력 사진은 읽기만 했다. 모델 정리본에 연결한 FPN·RF-DETR의 원 논문 그림은 외부 이미지이며, 위 재구성 그림과 구분한다.


## 2026-10-03 실제 오류 사진

- [박스 크기 불일치](20_오류사례_박스크기.png): 같은 검증 이물 주변에 세 모델이 낸 작은 박스와 공식 TXT 박스를 비교한다.
- [탐지 임계값 미달](20_오류사례_임계값미달.png): 테스트 사진에서 YOLO의 낮은 점수 후보가 제외된 예시다.
- [TXT에 없는 예측](20_오류사례_TXT없는예측.png): YOLO가 라벨 없는 어두운 무늬를 표시한 예시다. 그 무늬의 실제 이물 여부는 미확정이다.

파랑은 공식 TXT, 주황은 채택 예측, 보라 점선은 임계값 미달 후보다. 실제 입력을 확대하고 설명용 선만 덧그렸다. 가상 예측이나 학습 증강 사진이 아니다. 생성: `scripts/xray_error_figures.py`. 전체 15장과 원본 해시: `reports/overnight_20261003/diagnostics/{figure_manifest,visual_review}.json`.

- `19_전달자료와_현재자료_동일원본비교.png`: 같은 원본의 전달 합성본과 현재 전체 자르기·회전본. 스크립트 `scripts/xray_preprocessing_comparison_figure.py`, 출처 `reports/preprocessing_comparison_20261003/figure_sources.json`. PNG에는 없는 TXT 설명선을 덧그린 도식이다.

## 2026-10-04 학습 기록 재검토

- [학습 곡선과 저장 모델](20_학습곡선과_저장모델_재검토.png): 전체 혼합 2,453장·20epoch의 실제 저장 로그를 그렸다. YOLO·RF-DETR의 자체 검증 mAP 최고 시점과 마지막 시점을 비교한다. Faster R-CNN은 epoch별 검증 기록이 없어 학습 손실만 표시한다. 모델 사이의 손실 정의가 다르며 자체 mAP는 공통 F1과 다르다. 과적합 확정이나 새 추론 결과가 아니다.
- 생성: `scripts/xray_training_history_figure.py`. 기록 점검: `scripts/xray_training_history_audit.py`. 공유 근거: `docs/evidence/training_history_20261004/`.


## 2026-10-04 EDA와 실제 위치 오류 보완

- `19_EDA_라벨과_박스크기.png`: 원본·라벨 수와 정답 박스 크기 분포.
- `20_학습곡선과_저장모델_재검토.png`: 과거 학습 이력의 확인 범위. 새 공통 네 모델 결과와 구분한다.
- `common20_localization_example.png`, `common20_threshold_example.png`, `common20_containment_detail.png`: 공통 선택 저장본의 실제 예측과 공식 TXT를 비교한 사진. 원본 색 네모와 설명용 선을 구분한다.

- `common20_visual_review_examples.png`: AI 보조 시각 평가에서 확인한 표시 누락 두 사례와 중복 한 사례. 기존 IoU 오류 그림과 구분한다.
- `common20_cascade_validation_examples.png`: 단계적 추론 분석에서 확인한 YOLOv8n의 검증 위치 오류 세 사례. 공식 TXT와 네 모델의 MPS 예측을 표시했다. 세 사례 모두 검은 중심은 표시됐다. `scripts/xray_cascade_figure.py`로 생성한다.

- `common20_live_conflict.png`: 실제 검증 L002의 선 없는 확대·YOLO·RF-DETR 비교. 해당 공식 이물 주변의 예측만 표시했다. 추가 표시는 불확실하여 재검사로 보존했다.

- `20_여섯모델_공통검증F1_학습곡선.png`: 현재 여섯 모델의 20epoch 공통 검증 F1. 검증 임계값 선택·저장 시점 표시, 테스트 곡선과 구분.
