# 새 AI 라벨 자료의 학습·선택 기록

학습 사진 12,161장, 검증 107장, 테스트 97장으로 완료한 YOLO11s·RF-DETR-S 실험의 작은 근거 묶음이다. 대용량 영상·TXT 전체·가중치·예측 원본은 저장소에 포함하지 않는다.

- `dataset_summary.json`: 라벨·증강 구성 수량과 원본 provenance 해시.
- `labeling_final_checks.json`, `labeling_geometry_visual_qa.json`: 후보 1,900장 중 채택 1,876장·보류 24장과 AI 시각 검토·좌표 검사 기록. 사람이 전수 라벨링한 자료라는 뜻은 아니다.
- 모델별 `execution.json`, `validation/epoch_*.json`: 20epoch 학습과 매 epoch 검증 기록.
- 모델별 `selection.json`, `evaluation_frozen.json`: 검증으로 선택한 저장본·임계값과 해시. YOLO11s 2epoch, RF-DETR-S 10epoch다.
- 모델별 `evaluation.json`: 선택 뒤 고정 설정으로 수행한 기존 박스 평가. 새 위치 평가와 점수의 의미가 다르다.
- `rescore_audit.json`, `rescore_summary.csv`, `consistency_checks.json`: 저장 예측·선택 기록 재확인 결과.
- `source_manifest.json`: 원본 해시와 공유 사본의 해시. 로컬 홈 경로만 상대경로로 바꾸고 수치는 보존했다.

현재 제출 본문의 위치 점수·규칙은 [중심 위치 평가 근거](../center_evaluation_20261007/README.md)를 따른다. 기존 IoU 결과는 연구 이력으로 보존한다. RF는 CUDA 14epoch 뒤 MPS로 20epoch까지 이어 학습했으며 선택된 10epoch 저장본은 CUDA 시점이다. 전체 재학습이 가능한 입력 자료를 모두 공개한 묶음은 아니다.
