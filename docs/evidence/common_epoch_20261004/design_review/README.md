# 식품 FN 위험 우선 설계의 공유 근거

여섯 모델 비교·공통 오류·실제 전체 추가 검사·가공 데이터 무결성·재현 자료 검사 기록이다. 최종 연구 제안은 food_fn_priority_proposal.json에 있으며, 이전 F1 우선 proposal_frozen.json도 보존한다. 두 제안을 같은 시점의 사전 선택으로 합치지 않는다.

- six_model_validation.json: 고정 임계값 MPS 검증과 모델끼리의 공통 위치 FN. 운영 시스템의 F1이 아니다.
- latest_validation_comparison.json / latest_test_descriptive.json: 최신 모델의 학습 완료와 검증 선택·기존 테스트 결과.
- yolo8_faster_all, yolo11_rfdetr_all: 각 조합의 검증 107장×3회 실제 실행과 시간 범위.
- shared_error_audit: 반전 결과가 비슷해도 남는 오류. latest_visual/review.json은 오류 위치 한정 AI 검토다.
- replay_result.json: 새 추출 폴더의 전체 파일 해시와 여섯 저장 예측 재채점.
- package_inference_smoke.json: 저장본 6개 로드, 두 모델×3장 추론 확인. 기존 설치 환경이며 전체 새 환경 재학습은 아니다.

공유본의 개인 장치 경로는 정규화했다. 값은 유지하며 원본·공유본 해시는 final_evidence_manifest.json에 기록했다. 큰 가중치·사진·재현 ZIP은 git에 넣지 않는다. 기존 중간 기록은 당시 이력이다. 현재 결과는 위 파일과 모델 정리본 18.16~18.18절을 따른다.

- `epoch_curves.csv` / `epoch_curves_audit.json`: 현재 여섯 모델의 20epoch 공통 IoU 0.5 검증 곡선. 검증 자료에서 epoch별 탐지 임계값을 선택한 값이며 테스트 지표와 구분한다.
