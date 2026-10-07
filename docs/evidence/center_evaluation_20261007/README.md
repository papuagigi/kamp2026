# 중심 위치 평가의 근거 — 2026-10-07

기존 저장 예측을 재채점한 사후 민감도 분석이다. 새 학습·추론·임계값 선택이 아니다. 기존 테스트를 확인한 뒤 추가한 평가이며, 현장 허용 오차를 확정한 결과가 아니다.

- `protocol.json`: 계산 전에 기록한 2·4·6·8 원본 픽셀 기준, 고정 저장본·탐지 임계값, 입력 파일 해시.
- `summary.csv`: 검증 107장·243개, 테스트 97장·225개 정답의 전체 결과. 모든 거리 기준을 보존한다.
- `complementarity.csv`: 정답 위치별로 두 모델이 보완하거나 함께 놓친 수. 합성 시스템 F1과 다르다.
- `conditions.csv`: 호기·대비·크기·경계 조건별 정답 수와 FN. 서로 겹치는 조건을 합산하지 않는다.
- `photos.csv`: 사진 단위 표시 유무. 사진에 경고가 있다고 모든 이물을 찾았다는 뜻은 아니다.
- `checks.json`: 원본 해시, 분할, 기존 12개 채점 결과 회귀 확인.
- `visual_review_decisions.json`: 중심 기준 실패 위치 40곳과 학습 라벨 12개 표본의 AI 시각 확인 기록.

재현: 프로젝트 루트에서 `.venv/bin/python scripts/xray_center_analysis.py`. 공통 채점기는 `scripts/xray_eval.py`다. 기존 데이터·선택 저장본의 예측 파일이 필요하며, 파일 위치와 해시는 protocol에 있다. 사진과 대용량 예측 원본은 로컬 reports·runs에 보존한다. 현재 공개된 근거만으로 전체 재학습이 실행되지는 않는다.

추가 근거: `targets.csv`는 정답별 위치 대응과 임계값 미달을, `extra_predictions.csv`는 미대응 예측을 보존한다. `data_audit.json`, `pair_benchmark.json`, `device_comparison.json`, `learning_curves.csv`는 앞선 데이터·실제 추론·학습 기록의 사본이다.

로컬 분석 폴더가 없다면 먼저 `reports/center_evaluation_20261007/`를 만들고 이 폴더의 `protocol.json`을 같은 이름으로 복사한다. 기존 결과 회귀 검사에 필요한 `docs/evidence/direct_training_20261006/rescore_summary.csv`도 `reports/direct_analysis_20261007/rescore_summary.csv`로 복사한다. protocol에 적힌 원본 예측·공식 라벨·고정 분할을 준비한 뒤 위 명령을 실행한다. 임계값이나 평가 라벨을 바꾸지 않는다.

공유 사본의 절대 홈 경로는 프로젝트 상대경로로 바꿨다. 수치와 원본 파일 해시는 보존했다. 보고서 파일·배치 확인 기록은 `report_edit_checks.json`에 있다.
