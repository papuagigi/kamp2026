# 두 모델 실제 연결 시험 근거

검증 107장·공식 이물 243개를 맥북 MPS에서 3회 추론했다. YOLO 원래 사진·좌우 반전 비교 후 필요한 사진만 RF-DETR-S로 전달했다. 기존 저장본·탐지 임계값·분할은 보존했다.

- `policy.json`: 실행 전 고정한 모델·규칙·해시. `execution.json`: 세 번의 실제 상태·예측·시간·메모리 기록.
- `summary.json`: 처리 경로·오류·지연·대조 재채점. operational_f1은 null이다. 의견 충돌을 보류하는 정책과 예측 교체 대조 실험을 구분한다.
- `visual_scope.json`, `visual/cases.json`, `visual_review.json`: 추가 검사 또는 IoU 오류가 있는 10장·28개 공식 이물의 국소 시각 검토. 검증 전체 시각 F1이 아니다. L002의 추가 YOLO 표시는 판단 어려움으로 남겼다.
- CSV: 세 반복의 원래·반전·두 번째 모델 예측과 교체 대조 예측. 1행 1박스, 픽셀 좌표.
- `manifest.json`: 원본 경로와 SHA-256. 전체 확대 그림은 로컬 `reports/.../cascade_live/visual/`에 보존했다. 공유 실제 예시는 `docs/figures/common20_live_conflict.png`다.

실행: `.venv/bin/python scripts/xray_cascade_live.py` → `scripts/xray_cascade_live_audit.py` → `scripts/xray_cascade_live_gallery.py`. 이미 완료한 출력 폴더에는 새 실행을 덮어쓰지 않는다. 시각 판단은 저장된 AI 검토자의 관측이며 그림 생성만으로 다시 검토되지 않는다.

정상 제품 0장, 재사용 검증 자료, 단일 맥북 측정이다. 1초 개발 예산은 동기 호출이 반환된 뒤 확인하며 강제 종료 watchdog이 아니다. 실제 공장 처리량·정상 오경보율·자동 PASS 안전성은 미검증이다.
