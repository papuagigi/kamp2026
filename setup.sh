#!/usr/bin/env bash
# 가상환경 만들기. 프로젝트 루트에서: bash setup.sh
# 이미 .venv가 있으면 패키지만 맞춘다. Python 3.9~3.13 어느 것이든 된다.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
if [ ! -x .venv/bin/python ]; then
  echo "[1/3] 가상환경 생성 ($($PY --version))"; $PY -m venv .venv
else
  echo "[1/3] 기존 .venv 사용 ($(.venv/bin/python --version))"
fi
echo "[2/3] 패키지 설치 (requirements.txt)"
.venv/bin/pip install -q -U pip
.venv/bin/pip install -q -r requirements.txt
echo "[3/3] 확인"
.venv/bin/python - <<'PYEOF'
import torch, ultralytics, cv2, pandas, sklearn
dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"torch {torch.__version__} | ultralytics {ultralytics.__version__} | opencv {cv2.__version__} | 학습 장치: {dev}")
PYEOF
echo "완료. 실행: bash run.sh"
