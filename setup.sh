#!/usr/bin/env bash
# 가상환경 만들기. 프로젝트 루트에서: bash setup.sh
# Python 버전이 다른 기존 환경은 자동 교체하지 않는다.
set -euo pipefail
cd "$(dirname "$0")"
command -v uv >/dev/null || { echo "uv를 설치하고 터미널의 PATH를 갱신하세요." >&2; exit 1; }
PY_VERSION=$(tr -d '[:space:]' < .python-version)
if [ ! -x .venv/bin/python ]; then
  echo "[1/3] 가상환경 생성 (Python $PY_VERSION)"
  uv venv --python "$PY_VERSION" --seed .venv
else
  ACTUAL_VERSION=$(.venv/bin/python -c 'import platform; print(platform.python_version())')
  if [ "$ACTUAL_VERSION" != "$PY_VERSION" ]; then
    echo "기존 Python $ACTUAL_VERSION과 지정 버전 $PY_VERSION이 다릅니다. 기존 환경을 보존한 뒤 전환하세요." >&2
    exit 1
  fi
  echo "[1/3] 기존 .venv 사용 ($(.venv/bin/python --version))"
fi
echo "[2/3] 패키지 설치 (직접 의존성 및 잠금 목록)"
uv pip install --python .venv/bin/python -r requirements.txt -r requirements.lock.txt
uv pip check --python .venv/bin/python
echo "[3/3] 확인"
.venv/bin/python - <<'PYEOF'
import torch, ultralytics, cv2, pandas, sklearn
from rfdetr import RFDETRSmall
dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"torch {torch.__version__} | ultralytics {ultralytics.__version__} | opencv {cv2.__version__} | 학습 장치: {dev}")
PYEOF
echo "완료. 실행: bash run.sh"
