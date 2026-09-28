#!/usr/bin/env bash
# 원클릭 재현 파이프라인 (초안). 프로젝트 루트에서 실행: bash run.sh
# 순서: 데이터 준비(마스킹·분할) → 고전 기준선 → YOLO 학습·예측 → 채점 → 마스킹 흔적 검사
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
MODEL=${MODEL:-weights/yolov8n.pt}; IMGSZ=${IMGSZ:-640}; EPOCHS=${EPOCHS:-30}; NAME=${NAME:-v1_yolov8n_640}
DEVICE=${DEVICE:-$($PY -c "import torch;print('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')")}
echo "[1/5] 데이터 준비";            $PY scripts/xray_prepare.py
echo "[2/5] 고전 기준선";            (cd scripts && ../$PY xray_baseline_classical.py)
echo "[3/5] YOLO 학습 ($NAME, $DEVICE)"; $PY scripts/xray_train_yolo.py --model "$MODEL" --imgsz "$IMGSZ" --epochs "$EPOCHS" --name "$NAME" --device "$DEVICE"
echo "[4/5] 채점";                   for s in val test; do $PY scripts/xray_eval.py "reports/preds_classical_dog_$s.csv" --split "$s"; $PY scripts/xray_eval.py "reports/preds_${NAME}_$s.csv" --split "$s"; done
echo "[5/5] 마스킹 흔적 검사";       (cd scripts && ../$PY xray_fake_mask.py) && $PY scripts/xray_predict.py --weights "runs/$NAME/weights/best.pt" --images data/xray_v1/images/val_fakemask --out "reports/preds_${NAME}_fakemask.csv" --imgsz "$IMGSZ" --device "$DEVICE"
echo "완료. 결과: reports/*.json, reports/*.csv"
