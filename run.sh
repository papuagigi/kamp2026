#!/usr/bin/env bash
# v2 실행: bash run.sh [all|prepare|evaluate|dry-run]
# 과거 v1은 XRAY_DATA=xray_v1로 명시한다. 기존 학습 폴더를 덮어쓰지 않는다.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
export XRAY_DATA=${XRAY_DATA:-xray_v2}
case "$XRAY_DATA" in
  xray_v2) PREPARE=scripts/xray_prepare_v2.py; PREFIX=v2_; DOG=v2_classical_dog ;;
  xray_v1) PREPARE=scripts/xray_prepare.py; PREFIX=v1_; DOG=classical_dog ;;
  *) echo "XRAY_DATA는 xray_v2 또는 xray_v1이어야 합니다" >&2; exit 2 ;;
esac
MODEL=${MODEL:-weights/yolov8n.pt}
IMGSZ=${IMGSZ:-640}
EPOCHS=${EPOCHS:-30}
NAME=${NAME:-${PREFIX}yolov8n_${IMGSZ}}
DEVICE=${DEVICE:-auto}
RUN_DOG=${RUN_DOG:-0}
MODE=${1:-all}
case "$MODE" in all|prepare|evaluate|dry-run) ;; *) echo "사용: bash run.sh [all|prepare|evaluate|dry-run]" >&2; exit 2 ;; esac
if [[ "$NAME" != "$PREFIX"* ]]; then
  echo "실험 이름은 데이터 버전과 맞게 ${PREFIX}로 시작해야 합니다" >&2; exit 2
fi
if [[ "$MODE" == dry-run ]]; then
  echo "데이터=$XRAY_DATA 준비=$PREPARE 실험=$NAME 장치=$DEVICE DoG=$RUN_DOG"
  echo "이미지=data/$XRAY_DATA/images 채점=custom_ap_v2 (검증 문턱값 고정)"
  exit 0
fi
if [[ "$MODE" == all || "$MODE" == prepare ]]; then
  "$PY" "$PREPARE"
fi
"$PY" scripts/xray_check_dataset.py
if [[ "$MODE" == prepare ]]; then exit 0; fi
if [[ "$MODE" == all ]]; then
  "$PY" scripts/xray_train_yolo.py --model "$MODEL" --imgsz "$IMGSZ" --epochs "$EPOCHS" --name "$NAME" --device "$DEVICE"
  if [[ "$RUN_DOG" == 1 ]]; then "$PY" scripts/xray_baseline_classical.py; fi
fi
threshold_from_json() {
  "$PY" - "$1" <<'PY'
import json, sys
value = json.load(open(sys.argv[1]))["thr"]
if value is None:
    raise SystemExit("검증 세트에서 문턱값을 정할 수 없습니다. 평가 세트로 대신 정하지 않습니다")
print(value)
PY
}
score_model() {
  local model_name="$1"
  "$PY" scripts/xray_eval.py "reports/preds_${model_name}_val.csv" --split val
  local threshold
  threshold=$(threshold_from_json "reports/preds_${model_name}_val_custom_ap_v2_eval.json")
  "$PY" scripts/xray_eval.py "reports/preds_${model_name}_test.csv" --split test --thr "$threshold" --tag custom_ap_v2_valthr
}
score_model "$NAME"
if [[ "$RUN_DOG" == 1 ]]; then score_model "$DOG"; fi
if [[ "$MODE" == all ]]; then
  "$PY" scripts/xray_fake_mask.py
  "$PY" scripts/xray_predict.py --weights "runs/$NAME/weights/best.pt" --images "data/$XRAY_DATA/images/val_fakemask" --out "reports/preds_${NAME}_fakemask.csv" --imgsz "$IMGSZ" --device "$DEVICE"
  THRESHOLD=$(threshold_from_json "reports/preds_${NAME}_val_custom_ap_v2_eval.json")
  "$PY" scripts/xray_fakemask_eval.py "reports/preds_${NAME}_fakemask.csv" --thr "$THRESHOLD"
fi
echo "완료. reports/*custom_ap_v2*_eval.json에 저장했습니다."
