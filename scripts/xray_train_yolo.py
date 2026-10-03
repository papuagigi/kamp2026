#!/usr/bin/env python3
"""
data/xray_v2 (촬영 묶음 분할)로 YOLO를 학습하고, val/test 예측을 CSV로 저장한다. 과거 v1은 XRAY_DATA=xray_v1로 선택한다.
사용: .venv/bin/python scripts/xray_train_yolo.py --model weights/yolov8n.pt --imgsz 640 --epochs 30 --name v2_yolov8n_640_new
예측 CSV: reports/preds_<name>_<split>.csv (stem, cx, cy, w, h, score; 픽셀 단위)
"""
import argparse, time, csv, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
from xray_config import DATA, select_device
REPORTS = ROOT / "reports"

def predict_to_csv(model, split, name, imgsz, device):
    REPORTS.mkdir(exist_ok=True)
    imgs = sorted((DATA / "images" / split).glob("*.png"))
    out = REPORTS / f"preds_{name}_{split}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["stem", "cx", "cy", "w", "h", "score"])
        for i in range(0, len(imgs), 32):
            for r in model.predict([str(p) for p in imgs[i:i+32]], imgsz=imgsz, conf=0.001, iou=0.5, device=device, verbose=False, max_det=50):
                stem = Path(r.path).stem
                for (cx, cy, bw, bh), s in zip(r.boxes.xywh.tolist(), r.boxes.conf.tolist()):
                    w.writerow([stem, f"{cx:.1f}", f"{cy:.1f}", f"{bw:.1f}", f"{bh:.1f}", f"{s:.4f}"])
    print("saved", out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="weights/yolov8n.pt"); ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--name", required=True)
    ap.add_argument("--device", default="auto"); ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    a.device = select_device(a.device)
    from xray_check_dataset import check_dataset
    check_dataset(DATA)
    if DATA.name == "xray_v2" and not a.name.startswith("v2_"):
        ap.error("v2 실험 이름은 v2_로 시작해야 합니다")
    if (ROOT / "runs" / a.name).exists():
        ap.error("기존 학습 결과가 있습니다. --name에 새 실험 이름을 지정하세요")
    print(f"데이터: {DATA.name}, 학습 장치: {a.device}")
    from ultralytics import YOLO
    model = YOLO(a.model); t0 = time.time()
    model.train(data=str(DATA / "data.yaml"), epochs=a.epochs, imgsz=a.imgsz, device=a.device, batch=a.batch, workers=2,
                project=str(ROOT / "runs"), name=a.name, exist_ok=True, verbose=False, plots=False, seed=0, deterministic=True, patience=100)
    print(f"학습 {a.name} ({a.device}): {a.epochs} epoch, imgsz {a.imgsz}, {time.time()-t0:.0f}초")
    best = YOLO(str(ROOT / "runs" / a.name / "weights" / "best.pt"))
    for split in ["val", "test"]:
        predict_to_csv(best, split, a.name, a.imgsz, a.device)

if __name__ == "__main__":
    main()
