#!/usr/bin/env python3
"""학습된 YOLO 가중치로 임의 폴더의 사진을 예측해 CSV로 저장. 사용: xray_predict.py --weights runs/x/weights/best.pt --images data/xray_v2/images/val_fakemask --out reports/preds_x_fakemask.csv --imgsz 640"""
import argparse, csv
from pathlib import Path
from xray_config import ROOT, select_device
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--weights", required=True); ap.add_argument("--images", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", type=int, default=640); ap.add_argument("--device", default="auto"); a = ap.parse_args()
    a.device = select_device(a.device)
    a.weights = str(ROOT / a.weights)
    a.images = str(ROOT / a.images)
    a.out = str(ROOT / a.out)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    print(f"예측 장치: {a.device}")
    from ultralytics import YOLO
    model = YOLO(a.weights); imgs = sorted(Path(a.images).glob("*.png"))
    if not imgs:
        ap.error("예측할 PNG 사진이 없습니다")
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["stem", "cx", "cy", "w", "h", "score"])
        for i in range(0, len(imgs), 32):
            for r in model.predict([str(p) for p in imgs[i:i+32]], imgsz=a.imgsz, conf=0.001, iou=0.5, device=a.device, verbose=False, max_det=50):
                for (cx, cy, bw, bh), s in zip(r.boxes.xywh.tolist(), r.boxes.conf.tolist()):
                    w.writerow([Path(r.path).stem, f"{cx:.1f}", f"{cy:.1f}", f"{bw:.1f}", f"{bh:.1f}", f"{s:.4f}"])
    print("saved", a.out)
if __name__ == "__main__": main()
