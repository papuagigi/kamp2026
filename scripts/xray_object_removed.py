#!/usr/bin/env python3
"""
'이물 자체를 보고 있는가' 검사. 마스킹된 val 사진에서 정답 이물 자리를 주변 밝기로 지워(인페인팅) 이물만 없앤다.
마스킹 흔적(네모 테두리 자리)은 그대로 남는다. 탐지기가 그 자리를 여전히 이물로 잡으면 흔적이나 위치를 외운 것이다.
출력: data/<데이터>/images/<split>_objremoved/*.png, 그리고 예측 후 채점. 데이터는 환경변수 XRAY_DATA(기본 xray_v2), split은 --split(기본 val).
사용: xray_object_removed.py --weights runs/x/weights/best.pt --imgsz 640 --thr 0.33
"""
import argparse, csv, os
from pathlib import Path
import numpy as np, cv2, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
from xray_config import DATA, select_device
def build(split="val"):
    out = DATA / "images" / f"{split}_objremoved"; out.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(DATA / "manifest.csv", encoding="utf-8-sig"); man = man[man.split == split]
    for r in man.itertuples():
        g = cv2.imread(str(DATA / "images" / split / f"{r.stem}.png"), cv2.IMREAD_GRAYSCALE); m = np.zeros_like(g)
        for l in (DATA / "labels" / split / f"{r.stem}.txt").read_text().splitlines():
            if not l.strip(): continue
            _, cx, cy, w, h = map(float, l.split()); x, y = int(cx * r.width), int(cy * r.height)
            cv2.rectangle(m, (x - 7, y - 7), (x + 7, y + 7), 255, -1)
        cv2.imwrite(str(out / f"{r.stem}.png"), cv2.inpaint(g, m, 5, cv2.INPAINT_TELEA))
    return out
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--weights", required=True); ap.add_argument("--imgsz", type=int, default=640); ap.add_argument("--thr", type=float, required=True); ap.add_argument("--device", default="auto")
    ap.add_argument("--split", default="val"); a = ap.parse_args(); a.device = select_device(a.device); folder = build(a.split)
    from ultralytics import YOLO
    model = YOLO(a.weights); imgs = sorted(folder.glob("*.png")); hits = 0; total = 0
    preds = {}
    for i in range(0, len(imgs), 32):
        for res in model.predict([str(p) for p in imgs[i:i+32]], imgsz=a.imgsz, conf=a.thr, device=a.device, verbose=False):
            preds[Path(res.path).stem] = res.boxes.xywh.tolist()
    man = pd.read_csv(DATA / "manifest.csv", encoding="utf-8-sig"); man = man[man.split == a.split]
    for r in man.itertuples():
        for l in (DATA / "labels" / a.split / f"{r.stem}.txt").read_text().splitlines():
            if not l.strip(): continue
            _, cx, cy, w, h = map(float, l.split()); x, y = cx * r.width, cy * r.height; total += 1
            hits += any(((px - x)**2 + (py - y)**2) ** 0.5 <= 10 for px, py, *_ in preds.get(r.stem, []))
    print(f"이물을 지운 자리에서 여전히 탐지된 비율: {hits/total:.3f} ({hits}/{total}), 임계값 {a.thr}")
    print("해석: 0에 가까우면 모델이 이물 자체를 보고 있음. 높으면 마스킹 흔적이나 위치를 외운 것.")
if __name__ == "__main__": main()
