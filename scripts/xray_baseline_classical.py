#!/usr/bin/env python3
"""
딥러닝 없는 기준선: 마스킹된 흑백 사진에서 '주변보다 어두운 작은 점'을 찾는다.
- 작은 가우시안(σ1)과 큰 가우시안(σ2)의 차(DoG)가 어두운 점에서 양수 봉우리가 된다
- 봉우리를 국소 최대로 골라내고, 봉우리 높이를 국소 노이즈로 나눈 값을 점수로 쓴다
- 제품 영역 안(가장자리 4px 제외)만 본다
사용: .venv/bin/python scripts/xray_baseline_classical.py  → reports/preds_v2_classical_dog_{val,test}.csv
      v2: XRAY_DATA=xray_v2 .venv/bin/python scripts/xray_baseline_classical.py  → reports/preds_v2_classical_dog_{val,test}.csv
"""
import csv, os, sys
from pathlib import Path
import numpy as np, cv2, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; REPORTS = ROOT / "reports"
from xray_config import DATA, select_device
TAG = "" if DATA.name == "xray_v1" else DATA.name.replace("xray_", "") + "_"   # v1 파일 이름은 예전 그대로
from xray_eval import product_mask

def detect(gray, s1=1.2, s2=3.0, box=12, max_det=30):
    g = gray.astype(np.float32)
    dog = cv2.GaussianBlur(g, (0, 0), s2) - cv2.GaussianBlur(g, (0, 0), s1)   # 어두운 점 → 양수
    pm = cv2.erode(product_mask(gray), np.ones((9, 9), np.uint8))
    noise = cv2.GaussianBlur(np.abs(dog), (0, 0), 8) + 1e-3                   # 국소 노이즈 수준
    score = dog / noise
    score[pm == 0] = -1
    peak = (score == cv2.dilate(score, np.ones((7, 7), np.uint8))) & (score > 1.0)
    ys, xs = np.where(peak); vals = score[ys, xs]; order = np.argsort(-vals)[:max_det]
    return [(float(xs[i]), float(ys[i]), float(box), float(box), float(vals[i])) for i in order]

def main():
    REPORTS.mkdir(exist_ok=True)
    if "--images" in sys.argv:   # 임의 폴더 예측: --images <PNG 폴더> --out <CSV>
        folder, out = Path(sys.argv[sys.argv.index("--images") + 1]), Path(sys.argv[sys.argv.index("--out") + 1])
        with open(out, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["stem", "cx", "cy", "w", "h", "score"])
            for p in sorted(folder.glob("*.png")):
                gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                for cx, cy, bw, bh, s in detect(gray): w.writerow([p.stem, f"{cx:.1f}", f"{cy:.1f}", bw, bh, f"{s:.4f}"])
        print("saved", out); return
    splits = sys.argv[sys.argv.index("--splits") + 1].split(",") if "--splits" in sys.argv else ["val", "test"]   # 예: --splits train,val,test
    for split in splits:
        out = REPORTS / f"preds_{TAG}classical_dog_{split}.csv"
        with open(out, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["stem", "cx", "cy", "w", "h", "score"])
            for p in sorted((DATA / "images" / split).glob("*.png")):
                gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                for cx, cy, bw, bh, s in detect(gray): w.writerow([p.stem, f"{cx:.1f}", f"{cy:.1f}", bw, bh, f"{s:.4f}"])
        print("saved", out)

if __name__ == "__main__":
    main()
