#!/usr/bin/env python3
"""
마스킹 방식(v1, v2)이 딥러닝 없는 점 검출(DoG) 기준선에 주는 영향. 분할 차이를 빼고 마스킹만 비교하려고 라벨 사진 500장 전체를 쓴다.
- v1 사진(data/xray_v1)과 v2 사진(data/xray_v2) 500장 모두에 같은 DoG 검출을 돌려 예측 CSV를 만든다
  → reports/preds_classical_dog_all500.csv, reports/preds_v2_classical_dog_all500.csv (채점은 xray_eval.py로 따로)
- 정답 금속구 자리에서 DoG 점수를 "점의 깊이(DoG 값)"와 "주변 잡음"으로 나눠 v1과 v2를 비교한다 → reports/dog_mask_effect.csv, 화면 요약
사용: .venv/bin/python scripts/xray_dog_mask_effect.py
채점: XRAY_DATA=xray_v1 .venv/bin/python scripts/xray_eval.py reports/preds_classical_dog_all500.csv --split train,val,test
      XRAY_DATA=xray_v2 .venv/bin/python scripts/xray_eval.py reports/preds_v2_classical_dog_all500.csv --split train,val,test
"""
import csv, json
from pathlib import Path
import numpy as np, pandas as pd, cv2
from xray_baseline_classical import detect
ROOT = Path(__file__).resolve().parents[1]; REPORTS = ROOT / "reports"
SETS = {"v1": ROOT / "data" / "xray_v1", "v2": ROOT / "data" / "xray_v2"}

def dog_parts(gray, s1=1.2, s2=3.0):
    """xray_baseline_classical.detect 와 같은 계산에서 점의 깊이와 주변 잡음을 따로 돌려준다."""
    g = gray.astype(np.float32); dog = cv2.GaussianBlur(g, (0, 0), s2) - cv2.GaussianBlur(g, (0, 0), s1)
    return dog, cv2.GaussianBlur(np.abs(dog), (0, 0), 8) + 1e-3

def main():
    man = {k: pd.read_csv(v / "manifest.csv", encoding="utf-8-sig").set_index("stem") for k, v in SETS.items()}
    stems = sorted(man["v2"].index)
    imgs = {k: {s: cv2.imread(str(SETS[k] / "images" / man[k].loc[s, "split"] / f"{s}.png"), cv2.IMREAD_GRAYSCALE) for s in stems} for k in SETS}
    for k, name in [("v1", "preds_classical_dog_all500.csv"), ("v2", "preds_v2_classical_dog_all500.csv")]:
        with open(REPORTS / name, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["stem", "cx", "cy", "w", "h", "score"])
            for s in stems:
                for cx, cy, bw, bh, sc in detect(imgs[k][s]): w.writerow([s, f"{cx:.1f}", f"{cy:.1f}", bw, bh, f"{sc:.4f}"])
        print("saved", REPORTS / name)
    rows = []
    for s in stems:
        parts = {k: dog_parts(imgs[k][s]) for k in SETS}; H, W = imgs["v2"][s].shape
        for l in (SETS["v2"] / "labels" / man["v2"].loc[s, "split"] / f"{s}.txt").read_text().splitlines():
            if not l.strip(): continue
            _, cx, cy, _, _ = map(float, l.split()); x, y = int(cx * W), int(cy * H)
            row = dict(stem=s, machine=man["v2"].loc[s, "machine"])
            for k, (dog, noise) in parts.items():   # 라벨 중심 ±3픽셀 안에서 점수가 가장 높은 자리
                win = (slice(max(0, y - 3), y + 4), slice(max(0, x - 3), x + 4)); sc = dog[win] / noise[win]
                i = np.unravel_index(np.argmax(sc), sc.shape)
                row.update({f"depth_{k}": float(dog[win][i]), f"noise_{k}": float(noise[win][i]), f"score_{k}": float(sc[i])})
            rows.append(row)
    d = pd.DataFrame(rows); d.to_csv(REPORTS / "dog_mask_effect.csv", index=False, encoding="utf-8-sig")
    q = lambda c: {k: round(float(v), 3) for k, v in d[c].quantile([0.1, 0.5, 0.9]).items()}
    summ = {f"{c}_{k}": q(f"{c}_{k}") for c in ["depth", "noise", "score"] for k in SETS}
    summ.update(n_balls=len(d), score_v1_higher=round(float((d.score_v1 > d.score_v2).mean()), 3),
                noise_v1_lower=round(float((d.noise_v1 < d.noise_v2).mean()), 3),
                depth_diff_median=round(float((d.depth_v1 - d.depth_v2).median()), 3),
                noise_ratio_median=round(float((d.noise_v1 / d.noise_v2).median()), 3))
    print(json.dumps(summ, ensure_ascii=False, indent=1)); print("saved", REPORTS / "dog_mask_effect.csv")

if __name__ == "__main__":
    main()
