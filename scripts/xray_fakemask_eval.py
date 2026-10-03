#!/usr/bin/env python3
"""
마스킹 흔적 학습 검사. 예측 CSV와 data/xray_v2/fake_positions.csv를 맞춰,
가짜 마스킹 자리(fake)와 아무것도 안 한 자리(control)에서 임계값 이상 탐지가 나온 비율을 비교한다.
사용: xray_fakemask_eval.py reports/preds_x_fakemask.csv --thr 0.25
"""
import argparse, os
import pandas as pd, numpy as np
from pathlib import Path
from xray_config import ROOT, DATA
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("pred_csv"); ap.add_argument("--thr", type=float, required=True); a = ap.parse_args()
    preds = pd.read_csv(a.pred_csv); preds = preds[preds.score >= a.thr]
    pos = pd.read_csv(DATA / "fake_positions.csv")   # 환경변수 XRAY_DATA (기본 xray_v2)
    hit = []
    for r in pos.itertuples():
        p = preds[preds.stem == r.stem]
        hit.append(bool(len(p) and (((p.cx - r.cx)**2 + (p.cy - r.cy)**2) ** 0.5 <= 10).any()))
    pos["hit"] = hit
    g = pos.groupby("kind").hit.agg(["mean", "sum", "size"])
    print(f"임계값 {a.thr}: 가짜 마스킹 자리 탐지율 {g.loc['fake','mean']:.3f} ({int(g.loc['fake','sum'])}/{int(g.loc['fake','size'])}) vs 대조 자리 {g.loc['control','mean']:.3f} ({int(g.loc['control','sum'])}/{int(g.loc['control','size'])})")
    print("해석: 두 비율이 비슷하면 흔적을 배우지 않은 것. 가짜 자리가 뚜렷이 높으면 마스킹 흔적을 단서로 쓰고 있음.")
if __name__ == "__main__": main()
