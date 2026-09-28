#!/usr/bin/env python3
"""
가이드북 실습 가중치가 색 표시에 기대는지 보여 주는 그림.
xray_guidebook_probe.py 가 만든 예측(reports/preds_guidebook_last*_{raw,masked}_labelonly100.csv)에서
사진마다 가장 높은 점수를 뽑아, 색 네모가 남은 원본과 지운 사진을 나란히 그린다. 가이드북 기본 문턱값 0.3도 표시한다.
출력: docs/figures/07_guidebook_mark_dependence.png
사용: .venv/bin/python scripts/xray_guidebook_figure.py
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager
for f in ["AppleGothic", "Apple SD Gothic Neo", "NanumGothic"]:
    if any(f == x.name for x in font_manager.fontManager.ttflist): plt.rcParams["font.family"] = f; break
plt.rcParams.update({"axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
                     "axes.edgecolor": "#8a8984", "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.labelcolor": "#52514e"})
ROOT = Path(__file__).resolve().parents[1]; REPORTS = ROOT / "reports"
RAW_C, MASK_C = "#2a78d6", "#eb6834"

def top_scores(n, cond):
    p = REPORTS / f"preds_guidebook_last{n}_{cond}_labelonly100.csv"
    stems = (REPORTS / "guidebook_probe" / "label_only_100.txt").read_text().split()
    s = pd.read_csv(p).groupby("stem").score.max().reindex(stems, fill_value=0.0)
    return s.values

def main():
    ns = [15, 50, 100, 200, 300, 400]
    fig, ax = plt.subplots(figsize=(11, 4.6)); rng = np.random.default_rng(0)
    for i, n in enumerate(ns):
        for off, cond, c in [(-0.18, "raw", RAW_C), (0.18, "masked", MASK_C)]:
            v = top_scores(n, cond); x = i + off + rng.uniform(-0.07, 0.07, len(v))
            ax.scatter(x, v, s=9, color=c, alpha=0.55, linewidths=0, label=None)
            ax.plot([i + off - 0.12, i + off + 0.12], [np.median(v)] * 2, color="#0b0b0b", lw=1.6)
    ax.axhline(0.3, color="#52514e", lw=1, ls="--"); ax.text(len(ns) - 0.45, 0.31, "가이드북 기본 문턱값 0.3", ha="right", va="bottom", color="#52514e")
    ax.set_xticks(range(len(ns))); ax.set_xticklabels([f"실습 {n}장\n가중치" for n in ns]); ax.set_ylim(0, 1.0)
    ax.set_ylabel("사진별 최고 점수"); ax.grid(axis="y", color="#e6e5e0", lw=0.6); ax.set_axisbelow(True)
    ax.scatter([], [], s=25, color=RAW_C, label="색 네모가 남은 원본 (가이드북이 학습한 형태)")
    ax.scatter([], [], s=25, color=MASK_C, label="색 네모를 지운 사진")
    ax.plot([], [], color="#0b0b0b", lw=1.6, label="중앙값")
    ax.legend(loc="upper left", frameon=False, ncol=3)
    ax.set_title("가이드북 실습 가중치가 처음 보는 100장에서 낸 점수: 색 네모를 지우면 점수가 크게 떨어진다", loc="left", fontsize=11)
    plt.tight_layout(); out = ROOT / "docs" / "figures" / "07_guidebook_mark_dependence.png"; plt.savefig(out, dpi=130); plt.close(); print("saved", out)

if __name__ == "__main__":
    main()
