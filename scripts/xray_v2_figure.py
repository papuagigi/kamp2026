#!/usr/bin/env python3
"""
v1과 v2 데이터셋을 한 장에 비교하는 그림.
- 윗줄: 같은 금속구에서 원본, v1 결과, v2 결과, 두 결과의 차이 (v1은 5×5 팽창 + 텔레아, v2는 팔레트 244~255번 + 나비에-스토크스)
- 아랫줄: 학습·검증·평가 세트의 사진 수를 시험편 종류(막대 3개 / 1개)로 나눈 막대그래프, v1과 v2 나란히
출력: docs/figures/08_v1_v2_compare.png
사용: .venv/bin/python scripts/xray_v2_figure.py   (먼저 xray_prepare.py, xray_prepare_v2.py 실행)
"""
from pathlib import Path
import numpy as np, pandas as pd, cv2
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager, patches
for f in ["AppleGothic", "Apple SD Gothic Neo", "NanumGothic"]:
    if any(f == x.name for x in font_manager.fontManager.ttflist): plt.rcParams["font.family"] = f; break
plt.rcParams.update({"axes.unicode_minus": False, "font.size": 9, "axes.titlesize": 10.5, "axes.titlecolor": "#0b0b0b"})
ROOT = Path(__file__).resolve().parents[1]
V1, V2 = ROOT / "data" / "xray_v1", ROOT / "data" / "xray_v2"
TRIPLE, SINGLE, LABEL_C = "#2a78d6", "#eb6834", "#00c800"
SPLITS = ["train", "val", "test"]; SPLIT_KO = {"train": "학습", "val": "검증", "test": "평가"}

def crop_panel(ax, img, x0, y0, s, title, box, cmap=None, vmax=None):
    c = img[y0:y0 + s, x0:x0 + s]
    ax.imshow(c, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest") if cmap else ax.imshow(c, interpolation="nearest")
    if box is not None:
        cx, cy, bw, bh = box
        ax.add_patch(patches.Rectangle((cx - bw / 2 - x0 - .5, cy - bh / 2 - y0 - .5), bw, bh, fill=False, ec=LABEL_C, lw=1.4, ls="--"))
    ax.set_title(title, loc="left"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)

def main():
    m1 = pd.read_csv(V1 / "manifest.csv", encoding="utf-8-sig").set_index("stem")
    m2 = pd.read_csv(V2 / "manifest.csv", encoding="utf-8-sig").set_index("stem")
    stem = "001_20200623_043227(1)"                       # docs/06의 팔레트 예시와 같은 사진 (3호기)
    raw = np.asarray(Image.open(ROOT / m2.loc[stem, "path"]).convert("RGB"))
    a = cv2.imread(str(V1 / "images" / m1.loc[stem, "split"] / f"{stem}.png"), 0)
    b = cv2.imread(str(V2 / "images" / m2.loc[stem, "split"] / f"{stem}.png"), 0)
    H, W = a.shape
    l = [t for t in (V2 / "labels" / m2.loc[stem, "split"] / f"{stem}.txt").read_text().splitlines() if t.strip()][0].split()
    box = (float(l[1]) * W, float(l[2]) * H, float(l[3]) * W, float(l[4]) * H)
    s = 30; x0, y0 = int(box[0]) - s // 2, int(box[1]) - s // 2
    diff = np.abs(a.astype(int) - b.astype(int))

    fig = plt.figure(figsize=(14, 8.2)); gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05], hspace=0.38, wspace=0.12)
    crop_panel(fig.add_subplot(gs[0, 0]), raw, x0, y0, s, "① 원본 (검사기 표시 있음)", box)
    crop_panel(fig.add_subplot(gs[0, 1]), a, x0, y0, s, "② v1: 표시 + 둘레 2픽셀을\n지우고 텔레아로 메움", box, cmap="gray", vmax=255)
    crop_panel(fig.add_subplot(gs[0, 2]), b, x0, y0, s, "③ v2: 표시 픽셀만 지우고\n나비에-스토크스로 메움", box, cmap="gray", vmax=255)
    crop_panel(fig.add_subplot(gs[0, 3]), diff, x0, y0, s, f"④ ②와 ③의 차이 (밝을수록 큼)\n이 사진 전체에서 다른 픽셀 {int((diff > 0).sum()):,}개", box, cmap="magma", vmax=30)

    ax = fig.add_subplot(gs[1, :])
    xs, labels = [], []
    for i, (name, m) in enumerate([("v1", m1), ("v2", m2)]):
        card = np.where(m.n_boxes >= 2, "막대3", "막대1")
        tab = pd.crosstab(m.split, card).reindex(SPLITS).fillna(0)
        for j, sp in enumerate(SPLITS):
            x = i * 4 + j; xs.append(x); labels.append(f"{name}\n{SPLIT_KO[sp]}")
            t3, t1 = int(tab.loc[sp].get("막대3", 0)), int(tab.loc[sp].get("막대1", 0))
            ax.bar(x, t3, color=TRIPLE, width=0.7, edgecolor="white", linewidth=1, label="막대 3개짜리 시험편" if (i, j) == (0, 0) else None)
            ax.bar(x, t1, bottom=t3, color=SINGLE, width=0.7, edgecolor="white", linewidth=1, label="막대 1개짜리 시험편" if (i, j) == (0, 0) else None)
            ax.text(x, t3 + t1 + 4, f"{t3 + t1}장\n막대 1개 {t1 / (t3 + t1):.0%}", ha="center", va="bottom", fontsize=8.5, color="#0b0b0b")
    ax.set_xticks(xs); ax.set_xticklabels(labels); ax.set_ylabel("사진 수"); ax.set_ylim(0, 380)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color="#e6e5e0", lw=0.6); ax.set_axisbelow(True); ax.legend(frameon=False, loc="upper right")
    ax.set_title("⑤ 세트별 사진 수와 시험편 종류: v1은 평가 세트에 막대 1개짜리가 몰려 있고, v2는 세 세트가 고르다", loc="left")
    fig.text(0.01, 0.005, "초록 점선은 사람이 단 정답 라벨. 한 칸이 원본 픽셀 하나. 분할은 둘 다 점검 묶음 단위이고, v2는 호기 × 시험편 종류마다 사진 수 기준 60:20:20으로 맞췄다.",
             fontsize=8.5, color="#52514e")
    out = ROOT / "docs" / "figures" / "08_v1_v2_compare.png"; plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close(); print("saved", out)

if __name__ == "__main__":
    main()
