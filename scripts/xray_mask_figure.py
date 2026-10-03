#!/usr/bin/env python3
"""
v1 마스킹과 팔레트 마스킹을 같은 금속구 위에서 나란히 보여 주는 그림 (docs/06 8-3).
두 방식 모두 텔레아 인페인팅을 쓰고, 다른 점은 5×5 팽창을 하느냐 하나뿐이다.
출력: docs/figures/06_mask_methods.png
사용: .venv/bin/python scripts/xray_mask_figure.py   (먼저 scripts/xray_data_audit.py 실행)
"""
from pathlib import Path
import numpy as np, cv2, pandas as pd
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager, patches
from xray_prepare import RAW

for f in ["AppleGothic", "Apple SD Gothic Neo", "NanumGothic"]:
    if any(f == x.name for x in font_manager.fontManager.ttflist): plt.rcParams["font.family"] = f; break
plt.rcParams.update({"axes.unicode_minus": False, "font.size": 9, "axes.titlesize": 9.5, "axes.titlecolor": "#0b0b0b"})
ROOT = Path(__file__).resolve().parents[1]
AUD = ROOT / "reports" / "data_audit"; FIG = ROOT / "docs" / "figures"
LABEL_C, RING_C = "#00c800", "#eb6834"
K5 = np.ones((5, 5), np.uint8)
CASES = [("001_20200623_043227(1)", 155, "보통 사례. 표시 선이 금속구 중심에서 7픽셀쯤 떨어져 있다."),
         ("001_20200624_203225(8)", 248, "가장 가까운 사례. 표시 선은 5픽셀, v1이 지운 영역은 3픽셀까지 다가온다.")]
HALF = 17                                   # 금속구 중심에서 위아래·좌우로 자를 픽셀 수

def main():
    raw = {}
    for p in sorted(RAW.rglob("*.bmp")): raw.setdefault(p.stem, p)
    lb = pd.read_csv(AUD / "label_boxes.csv")
    fig, axes = plt.subplots(len(CASES), 6, figsize=(13, 5.4))
    for row, (stem, ball_y, note) in enumerate(CASES):
        im = Image.open(raw[stem]); idx = np.asarray(im); rgb = np.asarray(im.convert("RGB"))
        gray = idx.astype(np.uint8); M = (idx >= 244).astype(np.uint8)
        D = cv2.dilate(M, K5)
        v1 = cv2.inpaint(gray, D, 3, cv2.INPAINT_TELEA); pal = cv2.inpaint(gray, M, 3, cv2.INPAINT_TELEA)
        b = lb[(lb.stem == stem) & (lb.ball_y == ball_y)].iloc[0]
        cx, cy = int(b.ball_x), int(b.ball_y)
        ys, xs = slice(cy - HALF, cy + HALF + 1), slice(cx - HALF, cx + HALF + 1)

        def show_gray(a, img): a.imshow(img[ys, xs], cmap="gray", vmin=0, vmax=243, interpolation="nearest")
        def overlay(a, m_black, m_ring=None):
            base = np.stack([gray[ys, xs]] * 3, -1).astype(float) / 243
            base[m_black[ys, xs] > 0] = 0
            if m_ring is not None: base[m_ring[ys, xs] > 0] = np.array(matplotlib.colors.to_rgb(RING_C))
            a.imshow(base, interpolation="nearest")
        def label_box(a):
            x0, y0 = b.cx - b.w / 2 - (cx - HALF) - 0.5, b.cy - b.h / 2 - (cy - HALF) - 0.5
            a.add_patch(patches.Rectangle((x0, y0), b.w, b.h, fill=False, ec=LABEL_C, lw=1.2, ls="--"))

        ax = axes[row]
        ax[0].imshow(rgb[ys, xs], interpolation="nearest"); label_box(ax[0])
        overlay(ax[1], M, D - M)
        show_gray(ax[2], v1); label_box(ax[2])
        overlay(ax[3], M)
        show_gray(ax[4], pal); label_box(ax[4])
        diff = np.abs(v1.astype(int) - pal.astype(int))
        ax[5].imshow(diff[ys, xs], cmap="magma", vmin=0, vmax=30, interpolation="nearest"); label_box(ax[5])
        titles = ["① 원본", "② v1이 지우는 픽셀\n검정 표시 + 주황 팽창", "③ v1 결과",
                  "④ 팔레트 방식이\n지우는 픽셀 (검정)", "⑤ 팔레트 방식 결과", "⑥ ③과 ⑤의 차이"]
        for a, t in zip(ax, titles):
            a.set_xticks([]); a.set_yticks([])
            if row == 0: a.set_title(t, loc="left")
        ax[0].text(0, -0.04, f"{stem}  {note}", transform=ax[0].transAxes, va="top", fontsize=8.5, color="#0b0b0b")
        ring_in_box = (D - M)[int(b.cy - b.h / 2):int(np.ceil(b.cy + b.h / 2)), int(b.cx - b.w / 2):int(np.ceil(b.cx + b.w / 2))].sum()
        ax[5].text(0, -0.04, f"라벨 박스 안에서 v1만 바꾼\n진짜 픽셀 {int(ring_in_box)}개", transform=ax[5].transAxes, va="top", fontsize=8.5, color="#0b0b0b")
    fig.text(0.01, 0.01, "초록 점선은 사람이 단 정답 라벨. 두 방식 모두 텔레아 인페인팅으로 메웠고, 다른 점은 5×5 팽창 하나뿐이다. "
             "한 칸이 원본 픽셀 하나. ⑥은 밝을수록 두 결과가 크게 다르다(0~30단계).", fontsize=8.5, color="#52514e")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.1, wspace=0.08, hspace=0.3)
    out = FIG / "06_mask_methods.png"; plt.savefig(out, dpi=130); plt.close(); print("saved", out)

if __name__ == "__main__":
    main()
