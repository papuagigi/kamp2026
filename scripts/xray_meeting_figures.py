"""Create source-backed figures for docs/13 without changing dataset files."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, patches
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
DATA = ROOT / "data" / "xray_v2"
STEM = "001_20200623_043227(1)"
for name in ("AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Noto Sans CJK KR"):
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams.update({"font.size": 15, "axes.unicode_minus": False, "text.color": "#182B3B"})


def save(fig, name):
    fig.savefig(OUT / name, dpi=190, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(name)


def clean(ax, title):
    ax.set_title(title, fontsize=16, pad=12, loc="left")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def label_box(ax, box, dx=0, dy=0):
    cx, cy, w, h = box
    ax.add_patch(patches.Rectangle((cx-w/2-dx-.5, cy-h/2-dy-.5), w, h,
                                  fill=False, ec="#11B853", lw=2, ls="--"))


def box(ax, xy, width, height, text, fill, color="#182B3B", fontsize=15):
    x, y = xy
    ax.add_patch(patches.FancyBboxPatch((x, y), width, height,
                 boxstyle="round,pad=0.012,rounding_size=0.018", fc=fill, ec="none"))
    ax.text(x+width/2, y+height/2, text, ha="center", va="center", fontsize=fontsize,
            color=color, linespacing=1.55)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops={"arrowstyle": "->", "lw": 1.7, "color": "#7A8994"})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(DATA / "manifest.csv").set_index("stem")
    row = manifest.loc[STEM]
    idx = np.asarray(Image.open(ROOT / row.path))
    raw = np.asarray(Image.open(ROOT / row.path).convert("RGB"))
    processed = np.asarray(Image.open(DATA / "images" / row.split / f"{STEM}.png"))
    height, width = idx.shape
    values = [float(v) for v in (DATA / "labels" / row.split / f"{STEM}.txt").read_text().splitlines()[0].split()]
    target = (values[1]*width, values[2]*height, values[3]*width, values[4]*height)
    cx, cy, _, _ = target
    side = 36
    x0, y0 = int(cx)-side//2, int(cy)-side//2
    region = np.s_[y0:y0+side, x0:x0+side]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), gridspec_kw={"width_ratios": [1.5, 1]}, layout="constrained")
    axes[0].imshow(raw, interpolation="nearest")
    axes[0].add_patch(patches.Rectangle((x0, y0), side, side, fill=False, ec="#ECA43B", lw=2))
    clean(axes[0], "전체 사진 · 주황 테두리 안을 확대")
    axes[1].imshow(raw[region], interpolation="nearest")
    label_box(axes[1], target, x0, y0)
    clean(axes[1], "작은 검은 점과 장비의 색 표시")
    save(fig, "13_raw_example.png")

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.3), layout="constrained")
    images = [raw[region], (idx[region] >= 244).astype(np.uint8)*255, processed[region], processed[region]]
    titles = ["1 원본 확대", "2 지울 위치만 선택", "3 v2 학습 입력", "4 정답 위치 확인"]
    for i, (ax, img, title) in enumerate(zip(axes, images, titles)):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        clean(ax, title)
        if i == 3:
            label_box(ax, target, x0, y0)
    save(fig, "13_v2_process.png")

    stats = json.loads((DATA / "summary.json").read_text())
    fig, ax = plt.subplots(figsize=(11.5, 3.3))
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    colors = ["#E7F0FA", "#E4F2EC", "#FCF0DD"]
    for i, (sp, ko) in enumerate([("train", "학습"), ("val", "검증"), ("test", "평가")]):
        x = 0.025+i*.335
        box(ax, (x, .58), .27, .30, f"촬영 묶음 {chr(65+i)}\n연속 사진을 함께 유지", colors[i], fontsize=16)
        arrow(ax, x+.135, .56, x+.135, .40)
        box(ax, (x, .05), .27, .33, f"{ko} 세트\n{stats['split_images'][sp]}장 · {stats['split_bursts'][sp]}묶음", colors[i], fontsize=17)
    save(fig, "13_group_split.png")

    fig, ax = plt.subplots(figsize=(12, 5.0))
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    flows = [
        ("YOLOv8n", "#E7F0FA", ["CNN 특징 추출", "여러 크기의 특징", "위치·종류 직접 예측"]),
        ("Faster R-CNN v2", "#E4F2EC", ["ResNet50 + FPN", "후보 영역 제안\nRPN", "영역 특징 수집·재판별\nRoIAlign + 판별부"]),
        ("RF-DETR-S", "#FCF0DD", ["DINOv2 기반\nVision Transformer", "객체 쿼리로 탐지\nTransformer 디코더", "물체의 위치·종류\n집합 예측"]),
    ]
    for y, (name, color, nodes) in zip([.76, .44, .12], flows):
        ax.text(.02, y+.19, name, fontsize=17, weight="bold", va="bottom")
        for i, text in enumerate(nodes):
            x = .025+i*.335
            box(ax, (x, y-.04), .27, .20, text, color, fontsize=15)
            if i < 2:
                arrow(ax, x+.282, y+.06, x+.325, y+.06)
    save(fig, "13_model_architectures.png")

    evidence = {"example_stem": STEM, "split": row.split, "source_path": row.path,
                "label_box_pixels": target, "crop_xywh": [x0, y0, side, side],
                "source_unchanged_outside_mask": bool(np.array_equal(processed[idx < 244], idx[idx < 244])),
                "split": stats, "figures": ["13_raw_example.png", "13_v2_process.png", "13_group_split.png", "13_model_architectures.png"]}
    dest = ROOT / "reports" / "meeting_20261001"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "figure_sources.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
