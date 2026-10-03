#!/usr/bin/env python3
"""
원본 데이터 설명용 그림. xray_data_audit.py 결과(reports/data_audit/)와 원본 사진을 읽어 docs/figures/06_*.png 를 만든다.
- 06_raw_walkthrough.png : 원본 사진 한 장 읽기 (색 네모, 정답 라벨, 지워야 할 픽셀, 마스킹 후)
- 06_image_types.png     : 사진 종류 (시험편 막대 3개·1개, 위치 이상 사진, 표시나 라벨이 빠진 사진)
- 06_time_pattern.png    : 하루 중 촬영 시각 분포, 날짜별·호기별 사진 수 (시험편 종류별)
- 06_object_stats.png    : 금속구 라벨 크기, 대비, 제품 가장자리까지 거리 (호기별)
사용: .venv/bin/python scripts/xray_data_figures.py   (먼저 scripts/xray_data_audit.py 실행)
"""
import json
from pathlib import Path
import numpy as np, pandas as pd, cv2
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager, patches
from xray_prepare import mask_image

for f in ["AppleGothic", "Apple SD Gothic Neo", "NanumGothic"]:
    if any(f == x.name for x in font_manager.fontManager.ttflist): plt.rcParams["font.family"] = f; break
plt.rcParams.update({"axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.titlesize": 11, "axes.titlecolor": "#0b0b0b", "font.size": 9})
ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "제조AI데이터셋" / "4. X-ray 검사장비 AI 데이터셋" / "dataset"
RAW = DS / "test1" / "yolov3" / "X선이물검출기(06.23_09.22)"; LAB = DS / "라벨링 6종 세트" / "labels"
AUD = ROOT / "reports" / "data_audit"; FIG = ROOT / "docs" / "figures"
BLUE, ORANGE, AQUA, GRAY, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c", "#52514e"   # 분류용 색 1~3번, 중립 회색, 보조 글자색
LABEL_C = "#00c800"                                                                   # 정답 라벨 표시 (색 네모 색과 겹치지 않게)
MACHINES = ["1호기", "2호기", "3호기"]

def rgb(path): return np.asarray(Image.open(path).convert("RGB"))
def boxes(stem, W, H):
    p = LAB / f"{stem}.txt"
    if not p.exists(): return []
    return [(float(a) * W, float(b) * H, float(c) * W, float(d) * H) for _, a, b, c, d in
            (l.split() for l in p.read_text().splitlines() if l.strip())]

def walkthrough(raw):
    stem = "001_20200623_123232(3)"                     # 3호기, 막대 3개 시험편, 라벨 3개
    r = raw[(raw.stem == stem) & raw.first_copy].iloc[0]; path = RAW / r.file
    img = rgb(path); H, W = img.shape[:2]; bx = boxes(stem, W, H)
    cx, cy, bw, bh = sorted(bx, key=lambda b: b[1])[1]  # 가운데 금속구
    x0, y0, s = int(cx) - 16, int(cy) - 16, 32
    idx = np.asarray(Image.open(path)); mk = (idx >= 244).astype(np.uint8); dil = cv2.dilate(mk, np.ones((5, 5), np.uint8))
    masked, _ = mask_image(path)
    fig, ax = plt.subplots(1, 4, figsize=(15, 5.0), gridspec_kw={"width_ratios": [1.6, 1, 1, 1]})
    ax[0].imshow(img); ax[0].add_patch(patches.Rectangle((x0, y0), s, s, fill=False, ec="#0b0b0b", lw=1.2))
    ax[0].set_title(f"① 원본 사진 (3호기, {W}×{H} 픽셀)\n{stem}.bmp  (검은 네모 = 오른쪽 확대 영역)", loc="left")
    crop = lambda a: a[y0:y0 + s, x0:x0 + s]
    def draw_label(a):
        a.add_patch(patches.Rectangle((cx - bw / 2 - x0 - .5, cy - bh / 2 - y0 - .5), bw, bh, fill=False, ec=LABEL_C, lw=1.6, ls="--"))
    ax[1].imshow(crop(img), interpolation="nearest"); draw_label(ax[1])
    ax[1].set_title("② 확대: 검사기가 그린 색 네모와\n정답 라벨(초록 점선)", loc="left")
    show = np.full((s, s, 3), 255, np.uint8); show[crop(dil) > 0] = (200, 200, 200); show[crop(mk) > 0] = (20, 20, 20)
    ax[2].imshow(show, interpolation="nearest"); draw_label(ax[2])
    ax[2].set_title("③ 지울 픽셀\n검정: 검사기 표시 색\n회색: 지금 방식이 더 지우는 2픽셀", loc="left")
    ax[3].imshow(crop(masked), cmap="gray", vmin=0, vmax=255, interpolation="nearest"); draw_label(ax[3])
    ax[3].set_title("④ 지금 방식으로 지운 뒤\n(지운 자리를 주변 값으로 메움)", loc="left")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values(): sp.set_visible(False)
    lines = [l.split() for l in (LAB / f"{stem}.txt").read_text().splitlines() if l.strip()]
    line = min(lines, key=lambda t: abs(float(t[1]) * W - cx) + abs(float(t[2]) * H - cy))
    fig.text(0.01, 0.02, "라벨 파일 한 줄 = 클래스 번호, 중심 x, 중심 y, 너비, 높이 (사진 너비·높이에 대한 비율).   이 금속구의 줄: "
             + " ".join([line[0]] + [f"{float(v):.4f}" for v in line[1:]]) + f"   →  중심 ({cx:.0f}, {cy:.0f}) 픽셀, 크기 {bw:.0f}×{bh:.0f} 픽셀 (소수 넷째 자리까지 표시)",
             fontsize=9, color=INK2)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.80, bottom=0.08, wspace=0.12)
    out = FIG / "06_raw_walkthrough.png"; plt.savefig(out, dpi=130); plt.close(); print("saved", out)

def image_types(raw):
    items = [("001_20200623_123232(3)", "막대 3개 시험편 (3호기, 6월)"), ("002_20200623_123051(1)", "막대 3개 시험편 (2호기, 6월)"),
             ("002_20200905_083410(5)", "막대 1개 시험편 (2호기, 9월)"), ("001_20200814_083051(9)", "막대 1개 시험편 (3호기, 8월)"),
             ("002_20200727_110524(7)", "위치 이상 사진: 제품이 잘리고\n왼쪽 끝에 노란 선 (1호기 7/27)"),
             ("001_20200706_163134(3)", "검사기 표시가 2개뿐인 사진\n(맨 위 금속구에 표시 없음, 라벨 없음)"),
             ("002_20200623_123055(5)", "라벨이 하나 빠진 사진\n(가운데 금속구, 초록 원 = 정답 라벨)"),
             ("001_20200624_083202(6)", "라벨은 있는데 검사기 표시가\n없는 금속구 (맨 위, 초록 원)")]
    fig, ax = plt.subplots(2, 4, figsize=(15, 6.6))
    for a, (stem, title) in zip(ax.ravel(), items):
        r = raw[(raw.stem == stem) & raw.first_copy].iloc[0]; img = rgb(RAW / r.file); H, W = img.shape[:2]
        a.imshow(img)
        if "초록" in title:
            for bx, by, bw, bh in boxes(stem, W, H): a.add_patch(patches.Circle((bx, by), 9, fill=False, ec=LABEL_C, lw=1.4))
        a.set_title(title, loc="left", fontsize=9.5); a.set_xticks([]); a.set_yticks([])
        a.text(3, H - 6, f"{stem}  {W}×{H}", fontsize=7, color="#0b0b0b", va="bottom", bbox=dict(fc="white", ec="none", alpha=.7, pad=1))
    plt.tight_layout(); out = FIG / "06_image_types.png"; plt.savefig(out, dpi=120); plt.close(); print("saved", out)

def time_pattern(raw, summ):
    u = raw[raw.first_copy].copy()
    u["kind"] = np.select([u.mark_type == "선", u.mark_boxes == 3, u.mark_boxes == 1], ["위치 이상 (노란 선)", "막대 3개 시험편", "막대 1개 시험편"], "기타 (표시 2개)")
    fig = plt.figure(figsize=(15, 7.2)); gs = fig.add_gridspec(3, 2, width_ratios=[1, 2.4], hspace=0.45, wspace=0.12)
    a = fig.add_subplot(gs[:, 0]); h = u.dt.dt.hour.value_counts().reindex(range(24), fill_value=0)
    a.bar(h.index, h.values, width=0.8, color=BLUE)
    a.set_xticks(range(0, 24, 2)); a.set_xlabel("촬영 시각 (시)"); a.set_ylabel("사진 수"); a.grid(axis="y", color="#e6e5e0", lw=0.6); a.set_axisbelow(True)
    t = summ["time"]
    a.set_title(f"① 하루 중 촬영 시각 (서로 다른 사진 {len(u):,}장)\n0·4·8·12·16·20시 15~50분에 {t['routine_share'] * 100:.1f}%", loc="left")
    kinds = [("막대 3개 시험편", BLUE), ("막대 1개 시험편", ORANGE), ("위치 이상 (노란 선)", AQUA), ("기타 (표시 2개)", GRAY)]
    days = pd.date_range(u.dt.min().normalize(), u.dt.max().normalize(), freq="D")
    for i, m in enumerate(MACHINES):
        b = fig.add_subplot(gs[i, 1]); g = u[u.machine == m]
        tab = pd.crosstab(g.dt.dt.normalize(), g.kind).reindex(days, fill_value=0)
        bottom = np.zeros(len(days))
        for k, c in kinds:
            v = tab[k].values if k in tab else np.zeros(len(days))
            b.bar(days, v, bottom=bottom, width=0.8, color=c, label=k, edgecolor="white", linewidth=0.3); bottom += v
        b.set_ylabel(m, rotation=0, ha="right", va="center", fontsize=10, color="#0b0b0b"); b.grid(axis="y", color="#e6e5e0", lw=0.6); b.set_axisbelow(True)
        b.set_ylim(0, max(1, bottom.max()) * 1.1)
        if i == 0:
            b.set_title("② 날짜별 사진 수 (호기별, 시험편 종류별)", loc="left")
            b.legend(ncol=4, fontsize=8.5, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.02))
        if i < 2: b.set_xticklabels([])
    out = FIG / "06_time_pattern.png"; plt.savefig(out, dpi=120, bbox_inches="tight"); plt.close(); print("saved", out)

def object_stats(lb):
    cols = [("size", "① 정답 라벨 크기 (긴 변, 픽셀)"), ("contrast", "② 대비: 주변 밝기 - 금속구 중심 밝기\n(회색 단계, 클수록 잘 보임)"),
            ("edge_dist", "③ 제품 가장자리까지 거리 (픽셀)")]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for a, (c, title) in zip(ax, cols):
        data = [lb[lb.machine == m][c].dropna().values for m in MACHINES]
        bp = a.boxplot(data, widths=0.5, patch_artist=True, showfliers=True,
                       medianprops=dict(color="#0b0b0b", lw=1.5), flierprops=dict(marker="o", ms=2.5, mfc=GRAY, mec="none"),
                       boxprops=dict(fc="#cde2fb", ec=BLUE, lw=1), whiskerprops=dict(color=BLUE, lw=1), capprops=dict(color=BLUE, lw=1))
        a.set_xticks(range(1, 4)); a.set_xticklabels([f"{m}\n(n={len(d)})" for m, d in zip(MACHINES, data)])
        for i, d in enumerate(data, 1): a.text(i + 0.3, np.median(d), f"{np.median(d):.0f}", va="center", fontsize=8.5, color="#0b0b0b")
        a.set_title(title, loc="left"); a.grid(axis="y", color="#e6e5e0", lw=0.6); a.set_axisbelow(True)
    plt.tight_layout(); out = FIG / "06_object_stats.png"; plt.savefig(out, dpi=130); plt.close(); print("saved", out)

def main():
    FIG.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(AUD / "raw_images.csv", encoding="utf-8-sig", parse_dates=["dt"])
    lb = pd.read_csv(AUD / "label_boxes.csv", encoding="utf-8-sig")
    summ = json.loads((AUD / "summary.json").read_text(encoding="utf-8"))
    walkthrough(raw); image_types(raw); time_pattern(raw, summ); object_stats(lb)

if __name__ == "__main__":
    main()
