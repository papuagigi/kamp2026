#!/usr/bin/env python3
"""
검사기 표시를 지우는 방식 비교 (라벨 사진 500장).
표시 밑의 진짜 픽셀은 원본에 남아 있지 않아서, 메운 값이 맞는지 바로 잴 수 없다. 그래서 두 가지로 잰다.

1) 옮겨 놓기 시험: 사진의 표시 모양(연결된 덩어리 하나씩)을 표시가 없는 제품 위로 옮겨 놓고,
   그 자리를 각 방식으로 지우고 메운 뒤 원래 픽셀과 비교한다. 정답을 아는 자리에서 메우기 정확도를 잰다.
2) 제자리 측정: 실제 표시 자리에서 v1의 팽창이 지운 둘레 2픽셀은 원본에 진짜 값이 있다.
   라벨 박스 안에서 v1이 이 진짜 픽셀을 얼마나 바꿨는지, 금속구 대비가 얼마나 달라졌는지 잰다.

비교하는 방식
- v1        : RGB 차이 40으로 표시를 고르고 5×5 팽창 후 텔레아 인페인팅 (xray_prepare.mask_image와 같음)
- 팔레트+T  : 팔레트 244~255번만 지우고 텔레아 인페인팅 (docs/06 8-3 제안)
- 팔레트+NS : 팔레트 244~255번만 지우고 나비에-스토크스 인페인팅
- 단색      : 팔레트 244~255번을 둘레 픽셀의 중앙값 하나로 칠함 (권하지 않는 방식, 비교용)
출력: reports/mask_compare/shift_test.csv, boxes.csv, summary.json, 화면에 요약
사용: .venv/bin/python scripts/xray_mask_compare.py
"""
import json
from pathlib import Path
import numpy as np, cv2, pandas as pd
from PIL import Image
from xray_prepare import RAW, LAB

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "mask_compare"
AUD = ROOT / "reports" / "data_audit"
K5, K9 = np.ones((5, 5), np.uint8), np.ones((9, 9), np.uint8)
SHIFTS = [(dy, dx) for d in range(20, 121, 10) for dy, dx in ((0, d), (0, -d), (d, 0), (-d, 0), (d, d), (-d, -d), (d, -d), (-d, d))]

def fill(gray, m, how):
    m8 = m.astype(np.uint8)
    if how == "v1": return cv2.inpaint(gray, cv2.dilate(m8, K5), 3, cv2.INPAINT_TELEA)
    if how == "pal_telea": return cv2.inpaint(gray, m8, 3, cv2.INPAINT_TELEA)
    if how == "pal_ns": return cv2.inpaint(gray, m8, 3, cv2.INPAINT_NS)
    if how == "const":
        out = gray.copy(); n, lab = cv2.connectedComponents(m8)
        for k in range(1, n):
            comp = lab == k; ring = cv2.dilate(comp.astype(np.uint8), K5).astype(bool) & ~m
            out[comp] = np.uint8(np.median(gray[ring])) if ring.any() else out[comp]
        return out
    raise ValueError(how)

METHODS = ["v1", "pal_telea", "pal_ns", "const"]

def contrast(img, bx, by, exclude=None):
    """금속구 대비 = 둘레(반경 5~9픽셀) 중앙값 - 중심 3×3 최솟값. xray_data_audit.py와 같은 정의."""
    H, W = img.shape; yy, xx = np.mgrid[0:H, 0:W]; rr = np.hypot(xx - bx, yy - by)
    ann = (rr >= 5) & (rr <= 9)
    if exclude is not None: ann &= ~exclude
    cen = img[max(0, by - 1):by + 2, max(0, bx - 1):bx + 2]
    core = np.ones_like(cen, bool) if exclude is None else ~exclude[max(0, by - 1):by + 2, max(0, bx - 1):bx + 2]
    if not ann.any() or not core.any(): return np.nan
    return float(np.median(img[ann])) - float(cen[core].min())

def main():
    raw = {}
    for p in sorted(RAW.rglob("*.bmp")): raw.setdefault(p.stem, p)
    lb = pd.read_csv(AUD / "label_boxes.csv")                      # 금속구 위치(ball_x, ball_y)는 원본 점검 결과를 쓴다
    stems = sorted(p.stem for p in LAB.glob("*.txt"))
    shift_rows, shift_res, box_rows = [], [], []
    for s in stems:
        idx = np.asarray(Image.open(raw[s])); H, W = idx.shape
        gray = idx.astype(np.uint8); M = idx >= 244

        # 1) 옮겨 놓기 시험
        busy = cv2.dilate(M.astype(np.uint8), K9).astype(bool)   # 원래 표시와 그 둘레는 피한다
        placed = np.zeros_like(M)
        n, lab = cv2.connectedComponents(M.astype(np.uint8))
        for k in range(1, n):
            ys, xs = np.nonzero(lab == k)
            for dy, dx in SHIFTS:
                y2, x2 = ys + dy, xs + dx
                if y2.min() < 4 or x2.min() < 4 or y2.max() >= H - 4 or x2.max() >= W - 4: continue
                comp = np.zeros_like(M); comp[y2, x2] = True
                near = cv2.dilate(comp.astype(np.uint8), K9).astype(bool)
                if (near & (busy | placed)).any(): continue
                if np.median(gray[comp]) >= 200: continue          # 제품 밖 빈 곳(210)은 메우기가 너무 쉬워서 뺀다
                placed |= comp; shift_rows.append(dict(stem=s, comp=k, px=len(ys), dy=dy, dx=dx)); break
        if placed.any():
            ring = cv2.dilate(placed.astype(np.uint8), K5).astype(bool) & ~placed
            outs = {h: fill(gray, placed, h) for h in METHODS}
            n2, lab2 = cv2.connectedComponents(placed.astype(np.uint8))
            ring_lab = cv2.dilate(lab2.astype(np.uint16), K5)       # 둘레 픽셀이 어느 덩어리 것인지 (덩어리끼리는 9픽셀 넘게 떨어져 있음)
            for k2 in range(1, n2):
                comp = lab2 == k2; rg = ring & (ring_lab == k2)
                row = dict(stem=s, px=int(comp.sum()), ring_px=int(rg.sum()), truth_mean=float(gray[comp].mean()),
                           local_std=float(gray[rg].std()))
                for h in METHODS:
                    row[f"err_{h}"] = float(np.abs(outs[h][comp].astype(int) - gray[comp]).mean())
                row["ring_err_v1"] = float(np.abs(outs["v1"][rg].astype(int) - gray[rg]).mean())
                shift_res.append(row)

        # 2) 제자리 측정: v1이 라벨 박스 안의 진짜 픽셀을 얼마나 바꿨나
        v1 = fill(gray, M, "v1"); pt = fill(gray, M, "pal_telea")
        dil_real = cv2.dilate(M.astype(np.uint8), K5).astype(bool); ring_real = dil_real & ~M
        for b in lb[lb.stem == s].itertuples():                   # 박스 경계는 xray_data_audit.py와 같게 (내림·올림)
            x0, x1 = max(0, int(b.cx - b.w / 2)), min(W, int(np.ceil(b.cx + b.w / 2)))
            y0, y1 = max(0, int(b.cy - b.h / 2)), min(H, int(np.ceil(b.cy + b.h / 2)))
            box = np.zeros_like(M); box[y0:y1, x0:x1] = True
            ch = ring_real & box
            c_raw = contrast(gray, b.ball_x, b.ball_y, exclude=M)
            box_rows.append(dict(stem=s, machine=b.machine, box_px=int(box.sum()), mark_px_in_box=int((M & box).sum()),
                                 v1_erased_frac=float(dil_real[box].mean()), pal_erased_frac=float(M[box].mean()),
                                 v1_changed_real_px=int(ch.sum()),
                                 v1_change_mean=float(np.abs(v1[ch].astype(int) - gray[ch]).mean()) if ch.any() else 0.0,
                                 contrast_raw=c_raw, contrast_v1=contrast(v1, b.ball_x, b.ball_y),
                                 contrast_pal=contrast(pt, b.ball_x, b.ball_y)))

    sh = pd.DataFrame(shift_res); bx = pd.DataFrame(box_rows)
    bx["dc_v1"] = (bx.contrast_v1 - bx.contrast_raw).abs(); bx["dc_pal"] = (bx.contrast_pal - bx.contrast_raw).abs()
    OUT.mkdir(parents=True, exist_ok=True)
    sh.to_csv(OUT / "shift_test.csv", index=False, encoding="utf-8-sig"); bx.to_csv(OUT / "boxes.csv", index=False, encoding="utf-8-sig")

    q = lambda s: dict(median=round(float(s.median()), 2), p90=round(float(s.quantile(0.9)), 2), mean=round(float(s.mean()), 2))
    summ = dict(
        shift_test=dict(n_components=len(sh), n_images=int(sh.stem.nunique()), px_total=int(sh.px.sum()),
                        local_std=q(sh.local_std), truth_mean=q(sh.truth_mean),
                        fill_err={h: q(sh[f"err_{h}"]) for h in METHODS}, v1_ring_err=q(sh.ring_err_v1),
                        v1_ring_px=int(sh.ring_px.sum())),
        in_place=dict(n_boxes=len(bx), boxes_with_mark_px=int((bx.mark_px_in_box > 0).sum()),
                      boxes_erased_over_10pct=dict(v1=int((bx.v1_erased_frac > 0.1).sum()), pal=int((bx.pal_erased_frac > 0.1).sum())),
                      boxes_v1_changed=int((bx.v1_changed_real_px > 0).sum()), v1_changed_px_total=int(bx.v1_changed_real_px.sum()),
                      v1_changed_px_per_box=q(bx.v1_changed_real_px[bx.v1_changed_real_px > 0]),
                      v1_change_mean=q(bx.v1_change_mean[bx.v1_changed_real_px > 0]),
                      contrast_raw=q(bx.contrast_raw), contrast_change_v1=q(bx.dc_v1), contrast_change_pal=q(bx.dc_pal),
                      boxes_contrast_change_ge2=dict(v1=int((bx.dc_v1 >= 2).sum()), pal=int((bx.dc_pal >= 2).sum()))))
    (OUT / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")

    t = summ["shift_test"]; ip = summ["in_place"]
    print(f"[1] 옮겨 놓기 시험: 사진 {t['n_images']}장, 표시 덩어리 {t['n_components']}개, 지운 픽셀 {t['px_total']:,}개")
    print(f"    그 자리 원래 밝기 중앙값 {t['truth_mean']['median']}, 둘레 픽셀의 흔들림(표준편차) 중앙값 {t['local_std']['median']}")
    print("    지운 자리의 평균 오차(밝기 단계, 덩어리별 중앙값 / 90% 지점):")
    for h in METHODS: print(f"      {h:10s} {t['fill_err'][h]['median']:6.2f} / {t['fill_err'][h]['p90']:6.2f}")
    print(f"    v1이 팽창으로 더 지운 둘레 픽셀 {t['v1_ring_px']:,}개의 평균 오차: 중앙값 {t['v1_ring_err']['median']} / 90% 지점 {t['v1_ring_err']['p90']} (팔레트 방식은 이 픽셀을 건드리지 않아 0)")
    print(f"[2] 제자리 측정: 라벨 박스 {ip['n_boxes']}개")
    print(f"    표시 픽셀 자체가 박스 안에 들어온 박스 {ip['boxes_with_mark_px']}개 (어느 방식이든 이 픽셀은 추정값으로 메워야 함)")
    print(f"    넓이의 10% 넘게 지워지는 박스: v1 {ip['boxes_erased_over_10pct']['v1']}개, 팔레트 {ip['boxes_erased_over_10pct']['pal']}개")
    print(f"    v1이 박스 안의 진짜 픽셀을 바꾼 박스 {ip['boxes_v1_changed']}개, 바뀐 진짜 픽셀 합계 {ip['v1_changed_px_total']:,}개")
    print(f"      박스당 바뀐 픽셀 중앙값 {ip['v1_changed_px_per_box']['median']}, 바뀐 양의 평균 중앙값 {ip['v1_change_mean']['median']} / 90% 지점 {ip['v1_change_mean']['p90']}")
    print(f"    금속구 대비(원본 기준 중앙값 {ip['contrast_raw']['median']})가 원본과 달라진 정도: v1 중앙값 {ip['contrast_change_v1']['median']} / 90% 지점 {ip['contrast_change_v1']['p90']}, "
          f"팔레트 중앙값 {ip['contrast_change_pal']['median']} / 90% 지점 {ip['contrast_change_pal']['p90']}")
    print(f"    대비가 2단계 이상 달라진 박스: v1 {ip['boxes_contrast_change_ge2']['v1']}개, 팔레트 {ip['boxes_contrast_change_ge2']['pal']}개")

if __name__ == "__main__":
    main()
