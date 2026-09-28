#!/usr/bin/env python3
"""
예측 CSV(stem, cx, cy, w, h, score)를 정답과 맞춰 채점한다.
- 박스 매칭: 중심 거리 8px 이내 또는 IoU 0.3 이상 (이물이 10~15px라 IoU 0.5는 가혹)
- 점수 문턱값을 훑어 정밀도/재현율/F1, 최고 F1, AP(PR 곡선 면적)
- 조건별 재현율: 시편 여부, 호기, 이미지 크기, 이물 크기, 제품 가장자리 거리, 대비
사용: .venv/bin/python scripts/xray_eval.py reports/preds_v1_yolov8n_640_val.csv --split val [--thr 0.25]
      일부 사진만: --split train,val,test --stems 사진이름목록.txt --tag 이름  (결과: <예측파일>_<이름>_eval.json)
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, cv2
ROOT = Path(__file__).resolve().parents[1]; DATA = ROOT / "data" / "xray_v1"; REPORTS = ROOT / "reports"

def product_mask(gray):
    bg = np.median(np.r_[gray[:5].ravel(), gray[-5:].ravel(), gray[:, :5].ravel(), gray[:, -5:].ravel()])
    m = (gray < bg - 25).astype(np.uint8); m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1: return m
    return (lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)

def load_gt(split, stems=None):
    man = pd.read_csv(DATA / "manifest.csv", encoding="utf-8-sig"); man = man[man.split.isin(split.split(","))]
    if stems is not None: man = man[man.stem.isin(stems)]
    gts = []
    for r in man.itertuples():
        W, H = r.width, r.height
        gray = cv2.imread(str(DATA / "images" / r.split / f"{r.stem}.png"), cv2.IMREAD_GRAYSCALE)
        pm = product_mask(gray); dist = cv2.distanceTransform(pm, cv2.DIST_L2, 3)
        for l in (DATA / "labels" / r.split / f"{r.stem}.txt").read_text().splitlines():
            if not l.strip(): continue
            _, cx, cy, w, h = map(float, l.split()); cx, cy, w, h = cx * W, cy * H, w * W, h * H
            xi, yi = int(min(max(cx, 0), W - 1)), int(min(max(cy, 0), H - 1))
            inner = gray[max(0, yi - 2):yi + 3, max(0, xi - 2):xi + 3].min(); ring = np.median(gray[max(0, yi - 10):yi + 11, max(0, xi - 10):xi + 11])
            gts.append(dict(stem=r.stem, cx=cx, cy=cy, w=w, h=h, size=max(w, h), edge=float(dist[yi, xi]), contrast=float(ring) - float(inner),
                            is_testpiece=bool(r.is_testpiece), machine=r.machine, width=W, n_boxes=r.n_boxes))
    return pd.DataFrame(gts), man

def iou(a, b):
    ax0, ay0, ax1, ay1 = a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2
    bx0, by0, bx1, by1 = b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2
    iw, ih = max(0, min(ax1, bx1) - max(ax0, bx0)), max(0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih; return inter / (a[2]*a[3] + b[2]*b[3] - inter + 1e-9)

def match(preds, gts, dist_thr=8.0, iou_thr=0.3):
    """예측(점수 내림차순)을 정답에 하나씩 매칭. 반환: 각 예측의 매칭된 정답 index(-1이면 오탐), 각 정답의 매칭 여부와 점수"""
    gt_hit = -np.ones(len(gts)); pred_hit = []
    gt_by_stem = {}
    for i, g in enumerate(gts.itertuples()): gt_by_stem.setdefault(g.stem, []).append(i)
    preds = preds.sort_values("score", ascending=False)
    for p in preds.itertuples():
        best, best_d = -1, 1e9
        for i in gt_by_stem.get(p.stem, []):
            if gt_hit[i] >= 0: continue
            g = gts.iloc[i]; d = ((p.cx - g.cx)**2 + (p.cy - g.cy)**2) ** 0.5
            if (d <= dist_thr or iou((p.cx, p.cy, p.w, p.h), (g.cx, g.cy, g.w, g.h)) >= iou_thr) and d < best_d: best, best_d = i, d
        pred_hit.append(best)
        if best >= 0: gt_hit[best] = p.score
    preds = preds.assign(hit=pred_hit); return preds, gt_hit

def evaluate(pred_csv, split, thr=None, stems=None):
    gts, man = load_gt(split, stems); preds = pd.read_csv(pred_csv); preds = preds[preds.stem.isin(set(man.stem))]
    preds, gt_hit = match(preds, gts)
    scores = np.sort(preds.score.unique())[::-1]
    rows = []
    for s in scores:
        sel = preds[preds.score >= s]; tp = (sel.hit >= 0).sum(); fp = (sel.hit < 0).sum(); fn = len(gts) - tp
        p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1); rows.append((s, p, r, 2*p*r/max(p+r, 1e-9), int(fp)))
    curve = pd.DataFrame(rows, columns=["thr", "precision", "recall", "f1", "fp"])
    ap = float(np.trapz(np.maximum.accumulate(curve.precision.values[::-1])[::-1], curve.recall.values)) if len(curve) > 1 else 0.0
    best = curve.iloc[curve.f1.idxmax()]
    thr = float(best.thr) if thr is None else thr
    sel = preds[preds.score >= thr]; tp = (sel.hit >= 0).sum(); fp = (sel.hit < 0).sum()
    gts["detected"] = gt_hit >= thr
    # 재현율 99% 를 만족하는 최저 문턱값 (안전 기준)
    r99 = curve[curve.recall >= 0.99]; thr99 = float(r99.thr.max()) if len(r99) else None; fp99 = int(r99[r99.thr == thr99].fp.iloc[0]) if len(r99) else None
    res = dict(pred_csv=str(pred_csv), split=split, n_images=int(man.shape[0]), n_gt=int(len(gts)), thr=thr,
               precision=float(tp / max(tp + fp, 1)), recall=float(tp / max(len(gts), 1)), f1=float(2*tp / max(2*tp + fp + (len(gts) - tp), 1)),
               fp_per_image=float(fp / max(man.shape[0], 1)), ap=ap, best_f1=float(best.f1), thr_recall99=thr99, fp_at_recall99=fp99,
               images_fully_detected=float(gts.groupby("stem").detected.all().mean()))
    cond = {}
    gts["size_bin"] = pd.cut(gts["size"], [0, 8, 11, 14, 100], labels=["~8px", "8~11px", "11~14px", "14px~"])
    gts["edge_bin"] = pd.cut(gts["edge"], [-1, 20, 40, 60, 1000], labels=["가장자리20px이내", "20~40px", "40~60px", "60px~"])
    gts["contrast_bin"] = pd.cut(gts["contrast"], [-1000, 35, 45, 55, 1000], labels=["대비~35", "35~45", "45~55", "55~"])
    for c in ["is_testpiece", "machine", "width", "n_boxes", "size_bin", "edge_bin", "contrast_bin"]:
        g = gts.groupby(c, observed=True).detected.agg(["mean", "size"]); cond[c] = {str(k): (round(float(v["mean"]), 3), int(v["size"])) for k, v in g.iterrows()}
    res["recall_by_condition"] = cond
    return res, curve, gts

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("pred_csv"); ap.add_argument("--split", required=True); ap.add_argument("--thr", type=float, default=None)
    ap.add_argument("--stems", default=None, help="채점할 사진 이름 목록 파일 (한 줄에 하나). 없으면 split 전체")
    ap.add_argument("--tag", default="", help="결과 파일 이름에 붙일 말")
    a = ap.parse_args(); stems = set(Path(a.stems).read_text().split()) if a.stems else None
    res, curve, gts = evaluate(a.pred_csv, a.split, a.thr, stems)
    out = REPORTS / (Path(a.pred_csv).stem + (f"_{a.tag}" if a.tag else "") + "_eval.json"); out.write_text(json.dumps(res, ensure_ascii=False, indent=2)); 
    print(json.dumps({k: v for k, v in res.items() if k != "recall_by_condition"}, ensure_ascii=False))
    for c, d in res["recall_by_condition"].items(): print(f"  {c}: {d}")
    print("saved", out)

if __name__ == "__main__":
    main()
