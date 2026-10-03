#!/usr/bin/env python3
"""
예측 CSV(stem, cx, cy, w, h, score)를 정답과 맞춰 채점한다.
- 박스 매칭: 중심 거리 8px 이내 또는 IoU 0.3 이상 (이물이 10~15px라 IoU 0.5는 가혹)
- AP: 자체 매칭의 보간 PR 계단 면적(all-points). COCO AP와 구분한다.
- 빈 예측은 미탐으로 집계하고, 정답 없는 세트의 AP/재현율은 null(해당 없음).
사용: .venv/bin/python scripts/xray_eval.py reports/preds_v2_yolov8n_640_val.csv --split val [--thr 0.25]
      일부 사진만: --split train,val,test --stems 사진이름목록.txt --tag 이름  (결과: <예측파일>_<이름>_eval.json)
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np, pandas as pd, cv2
from xray_config import ROOT, DATA
REPORTS = ROOT / "reports"
METRIC_VERSION = "custom_ap_v2"
MATCH_POLICIES = {"legacy": (8.0, 0.3, "custom_ap_v2"),
                  "iou50": (None, 0.5, "iou50_v1"),
                  "iou75": (None, 0.75, "iou75_v1")}
GT_COLUMNS = ["stem", "cx", "cy", "w", "h", "size", "edge", "contrast", "card_type", "machine", "width", "n_boxes"]
PRED_COLUMNS = ["stem", "cx", "cy", "w", "h", "score"]

def product_mask(gray):
    bg = np.median(np.r_[gray[:5].ravel(), gray[-5:].ravel(), gray[:, :5].ravel(), gray[:, -5:].ravel()])
    m = (gray < bg - 25).astype(np.uint8); m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1: return m
    return (lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)

def load_gt(split, stems=None):
    man = pd.read_csv(DATA / "manifest.csv", encoding="utf-8-sig"); man = man[man.split.isin(split.split(","))]
    if stems is not None: man = man[man.stem.isin(stems)]
    if man.empty:
        raise ValueError("평가할 사진이 없습니다. split과 사진 목록을 확인하세요")
    gts = []
    for r in man.itertuples():
        W, H = r.width, r.height
        gray = cv2.imread(str(DATA / "images" / r.split / f"{r.stem}.png"), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f"평가 이미지를 읽지 못했습니다: {r.stem}")
        pm = product_mask(gray); dist = cv2.distanceTransform(pm, cv2.DIST_L2, 3)
        for l in (DATA / "labels" / r.split / f"{r.stem}.txt").read_text().splitlines():
            if not l.strip(): continue
            _, cx, cy, w, h = map(float, l.split()); cx, cy, w, h = cx * W, cy * H, w * W, h * H
            xi, yi = int(min(max(cx, 0), W - 1)), int(min(max(cy, 0), H - 1))
            inner = gray[max(0, yi - 2):yi + 3, max(0, xi - 2):xi + 3].min(); ring = np.median(gray[max(0, yi - 10):yi + 11, max(0, xi - 10):xi + 11])
            gts.append(dict(stem=r.stem, cx=cx, cy=cy, w=w, h=h, size=max(w, h), edge=float(dist[yi, xi]), contrast=float(ring) - float(inner),
                            card_type=getattr(r, "card_type", "막대3" if r.n_boxes >= 2 else "막대1"), machine=r.machine, width=W, n_boxes=r.n_boxes,
                            **({"is_testpiece": bool(r.is_testpiece)} if hasattr(r, "is_testpiece") else {})))
    return pd.DataFrame(gts) if gts else pd.DataFrame(columns=GT_COLUMNS), man

def iou(a, b):
    ax0, ay0, ax1, ay1 = a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2
    bx0, by0, bx1, by1 = b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2
    iw, ih = max(0, min(ax1, bx1) - max(ax0, bx0)), max(0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih; union = a[2]*a[3] + b[2]*b[3] - inter
    return inter / union if union > 0 else 0.0

def match(preds, gts, dist_thr=8.0, iou_thr=0.3, iou_only=False):
    """예측(점수 내림차순)을 정답에 하나씩 매칭. 반환: 각 예측의 매칭된 정답 index(-1이면 오탐), 각 정답의 매칭 여부와 점수"""
    gt_hit = np.full(len(gts), np.nan); pred_hit = []
    gt_by_stem = {}
    for i, g in enumerate(gts.itertuples()): gt_by_stem.setdefault(g.stem, []).append(i)
    preds = preds.sort_values("score", ascending=False, kind="stable")
    for p in preds.itertuples():
        best, best_d, best_iou = -1, 1e9, -1.0
        for i in gt_by_stem.get(p.stem, []):
            if np.isfinite(gt_hit[i]): continue
            g = gts.iloc[i]; d = ((p.cx - g.cx)**2 + (p.cy - g.cy)**2) ** 0.5
            overlap = iou((p.cx, p.cy, p.w, p.h), (g.cx, g.cy, g.w, g.h))
            if iou_only:
                if overlap >= iou_thr and overlap > best_iou:
                    best, best_iou = i, overlap
            elif (d <= dist_thr or overlap >= iou_thr) and d < best_d:
                best, best_d = i, d
        pred_hit.append(best)
        if best >= 0: gt_hit[best] = p.score
    preds = preds.assign(hit=pred_hit); return preds, gt_hit

def interpolated_ap(curve, n_gt):
    """관측 재현율의 증가분 × 오른쪽 보간 정밀도를 합한다. 동점은 한 임계값."""
    if n_gt == 0:
        return None
    if curve.empty:
        return 0.0
    recall = np.r_[0.0, curve.recall.to_numpy(dtype=float), 1.0]
    precision = np.r_[0.0, curve.precision.to_numpy(dtype=float), 0.0]
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.sum(np.diff(recall) * precision[1:]))


def evaluate(pred_csv, split, thr=None, stems=None, matching="legacy"):
    if matching not in MATCH_POLICIES:
        raise ValueError(f"Unknown matching policy: {matching}")
    if matching != "legacy" and "test" in split.split(",") and thr is None:
        raise ValueError("테스트 평가에는 검증에서 고정한 탐지 임계값 --thr이 필요합니다")
    dist_thr, iou_thr, metric_version = MATCH_POLICIES[matching]
    if thr is not None and not np.isfinite(thr):
        raise ValueError("임계값은 유한한 수여야 합니다")
    gts, man = load_gt(split, stems)
    if man.empty:
        raise ValueError("평가할 사진이 없습니다")
    # 이전 호출자나 정답 없는 대조 세트도 열 형식을 유지한다.
    gts = gts.copy()
    if gts.empty:
        gts = gts.reindex(columns=GT_COLUMNS)
    try:
        preds = pd.read_csv(pred_csv, dtype={"stem": str})
    except pd.errors.EmptyDataError as exc:
        raise ValueError("예측 CSV에 헤더가 없습니다. 검출이 없어도 열 이름은 저장해야 합니다") from exc
    if not set(PRED_COLUMNS).issubset(preds.columns):
        raise ValueError(f"예측 CSV에 필요한 열: {PRED_COLUMNS}")
    numeric = preds[PRED_COLUMNS[1:]].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all() or preds.stem.isna().any():
        raise ValueError("예측 CSV에 누락 또는 비유한 값이 있습니다")
    if (numeric[["w", "h"]] < 0).any().any():
        raise ValueError("예측 박스의 폭과 높이는 음수일 수 없습니다")
    preds[PRED_COLUMNS[1:]] = numeric
    preds = preds[preds.stem.isin(set(man.stem))].copy()
    preds, gt_hit = match(preds, gts, dist_thr=dist_thr, iou_thr=iou_thr, iou_only=matching != "legacy")
    scores = np.sort(preds.score.unique())[::-1]
    rows = []
    # Score-tied predictions enter together. Cumulative counts avoid scanning the
    # entire table again for each distinct score in dense detector outputs.
    score_groups = preds.assign(tp=(preds.hit >= 0).astype(int), fp=(preds.hit < 0).astype(int)).groupby("score")[["tp", "fp"]].sum().sort_index(ascending=False).cumsum()
    for s, counts in score_groups.iterrows():
        tp, fp = int(counts.tp), int(counts.fp); fn = len(gts) - tp
        p = tp / max(tp + fp, 1); r = tp / (tp + fn) if len(gts) else np.nan
        f1 = 2 * tp / (2 * tp + fp + fn) if len(gts) else np.nan
        rows.append((s, p, r, f1, int(fp)))
    curve = pd.DataFrame(rows, columns=["thr", "precision", "recall", "f1", "fp"])
    ap = interpolated_ap(curve, len(gts))
    best = curve.iloc[curve.f1.idxmax()] if len(gts) and not curve.empty else None
    threshold_source = "provided" if thr is not None else "best_f1_on_" + split
    if thr is None:
        thr = float(best.thr) if best is not None else None
        if best is None:
            threshold_source = "unavailable_no_predictions" if preds.empty else "unavailable_no_ground_truth"
    sel = preds[preds.score >= thr] if thr is not None else preds
    tp = int((sel.hit >= 0).sum()); fp = int((sel.hit < 0).sum()); fn = len(gts) - tp
    gts["detected"] = np.isfinite(gt_hit) & (gt_hit >= thr) if thr is not None else np.isfinite(gt_hit)
    # 재현율 99%를 만족하는 가장 높은 임계값. 운영 임계값과 별개의 진단값이다.
    r99 = curve[curve.recall >= 0.99]; thr99 = float(r99.thr.max()) if len(r99) else None; fp99 = int(r99[r99.thr == thr99].fp.iloc[0]) if len(r99) else None
    positive_stems = set(gts.stem)
    negative_stems = set(man.stem) - positive_stems
    negative_with_fp = len(negative_stems & set(sel.stem))
    split_file = DATA / "split.csv"
    res = dict(pred_csv=str(pred_csv), split=split, n_images=int(man.shape[0]), n_gt=int(len(gts)), thr=thr,
               precision=float(tp / (tp + fp)) if tp + fp else None,
               recall=float(tp / len(gts)) if len(gts) else None,
               f1=float(2 * tp / (2 * tp + fp + fn)) if len(gts) else None,
               fp_per_image=float(fp / len(man)), ap=ap,
               best_f1=float(best.f1) if best is not None else (0.0 if len(gts) else None),
               thr_recall99=thr99, fp_at_recall99=fp99,
               images_fully_detected=float(gts.groupby("stem").detected.all().mean()) if len(gts) else None,
               metric_version=metric_version, ap_definition="all_points_interpolated_custom_matching" if matching == "legacy" else "all_points_interpolated_iou_matching_not_coco",
               matching={"center_distance_px": dist_thr, "iou": iou_thr, "operator": "or" if matching == "legacy" else "iou_only"},
               data_version=DATA.name,
               split_csv_md5=hashlib.md5(split_file.read_bytes()).hexdigest() if split_file.exists() else None,
               threshold_source=threshold_source, tp=tp, fp=fp, fn=fn, n_predictions=int(len(preds)),
               n_selected_predictions=int(len(sel)),
               n_zero_area_predictions=int(((preds.w == 0) | (preds.h == 0)).sum()),
               n_negative_images=len(negative_stems),
               negative_images_with_fp=negative_with_fp,
               image_false_alarm_rate=negative_with_fp / len(negative_stems) if negative_stems else None,
               evaluation_status="no_ground_truth" if not len(gts) else "no_predictions" if preds.empty else "ok")
    cond = {}
    gts["size_bin"] = pd.cut(gts["size"].astype(float), [0, 8, 11, 14, 100], labels=["~8px", "8~11px", "11~14px", "14px~"])
    gts["edge_bin"] = pd.cut(gts["edge"].astype(float), [-1, 20, 40, 60, 1000], labels=["가장자리20px이내", "20~40px", "40~60px", "60px~"])
    gts["contrast_bin"] = pd.cut(gts["contrast"].astype(float), [-1000, 35, 45, 55, 1000], labels=["대비~35", "35~45", "45~55", "55~"])
    for c in [c for c in ["card_type", "machine", "width", "n_boxes", "size_bin", "edge_bin", "contrast_bin"] if c in gts]:
        g = gts.groupby(c, observed=True).detected.agg(["mean", "size"]); cond[c] = {str(k): (round(float(v["mean"]), 3), int(v["size"])) for k, v in g.iterrows()}
    res["recall_by_condition"] = cond
    return res, curve, gts

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("pred_csv"); ap.add_argument("--split", required=True); ap.add_argument("--thr", type=float, default=None)
    ap.add_argument("--stems", default=None, help="채점할 사진 이름 목록 파일 (한 줄에 하나). 없으면 split 전체")
    ap.add_argument("--matching", choices=list(MATCH_POLICIES), default="legacy")
    ap.add_argument("--tag", default=None, help="결과 파일 이름에 붙일 말 (기본: 선택한 채점 버전)")
    a = ap.parse_args(); stems = set(Path(a.stems).read_text().split()) if a.stems else None
    res, curve, gts = evaluate(a.pred_csv, a.split, a.thr, stems, a.matching)
    REPORTS.mkdir(parents=True, exist_ok=True)
    tag = MATCH_POLICIES[a.matching][2] if a.tag is None else a.tag
    out = REPORTS / (Path(a.pred_csv).stem + (f"_{tag}" if tag else "") + "_eval.json")
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "recall_by_condition"}, ensure_ascii=False))
    for c, d in res["recall_by_condition"].items(): print(f"  {c}: {d}")
    print("saved", out)

if __name__ == "__main__":
    main()
