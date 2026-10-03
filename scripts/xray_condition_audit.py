"""Describe train/validation conditions with frozen scores; never train or tune on test.

Group boundaries are the official training boxes' quartiles. Shape and background
descriptors are proxies, not verified physical categories. Reuses xray_eval only.
"""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from xray_config import DATA, ROOT, V2_SPLIT_MD5
from xray_eval import evaluate, load_gt

OUT = ROOT / "reports/condition_audit_20261003"
RUNS = ["v2_mix2453_yolo_mps", "v2_mix2453_faster_cuda", "v2_mix2453_rfdetr_mps"]
FEATURES = ["size", "edge", "contrast", "box_aspect_ratio", "background_gray", "background_std"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def features(gt, manifest, split):
    gt = gt.copy()
    gt["box_aspect_ratio"] = np.maximum(gt.w, gt.h) / np.minimum(gt.w, gt.h)
    for stem, rows in gt.groupby("stem"):
        gray = cv2.imread(str(DATA / "images" / split / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        assert gray is not None
        available = np.ones(gray.shape, dtype=bool)
        for g in rows.itertuples():
            x0, x1 = int(np.floor(g.cx - g.w / 2)), int(np.ceil(g.cx + g.w / 2))
            y0, y1 = int(np.floor(g.cy - g.h / 2)), int(np.ceil(g.cy + g.h / 2))
            available[max(0, y0):min(gray.shape[0], y1 + 1), max(0, x0):min(gray.shape[1], x1 + 1)] = False
        for idx, g in rows.iterrows():
            radius = int(np.ceil(max(g.w, g.h))) + 5
            x0, x1 = max(0, int(g.cx) - radius), min(gray.shape[1], int(g.cx) + radius + 1)
            y0, y1 = max(0, int(g.cy) - radius), min(gray.shape[0], int(g.cy) + radius + 1)
            patch = gray[y0:y1, x0:x1][available[y0:y1, x0:x1]]
            assert len(patch)
            gt.loc[idx, "background_gray"] = float(np.median(patch))
            gt.loc[idx, "background_std"] = float(np.std(patch))
    gt["burst_id"] = gt.stem.map(manifest.set_index("stem").burst_id)
    assert gt[FEATURES + ["burst_id"]].notna().all().all()
    return gt


def distribution(gt):
    return {k: {str(q): float(gt[k].quantile(q)) for q in [0, .25, .5, .75, 1]} for k in FEATURES}


def main():
    assert hashlib.md5((DATA / "split.csv").read_bytes()).hexdigest() == V2_SPLIT_MD5
    OUT.mkdir(parents=True, exist_ok=True)
    train, tm = load_gt("train")
    train = features(train, tm, "train")
    cuts = {k: [float(train[k].quantile(.25)), float(train[k].quantile(.75))] for k in FEATURES}
    definitions = {
        "size": "max(TXT box width, height), original-image pixels; not measured physical defect diameter",
        "edge": "center distance inside largest approximate product mask from xray_eval; not guaranteed bar boundary",
        "contrast": "21x21 local median minus 5x5 central minimum, grayscale levels; xray_eval proxy",
        "box_aspect_ratio": "long side / short side of official TXT box; not segmented object shape",
        "background_gray": "local patch median excluding every TXT box in that image; includes nearby structures",
        "background_std": "standard deviation in same patch; texture/edge proxy, not a verified complex-background label",
    }
    summaries, conditions, sources, details = [], [], [], []
    val_distribution = None
    for run in RUNS:
        recorded_path = ROOT / "reports/overnight_20261003" / f"{run}_evaluation.json"
        recorded = json.loads(recorded_path.read_text())["metrics"]["iou50"]["val"]
        pred = ROOT / "reports" / f"preds_{run}_val.csv"
        result, _, gt = evaluate(pred, "val", thr=recorded["thr"], matching="iou50")
        for key in ["tp", "fp", "fn", "n_gt", "n_images", "f1"]:
            assert np.isclose(result[key], recorded[key]), (run, key)
        _, vm = load_gt("val")
        gt = features(gt, vm, "val")
        if val_distribution is None:
            val_distribution = distribution(gt)
        raw = pd.read_csv(pred)
        selected = raw[raw.score >= result["thr"]]
        alarm_stems = set(selected.stem)
        positives = set(gt.stem)
        summaries.append({"run": run, "threshold": result["thr"], "images": len(vm), "gt": len(gt),
                          "tp": result["tp"], "fp": result["fp"], "fn": result["fn"],
                          "positive_images_with_any_alarm": len(positives & alarm_stems),
                          "positive_images_without_any_alarm": len(positives - alarm_stems),
                          "positive_images_with_any_correct_localization": int(gt.groupby("stem").detected.any().sum()),
                          "positive_images_with_all_boxes_found": int(gt.groupby("stem").detected.all().sum()),
                          "real_normal_images": len(vm) - len(positives)})
        for feature, (lo, hi) in cuts.items():
            groups = np.where(gt[feature] <= lo, "low_le_train_q25",
                              np.where(gt[feature] > hi, "high_gt_train_q75", "middle"))
            for label in ["low_le_train_q25", "middle", "high_gt_train_q75"]:
                sub = gt[groups == label]
                conditions.append({"run": run, "feature": feature, "group": label,
                                   "q25": lo, "q75": hi, "n_gt": len(sub), "n_images": sub.stem.nunique(),
                                   "n_bursts": sub.burst_id.nunique(), "tp": int(sub.detected.sum()),
                                   "fn": int((~sub.detected).sum()),
                                   "miss_rate": float((~sub.detected).mean()) if len(sub) else None})
        details.append(gt.assign(run=run))
        sources += [{"path": str(p.relative_to(ROOT)), "sha256": sha(p)} for p in [pred, recorded_path]]
    pd.DataFrame(conditions).to_csv(OUT / "validation_conditions.csv", index=False)
    detail = pd.concat(details, ignore_index=True)
    detail.to_csv(OUT / "validation_box_features.csv", index=False)
    errors = pd.read_csv(ROOT / "reports/overnight_20261003/diagnostics/errors.csv")
    errors = errors[(errors.split == "val") & (errors.error == "FN")]
    unique_gt = detail[["stem", "cx", "cy", "w", "h"]].drop_duplicates()
    assert len(unique_gt) == 243
    report = {"scope": "official train and validation only; saved predictions rescored; no training or inference",
              "split_md5": V2_SPLIT_MD5, "training_images": len(tm), "training_boxes": len(train),
              "train_distribution": distribution(train), "validation_distribution": val_distribution,
              "train_long_side_le_15": int((train["size"] <= 15).sum()),
              "cuts": cuts, "definitions": definitions, "models": summaries,
              "validation_fn_categories": errors.groupby(["run", "category"]).size().to_dict(),
              "validation_unique_missed_gt": int(errors[["stem", "gt_index"]].drop_duplicates().shape[0]),
              "sources": sources + [{"path": str(DATA.relative_to(ROOT) / "split.csv"), "sha256": sha(DATA / "split.csv")}],
              "limitations": ["Only 3 box FN per model. Associations do not establish causal failure conditions.",
                              "Same capture burst is correlated. No independent-object confidence intervals claimed.",
                              "Geometric FN and positive product without alarm are different endpoints.",
                              "Every validation image is positive. Normal false-alarm rate, factory review load and pass safety are not estimable.",
                              "Quartile thresholds fitted to training only; exploratory validation analysis after earlier test access.",
                              "GT-derived defect conditions cannot by themselves trigger live review when a defect is missed."]}
    report["validation_fn_categories"] = [{"run": k[0], "category": k[1], "count": v}
                                            for k, v in report["validation_fn_categories"].items()]
    (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps({k: report[k] for k in ["training_images", "training_boxes", "train_distribution",
                     "train_long_side_le_15", "models", "validation_fn_categories", "validation_unique_missed_gt"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
