#!/usr/bin/env python3
"""기존 채점기의 경계 사례를 가상 정답으로 재현한다. 채점 코드는 수정하지 않는다.

숫자는 설명용 예제이며 모델의 실험 성능이 아니다.
사용: .venv/bin/python scripts/xray_eval_diagnostics.py
"""
import argparse
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import pandas as pd

import xray_eval

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ["stem", "cx", "cy", "w", "h", "size", "edge", "contrast", "card_type", "machine", "width", "n_boxes"]


def run_case(n_gt, n_pred, tied_scores=False):
    gts = pd.DataFrame([dict(stem=f"toy_{i}", cx=20., cy=20., w=10., h=10., size=10., edge=50., contrast=40., card_type="막대1", machine="가상", width=100, n_boxes=1) for i in range(n_gt)], columns=COLUMNS)
    if n_gt == 0:
        # 이전 코드의 열 없는 정답 표를 받아도 안전하게 처리하는지 확인한다.
        gts = pd.DataFrame()
    man = pd.DataFrame({"stem": [f"toy_{i}" for i in range(max(n_gt, n_pred, 1))]})
    preds = pd.DataFrame([dict(stem=f"toy_{i}", cx=20., cy=20., w=10., h=10., score=0.9 if tied_scores else 1. - (i + 1) / (max(n_gt, n_pred) + 1)) for i in range(n_pred)], columns=["stem", "cx", "cy", "w", "h", "score"])
    with tempfile.TemporaryDirectory() as folder:
        csv = Path(folder) / "predictions.csv"
        preds.to_csv(csv, index=False)
        with patch.object(xray_eval, "load_gt", return_value=(gts, man)):
            try:
                result, _, _ = xray_eval.evaluate(csv, "val")
                return {"n_gt": n_gt, "n_pred": n_pred, "tied_scores": tied_scores, **{k: result[k] for k in ["ap", "precision", "recall", "f1"]}}
            except Exception as exc:
                return {"n_gt": n_gt, "n_pred": n_pred, "tied_scores": tied_scores, "error_type": type(exc).__name__, "error": str(exc)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/maintenance_20261001/eval_diagnostics.json")
    args = parser.parse_args()
    report = {"scope": "가상 예제를 기존 xray_eval.evaluate로 채점한 진단. 새 채점 구현과 실험 성능 수치가 아님", "cases": {}}
    for name, values in {
        "perfect_1": (1, 1, False), "perfect_5": (5, 5, False),
        "perfect_243": (243, 243, False), "perfect_5_tied": (5, 5, True),
        "one_of_two_images_missed": (2, 1, False),
        "no_predictions_positive_set": (5, 0, False),
        "negative_only_no_predictions": (0, 0, False),
        "negative_only_one_false_positive": (0, 1, False),
    }.items():
        report["cases"][name] = run_case(*values)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
