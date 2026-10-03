#!/usr/bin/env python3
"""인수 점검: 기존 전처리·채점 코드를 재실행하고 결과를 별도 폴더에 보존한다.

사용: .venv/bin/python scripts/xray_handover_check.py
원본, data/xray_v2, 기존 reports/runs는 읽기만 한다. 재학습·추론은 하지 않는다.
"""
import argparse
import ast
import contextlib
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

import pandas as pd
import torch

os.environ["XRAY_DATA"] = "xray_v2"
import xray_data_audit as audit
import xray_eval as evaluation
import xray_prepare_v2 as prepare

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/xray_v2"
EXPECTED_SPLIT_MD5 = "8e58184ae24dfd1b48da2e9dfd88fedc"


def digest(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def legacy_metrics_match(current, saved):
    """AP 정의 변경·추가 메타데이터와 기존 탐지 지표의 변화를 구분한다."""
    current = json.loads(json.dumps(current))
    for key, value in saved.items():
        if key == "ap":
            continue
        if isinstance(value, (int, float)):
            if current.get(key) is None or not math.isclose(current[key], value, abs_tol=1e-12):
                return False
        elif current.get(key) != value:
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/handover_20261001")
    args = parser.parse_args()
    out = ROOT / args.out
    # 점검 출력이 원본이나 기존 결과를 덮어쓰지 않도록 전용 하위 폴더만 허용한다.
    if out.parent.resolve() != (ROOT / "reports").resolve() or not out.name.startswith("handover_"):
        parser.error("--out은 reports/handover_로 시작하는 전용 폴더여야 합니다")
    out.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(DATA / "manifest.csv", encoding="utf-8-sig")
    split = pd.read_csv(DATA / "split.csv")
    checks = {}
    checks["split_md5_matches"] = digest(DATA / "split.csv") == EXPECTED_SPLIT_MD5
    checks["unique_stems"] = bool(man.stem.is_unique and split.stem.is_unique)
    checks["split_matches_manifest"] = man[["stem", "burst_id", "split"]].sort_values("stem").reset_index(drop=True).equals(split.sort_values("stem").reset_index(drop=True))
    checks["no_burst_crosses_splits"] = bool(man.groupby("burst_id").split.nunique().max() == 1)
    checks["no_source_md5_crosses_splits"] = bool(man.groupby("md5").split.nunique().max() == 1)
    checks["source_hashes_match"] = all(digest(ROOT / row.path) == row.md5 for row in man.itertuples())
    checks["missing_label_suspects_train_only"] = bool((man.loc[man.label_issue.fillna("").str.contains("라벨누락"), "split"] == "train").all())

    # 기존 원본 점검 스크립트를 그대로 쓰되 산출물 위치만 분리한다.
    old_out = audit.OUT
    audit.OUT = out / "data_audit"
    audit.OUT.mkdir(exist_ok=True)
    try:
        with contextlib.redirect_stdout(io.StringIO()) as log:
            audit.main()
        (out / "data_audit_stdout.txt").write_text(log.getvalue(), encoding="utf-8")
    finally:
        audit.OUT = old_out
    raw_summary = json.loads((out / "data_audit/summary.json").read_text())
    checks["raw_audit_matches_saved"] = raw_summary == json.loads((ROOT / "reports/data_audit/summary.json").read_text())
    raw = pd.read_csv(out / "data_audit/raw_images.csv", encoding="utf-8-sig")
    line_images = raw[raw.first_copy & (raw.mark_type == "선")]
    checks["line_images_excluded_from_labeled_data"] = not bool(set(line_images.stem) & set(man.stem))

    # 새 임시 폴더에서 원래 v2 준비 스크립트를 실행한다. 기존 v2는 수정하지 않는다.
    old_out, old_argv = prepare.OUT, sys.argv
    with tempfile.TemporaryDirectory(prefix="v2_rebuild_", dir=out) as tmp:
        prepare.OUT = Path(tmp) / "xray_v2"
        sys.argv = ["xray_prepare_v2.py", "--seed", "0"]
        try:
            with contextlib.redirect_stdout(io.StringIO()) as log:
                prepare.main()
            (out / "v2_rebuild_stdout.txt").write_text(log.getvalue(), encoding="utf-8")
            generated = json.loads((prepare.OUT / "summary.json").read_text())
            checks["rebuilt_summary_matches"] = generated == json.loads((DATA / "summary.json").read_text())
            checks["rebuilt_manifest_matches"] = (prepare.OUT / "manifest.csv").read_bytes() == (DATA / "manifest.csv").read_bytes()
            checks["rebuilt_split_matches"] = (prepare.OUT / "split.csv").read_bytes() == (DATA / "split.csv").read_bytes()
            images_equal = labels_equal = 0
            for row in man.itertuples():
                image = Path("images") / row.split / (row.stem + ".png")
                label = Path("labels") / row.split / (row.stem + ".txt")
                images_equal += (prepare.OUT / image).read_bytes() == (DATA / image).read_bytes()
                labels_equal += (prepare.OUT / label).read_bytes() == (DATA / label).read_bytes()
            checks["all_rebuilt_images_equal"] = images_equal == len(man)
            checks["all_rebuilt_labels_equal"] = labels_equal == len(man)
        finally:
            prepare.OUT, sys.argv = old_out, old_argv

    # 채점은 기존 evaluate만 사용한다. 평가 임계값은 검증에서 구한다.
    metrics = {}
    for model in ["v2_yolov8n_640", "v2_classical_dog"]:
        val_path = Path("reports") / f"preds_{model}_val.csv"
        test_path = Path("reports") / f"preds_{model}_test.csv"
        val, _, _ = evaluation.evaluate(str(val_path), "val")
        test, _, _ = evaluation.evaluate(str(test_path), "test", thr=val["thr"])
        metrics[model] = {"val": val, "test_at_val_threshold": test}
        saved_val = json.loads((ROOT / f"reports/preds_{model}_val_eval.json").read_text())
        # evaluate의 조건별 값은 tuple, 저장 JSON은 list이므로 저장 형식으로 비교한다.
        checks[model + "_val_non_ap_matches_saved"] = legacy_metrics_match(val, saved_val)
        metrics[model]["legacy_val_ap"] = saved_val["ap"]
        if model == "v2_yolov8n_640":
            saved_test = json.loads((ROOT / f"reports/preds_{model}_test_valthr_eval.json").read_text())
            checks[model + "_test_non_ap_matches_saved"] = legacy_metrics_match(test, saved_test)
            metrics[model]["legacy_test_ap"] = saved_test["ap"]

    versions = {}
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        if "==" in line and not line.startswith("#"):
            package, expected = line.split("==")
            actual = importlib.metadata.version(package)
            versions[package] = {"expected": expected, "actual": actual}
    checks["direct_dependency_versions_match"] = all(v["expected"] == v["actual"] for v in versions.values())
    for script in (ROOT / "scripts").glob("*.py"):
        ast.parse(script.read_text(), filename=script.name)
    checks["python_scripts_parse"] = True
    runs = []
    for result in sorted((ROOT / "runs").glob("*/results.csv")):
        rows = pd.read_csv(result)
        runs.append({"name": result.parent.name, "epochs_logged": len(rows), "best_weights_present": (result.parent / "weights/best.pt").is_file()})
    summary = {
        "scope": "기존 전처리·원본 점검·예측 CSV 재채점. 학습과 추론은 재실행하지 않음",
        "branch": git("branch", "--show-current"), "head": git("rev-parse", "--short", "HEAD"),
        "runtime": {"python": platform.python_version(), "available_device": "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu", "evaluation_device": "cpu", "versions": versions},
        "checks": checks, "all_checks_passed": all(checks.values()),
        "v2": generated, "rebuilt_equal_images": images_equal, "rebuilt_equal_labels": labels_equal,
        "raw": raw_summary["raw"], "line_images": raw_summary["marks"]["line_images"],
        "line_images_by_date": line_images.dt.str[:10].value_counts().sort_index().to_dict(),
        "metrics": metrics, "training_runs": runs,
        "outputs_file_count": sum(p.is_file() for p in (ROOT / "outputs").rglob("*")),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"runtime", "raw", "metrics", "v2"}}, ensure_ascii=False, indent=2))
    for name, values in metrics.items():
        for split_name, res in values.items():
            if not isinstance(res, dict):
                continue
            print(name, split_name, json.dumps({k: res[k] for k in ["n_images", "n_gt", "thr", "precision", "recall", "f1", "fp_per_image", "ap"]}))
    print("saved", out.relative_to(ROOT))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
