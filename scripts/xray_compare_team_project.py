"""Audit published aggregate counts; do not run or rescore external models."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    ref = args.reference
    exp = ref / "X-ray_실험"
    files = subprocess.check_output(["git", "-C", str(ref), "ls-files", "-z"]).decode().split("\0")
    files = [p for p in files if p]
    commit = subprocess.check_output(["git", "-C", str(ref), "rev-parse", "HEAD"]).decode().strip()
    sources = {}
    rows = []
    for name in ("yolov8n", "yolo11s", "rfdetr_s", "qwen25vl3b_lora"):
        p = exp / "preds" / f"{name}_eval.json"
        sources[str(p.relative_to(ref))] = digest(p)
        data = read_json(p)
        for split in ("val", "test"):
            d = data[split]
            tp, fp, fn = [d[k] for k in ("TP", "FP", "FN")]
            f1 = 2 * tp / (2 * tp + fp + fn)
            assert abs(f1 - d["F1"]) < 1e-10, (name, split)
            rows.append(dict(model=name, split=split, images=d["n_images"], gt=tp+fn,
                             tp=tp, fp=fp, fn=fn, f1=f1, recall=tp/(tp+fn),
                             normal_false_alarm_rate_synthetic_only=d["normal_false_alarm_rate"]))
    own_path = root / "docs/evidence/common_epoch_20261004/comparison.csv"
    own = list(csv.DictReader(own_path.open()))
    for row in own:
        tp, fp, fn = [int(row[k]) for k in ("tp", "fp", "fn")]
        assert abs(2*tp/(2*tp+fp+fn)-float(row["f1"])) < 1e-10
    ablation = read_json(exp / "report/a_vs_d.json")
    sources["report/a_vs_d.json"] = digest(exp / "report/a_vs_d.json")
    source_code = exp / "scripts/report_analysis.py"
    code = source_code.read_text()
    assert "cx >= b[0]-2" in code and "cx <= b[2]+2" in code
    assert "m1 >= THR[primary] and m2 >= THR[second]" in code
    sources["scripts/report_analysis.py"] = digest(source_code)
    audit = read_json(root / "docs/evidence/fullframe_policy_20261005/audit.json")
    live = read_json(root / "docs/evidence/approved_pair_20261005/live_summary.json")
    output = dict(
        scope="Source audit and F1 arithmetic from aggregate counts only; no training, inference, or prediction rescore",
        reference_url="https://github.com/kyungrae2002/kampai", reference_commit=commit,
        reference_tracked_files=len(files), reference_pdf_sha256=digest(args.pdf),
        reference_pdf_name=args.pdf.name, source_sha256=sources,
        reference_metrics=rows, reference_A_D=ablation,
        our_test_metrics=[{k: r[k] for k in ("model", "epoch", "tp", "fp", "fn", "f1", "recall")} for r in own],
        our_sources_sha256={str(own_path.relative_to(root)): digest(own_path)},
        our_pair_audit=audit, our_live_mean_ms=live["mean_ms"], our_live_p95_ms=live["p95_ms"],
        our_live_latency_scope=live["latency_scope"],
        confirmed_code_rules={"center_hit_margin_pixels_per_side": 2, "product_reject_rule": "both model maxima >= respective high thresholds"},
        limitations=["Different splits and preprocessing prohibit cross-project ranking",
                     "Reference per-box prediction CSVs, images, labels and weights absent from this snapshot",
                     "No actual normal production evaluation in either current comparison",
                     "Both-center and image alarms do not prove every physical foreign object was found",
                     "No model, threshold or split change"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"aggregate_f1_checks": len(rows)+len(own), "commit": commit,
                      "tracked_files": len(files), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
