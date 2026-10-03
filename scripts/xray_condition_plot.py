#!/usr/bin/env python3
"""
여러 모델의 채점 결과(reports/*_eval.json)를 모아 조건별 재현율 그림과 비교표를 만든다.
사용: .venv/bin/python scripts/xray_condition_plot.py reports/preds_classical_dog_val_eval.json reports/preds_v1_yolov8n_640_val_eval.json ...
출력: docs/figures/06_recall_by_condition.png, reports/model_comparison.md (--out 이름 을 주면 docs/figures/<이름>.png, reports/<이름>.md)
"""
import sys, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager
for f in ["AppleGothic", "Apple SD Gothic Neo"]:
    if any(f == x.name for x in font_manager.fontManager.ttflist): plt.rcParams["font.family"] = f; break
plt.rcParams["axes.unicode_minus"] = False
ROOT = Path(__file__).resolve().parents[1]
NAMES = {"classical_dog": "고전 DoG", "v1_yolov8n_640": "YOLOv8n 640", "v1_yolov8n_1024": "YOLOv8n 1024", "v1_yolov8s_1024": "YOLOv8s 1024",
         "v2_classical_dog": "고전 DoG (v2)", "v2_yolov8n_640": "YOLOv8n 640 (v2)"}
def label(path):
    s = Path(path).stem.replace("_custom_ap_v2_valthr", "").replace("_custom_ap_v2", "").replace("preds_", "").replace("_val_eval", "").replace("_test_eval", "")
    return NAMES.get(s, s)
def main():
    args = sys.argv[1:]; outname = None
    if "--out" in args: i = args.index("--out"); outname = args[i + 1]; args = args[:i] + args[i + 2:]
    files = args; res = {label(f): json.load(open(f)) for f in files}
    first = next(iter(res.values()))["recall_by_condition"]
    conds = ["card_type" if "card_type" in first else "is_testpiece", "machine", "width", "size_bin", "edge_bin", "contrast_bin"]
    titles = {"card_type": "시험편 종류", "is_testpiece": "시편 여부", "machine": "호기", "width": "이미지 폭(제품 규격)", "size_bin": "이물 크기", "edge_bin": "제품 가장자리 거리", "contrast_bin": "이물 대비"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    for ax, c in zip(axes.ravel(), conds):
        keys = sorted({k for r in res.values() for k in r["recall_by_condition"].get(c, {})})
        w = 0.8 / max(len(res), 1)
        for i, (name, r) in enumerate(res.items()):
            vals = [r["recall_by_condition"].get(c, {}).get(k, (float("nan"), 0))[0] for k in keys]
            ax.bar([j + i * w for j in range(len(keys))], vals, w, label=name)
        ns = [next((r["recall_by_condition"][c][k][1] for r in res.values() if k in r["recall_by_condition"].get(c, {})), 0) for k in keys]
        ax.set_xticks([j + w * (len(res) - 1) / 2 for j in range(len(keys))]); ax.set_xticklabels([f"{k}\n(n={n})" for k, n in zip(keys, ns)], fontsize=8)
        ax.set_ylim(0, 1.05); ax.set_title(titles[c]); ax.set_ylabel("재현율")
    splits = ", ".join(sorted({r["split"] for r in res.values()}))
    axes[0, 0].legend(fontsize=8); plt.suptitle(f"조건별 재현율 ({splits}, 각 채점 파일에 기록된 임계값)", fontsize=11); plt.tight_layout()
    out = ROOT / "docs" / "figures" / f"{outname or '06_recall_by_condition'}.png"; plt.savefig(out, dpi=130); plt.close(); print("saved", out)
    lines = ["| 모델 | 정밀도 | 재현율 | F1 | AP | 사진당 오탐 | 완전검출 사진 비율 | 재현율99% 임계값 | 그때 오탐 수 |", "|---|---|---|---|---|---|---|---|---|"]
    fmt = lambda value, digits=3: "해당 없음" if value is None else f"{value:.{digits}f}"
    for name, r in res.items():
        lines.append(f"| {name} | {fmt(r['precision'])} | {fmt(r['recall'])} | {fmt(r['f1'])} | {fmt(r['ap'])} | {fmt(r['fp_per_image'], 2)} | {fmt(r['images_fully_detected'], 2)} | {r['thr_recall99']} | {r['fp_at_recall99']} |")
    md = ROOT / "reports" / f"{outname or 'model_comparison'}.md"; md.write_text("\n".join(lines) + "\n", encoding="utf-8"); print("\n".join(lines))
if __name__ == "__main__": main()
