#!/usr/bin/env python3
"""
여러 모델의 채점 결과(reports/*_eval.json)를 모아 조건별 재현율 그림과 비교표를 만든다.
사용: .venv/bin/python scripts/xray_condition_plot.py reports/preds_classical_dog_val_eval.json reports/preds_v1_yolov8n_640_val_eval.json ...
출력: docs/figures/06_recall_by_condition.png, reports/model_comparison.md
"""
import sys, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager
for f in ["AppleGothic", "Apple SD Gothic Neo"]:
    if any(f == x.name for x in font_manager.fontManager.ttflist): plt.rcParams["font.family"] = f; break
plt.rcParams["axes.unicode_minus"] = False
ROOT = Path(__file__).resolve().parents[1]
NAMES = {"classical_dog": "고전 DoG", "v1_yolov8n_640": "YOLOv8n 640", "v1_yolov8n_1024": "YOLOv8n 1024", "v1_yolov8s_1024": "YOLOv8s 1024"}
def label(path):
    s = Path(path).stem.replace("preds_", "").replace("_val_eval", "").replace("_test_eval", "")
    return NAMES.get(s, s)
def main():
    files = sys.argv[1:]; res = {label(f): json.load(open(f)) for f in files}
    conds = ["is_testpiece", "machine", "width", "size_bin", "edge_bin", "contrast_bin"]
    titles = {"is_testpiece": "시편 여부", "machine": "호기", "width": "이미지 폭(제품 규격)", "size_bin": "이물 크기", "edge_bin": "제품 가장자리 거리", "contrast_bin": "이물 대비"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    for ax, c in zip(axes.ravel(), conds):
        keys = sorted({k for r in res.values() for k in r["recall_by_condition"][c]})
        w = 0.8 / max(len(res), 1)
        for i, (name, r) in enumerate(res.items()):
            vals = [r["recall_by_condition"][c].get(k, (0, 0))[0] for k in keys]
            ax.bar([j + i * w for j in range(len(keys))], vals, w, label=name)
        ns = [next((r["recall_by_condition"][c][k][1] for r in res.values() if k in r["recall_by_condition"][c]), 0) for k in keys]
        ax.set_xticks([j + w * (len(res) - 1) / 2 for j in range(len(keys))]); ax.set_xticklabels([f"{k}\n(n={n})" for k, n in zip(keys, ns)], fontsize=8)
        ax.set_ylim(0, 1.05); ax.set_title(titles[c]); ax.set_ylabel("재현율")
    axes[0, 0].legend(fontsize=8); plt.suptitle("조건별 재현율 (검증 세트, 각 모델의 최고 F1 문턱값 기준)", fontsize=11); plt.tight_layout()
    out = ROOT / "docs" / "figures" / "06_recall_by_condition.png"; plt.savefig(out, dpi=130); plt.close(); print("saved", out)
    lines = ["| 모델 | 정밀도 | 재현율 | F1 | AP | 사진당 오탐 | 완전검출 사진 비율 | 재현율99% 문턱값 | 그때 오탐 수 |", "|---|---|---|---|---|---|---|---|---|"]
    for name, r in res.items():
        lines.append(f"| {name} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {r['ap']:.3f} | {r['fp_per_image']:.2f} | {r['images_fully_detected']:.2f} | {r['thr_recall99']} | {r['fp_at_recall99']} |")
    md = ROOT / "reports" / "model_comparison.md"; md.write_text("\n".join(lines) + "\n", encoding="utf-8"); print("\n".join(lines))
if __name__ == "__main__": main()
