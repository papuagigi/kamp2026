"""Draw documented training histories; consumes audit CSV, does not rescore models."""
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/xray-history-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    font = Path('/System/Library/Fonts/AppleSDGothicNeo.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'font.size': 11, 'axes.unicode_minus': False})
    data = pd.read_csv(ROOT / 'reports/training_history_20261004/epoch_curves.csv')
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.2), layout='constrained')
    blue, orange = '#235C8C', '#B9611F'
    for col, (model, title) in enumerate([('yolo', 'YOLOv8n'), ('rfdetr', 'RF-DETR-S'), ('faster', 'Faster R-CNN')]):
        d = data[data.model == model]
        ax = axes[0, col]
        ax.plot(d.epoch, d.train_loss, color=blue, label='학습 손실', marker='o', markersize=3)
        if d.val_loss.notna().any():
            ax.plot(d.epoch, d.val_loss, color=orange, linestyle='--', label='검증 손실')
        ax.set_title(title + (' — 박스 손실' if model == 'yolo' else ' — 학습 총손실'), loc='left', fontweight='bold')
        ax.set_ylabel('손실: 낮을수록 좋음')
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, loc='upper right')
        ax = axes[1, col]
        if d.native_map_50_95.notna().any():
            ax.plot(d.epoch, d.native_map_50_95, color=blue, marker='o', markersize=4)
            best = d.loc[d.native_map_50_95.idxmax()]
            last = d.iloc[-1]
            ax.axvline(best.epoch, color=orange, linestyle='--', linewidth=1)
            ax.text(.03, .94, f"최고: {int(best.epoch)} epoch / {best.native_map_50_95:.4f}\n마지막: {last.native_map_50_95:.4f}",
                    transform=ax.transAxes, va='top', fontsize=10)
            ax.set_ylim(.30, .47)
            ax.set_ylabel('자체 검증 mAP: 높을수록 좋음')
            ax.set_title('mAP@[0.5:0.95] — 세로축 확대', loc='left')
        else:
            ax.text(.5, .60, 'epoch별 검증 기록 없음', transform=ax.transAxes, ha='center', fontsize=15, fontweight='bold')
            ax.text(.5, .40, '학습 손실 감소만으로\n과적합 여부를 판단할 수 없음', transform=ax.transAxes, ha='center', fontsize=12)
            ax.set_yticks([])
            ax.set_title('검증 곡선을 만들 수 없음', loc='left')
        for ax in axes[:, col]:
            ax.set_xlim(1, 20)
            ax.set_xticks([1, 5, 10, 15, 20])
            ax.set_xlabel('epoch: 학습 자료를 반복한 횟수')
            ax.grid(axis='y', alpha=.18)
            ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('저장된 20epoch 기록: 손실은 줄어도 검증 성능이 계속 좋아지지는 않는다', fontsize=17, fontweight='bold')
    fig.supxlabel('실제 저장 로그 · 모델마다 손실 정의가 다름 · 자체 mAP는 공통 F1과 다름 · 과적합 확정이나 새 평가 결과가 아님', fontsize=10)
    out = ROOT / 'docs/figures/20_학습곡선과_저장모델_재검토.png'
    fig.savefig(out, dpi=135, facecolor='white')
    plt.close(fig)
    print(out)


if __name__ == '__main__':
    main()
