"""Render source-backed EDA and frozen localization examples for docs/19 and docs/20."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/xray-review-mpl')
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
font_manager.fontManager.addfont('/System/Library/Fonts/AppleSDGothicNeo.ttc')
plt.rcParams.update({'font.family': 'Apple SD Gothic Neo', 'axes.unicode_minus': False})
OUT = ROOT / 'docs/figures'


def main():
    audit = ROOT / 'reports/eda_review_20261004'
    s = json.loads((audit / 'summary.json').read_text())
    boxes = pd.read_csv(audit / 'label_boxes.csv')
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    vals = [s['labels']['images'], s['raw']['distinct_images']-s['labels']['images']]
    ax[0].bar(['라벨 제공 데이터셋', '미라벨 데이터셋'], vals, color=['#147aab', '#a4b5c1'])
    for i, v in enumerate(vals):
        ax[0].text(i, v+40, f'{v:,}장', ha='center', fontsize=12)
    ax[0].set_ylim(0, 2400); ax[0].set_ylabel('사진 수')
    ax[0].set_title('고유 원본 2,532장 · 중복 사본 277개 제외')
    ax[1].hist(boxes['size'], bins=np.arange(4.5, boxes['size'].max()+1.5, 1), color='#147aab')
    ax[1].axvline(11, color='#d7681a', ls='--', lw=2, label='중앙값 11픽셀')
    ax[1].set_xlabel('정답 박스의 긴 변 (원본 영상 픽셀)'); ax[1].set_ylabel('박스 수')
    ax[1].set_title('라벨 제공 데이터셋의 정답 박스 1,147개'); ax[1].legend()
    fig.suptitle('EDA: 라벨 유무와 이물 박스 크기를 먼저 확인', fontsize=15)
    fig.text(.5,.025,'2026-09-28 분석의 동일 실행기로 2026-10-04 재확인. 박스 크기는 실제 이물 지름과 다름.',ha='center', fontsize=10)
    fig.tight_layout(rect=[0,.06,1,.94]); fig.savefig(OUT/'19_EDA_라벨과_박스크기.png',dpi=160); plt.close(fig)

    base = ROOT / 'reports/common_epoch_20261004/evaluation/diagnostics'
    errors = pd.read_csv(base/'errors.csv')
    stem = '002_20200623_203040(4)'
    errors = errors[(errors.split=='test') & (errors.stem==stem) & (errors.error=='FN')]
    names = ['YOLOv8n', 'Faster R-CNN', 'RF-DETR-S', 'D-FINE-S']
    suffixes = ['yolo_mps', 'faster_cuda', 'rfdetr_mps', 'dfine_cuda']
    pixels = np.asarray(Image.open(ROOT/f'data/xray_v2/images/test/{stem}.png').convert('RGB'))
    gt = errors.iloc[0]
    fig, axes = plt.subplots(1,5,figsize=(13,4.4))
    for ax in axes:
        ax.imshow(pixels, interpolation='nearest')
        ax.set_xlim(gt.cx-14,gt.cx+14); ax.set_ylim(gt.cy+14,gt.cy-14)
        ax.set_xticks([]); ax.set_yticks([])
    axes[0].set_title('실제 입력 사진 확대'); axes[0].set_xlabel('가운데 검은 점이 이물')
    for ax, name, suffix in zip(axes[1:], names, suffixes):
        row = errors[errors.run==f'v2_common20_{suffix}_20261004'].iloc[0]
        pred = pd.read_csv(ROOT/f'reports/preds_{row.run}_test.csv').iloc[int(row.companion_pred_index)]
        for box, color in [(row,'#0879c1'),(pred,'#de6500')]:
            ax.add_patch(Rectangle((box.cx-box.w/2,box.cy-box.h/2),box.w,box.h,fill=False,ec=color,lw=1.8))
        ax.set_title(name); ax.set_xlabel(f'IoU {row.companion_iou:.3f}\n현재 위치 채점: FP 1 + FN 1')
    fig.suptitle('이물이 박스 안에 보여도, TXT와 겹치는 면적이 작으면 위치 채점에서 실패',fontsize=14)
    fig.text(.5,.03,'파랑: 공식 TXT 정답 박스 (9×5픽셀)   |   주황: 고정 탐지 임계값을 통과한 예측 박스\n실제 영상·좌표 그대로 표시. 선은 설명용. 실제 이물의 픽셀 윤곽 정답은 없음.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.20,1,.91]); fig.savefig(OUT/'common20_containment_detail.png',dpi=160);plt.close(fig)
    print('Created docs/figures/19_EDA_라벨과_박스크기.png and common20_containment_detail.png')


if __name__=='__main__':
    main()
