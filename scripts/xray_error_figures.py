"""Render actual frozen prediction errors; no synthetic image edits or rescoring."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/xray-error-mpl')
import hashlib
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

from xray_config import DATA, ROOT
from xray_eval import iou
from xray_analyze_errors import RUNS, BOX, OUT

FONT = Path('/System/Library/Fonts/AppleSDGothicNeo.ttc')
font_manager.fontManager.addfont(str(FONT))
plt.rcParams.update({'font.family': font_manager.FontProperties(fname=FONT).get_name(),
                     'axes.unicode_minus': False, 'font.size': 10})
NAMES = ['YOLOv8n', 'Faster R-CNN', 'RF-DETR-S']
BLUE, ORANGE, PURPLE = '#0879C1', '#DE6500', '#9223B2'


def rectangle(ax, row, color, style='-', width=1.7):
    ax.add_patch(Rectangle((row.cx-row.w/2, row.cy-row.h/2), row.w, row.h,
                           fill=False, edgecolor=color, linestyle=style, linewidth=width))


def main():
    errors = pd.read_csv(OUT / 'errors.csv')
    gt = pd.read_csv(OUT / 'ground_truth_detail.csv')
    predictions = pd.read_csv(OUT / 'selected_predictions.csv')
    raw = {(run, split): pd.read_csv(ROOT / 'reports' / f'preds_{run}_{split}.csv')
           for run in RUNS for split in ['val', 'test']}
    cases = sorted(set(zip(errors.split, errors.stem)), key=lambda c: (c[0] != 'val', c[1]))
    folder = OUT / 'figures'
    folder.mkdir(exist_ok=True)
    manifest = []

    def draw_row(axes, case, number):
        split, stem = case
        path = DATA / 'images' / split / f'{stem}.png'
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        with Image.open(path) as image:
            pixels = np.asarray(image.convert('RGB'))
        H, W = pixels.shape[:2]
        e = errors[(errors.split == split) & (errors.stem == stem)]
        g = gt[(gt.run == RUNS[0]) & (gt.split == split) & (gt.stem == stem)]
        x0, y0 = float((e.cx-e.w/2).min()-22), float((e.cy-e.h/2).min()-22)
        x1, y1 = float((e.cx+e.w/2).max()+22), float((e.cy+e.h/2).max()+22)
        side = max(70., x1-x0, y1-y0)
        xmid, ymid = (x0+x1)/2, (y0+y1)/2
        x0, y0 = max(0., min(W-side, xmid-side/2)), max(0., min(H-side, ymid-side/2))
        x1, y1 = min(W, x0+side), min(H, y0+side)
        label = '검증' if split == 'val' else '테스트'
        for ax in axes:
            ax.imshow(pixels, interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
        axes[0].add_patch(Rectangle((x0,y0), x1-x0,y1-y0, fill=False, edgecolor=PURPLE, linewidth=1.6))
        axes[0].set_title(f'{number:02d} | {label} 사진 전체', fontweight='bold')
        axes[0].text(.5, -.06, stem + '\n보라 범위를 오른쪽에 확대', ha='center', va='top', transform=axes[0].transAxes, fontsize=9)
        for ax in axes[1:]:
            ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
        axes[1].set_title('확대 · 설명용 선 없음', fontweight='bold')
        axes[1].text(.5,-.06,'색 네모 제거 후 실제 입력 사진\n픽셀을 바꾸지 않고 확대',ha='center',va='top',transform=axes[1].transAxes,fontsize=9)
        for ax, run, name in zip(axes[2:], RUNS, NAMES):
            p = predictions[(predictions.run == run) & (predictions.split == split) & (predictions.stem == stem)]
            p = p[(p.cx >= x0) & (p.cx <= x1) & (p.cy >= y0) & (p.cy <= y1)]
            for _, row in g.iterrows():
                rectangle(ax, row, BLUE)
            descriptions = []
            for _, row in p.iterrows():
                rectangle(ax, row, ORANGE)
                overlap = max(iou(row[BOX].to_numpy(float), r[BOX].to_numpy(float)) for _, r in g.iterrows())
                descriptions.append(f"채택 점수 {row.score:.6f} / IoU {overlap:.3f}")
            low = e[(e.run == run) & (e.error == 'FN') & (e.category == 'matching_box_below_threshold')]
            for _, error in low.iterrows():
                row = raw[(run, split)].iloc[int(error.companion_pred_index)]
                rectangle(ax, row, PURPLE, '--')
                descriptions.append(f"제외 점수 {row.score:.6f} / IoU {error.companion_iou:.3f}")
            sub = e[e.run == run]
            ax.set_title(f'{name}\n사진 전체 FP {sum(sub.error=="FP")} / FN {sum(sub.error=="FN")}', fontweight='bold',fontsize=10)
            ax.text(.5,-.06,'\n'.join(descriptions) or '확대 범위에 채택 예측 없음',ha='center',va='top',transform=ax.transAxes,fontsize=8.6)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == checksum
        return {'case': number, 'split': split, 'stem': stem, 'source': str(path.relative_to(ROOT)),
                'source_sha256': checksum, 'crop_xyxy': [x0,y0,x1,y1]}

    for start in range(0, len(cases), 3):
        batch = cases[start:start+3]
        fig, axes = plt.subplots(len(batch), 5, figsize=(18, 4.1*len(batch)), squeeze=False)
        fig.suptitle('실제 오류 사진 비교 | 파랑 실선: 정답 박스 · 주황 실선: 채택 예측 · 보라 점선: 임계값 미달 예측', fontsize=15,y=.985)
        for i, case in enumerate(batch):
            row = draw_row(axes[i], case, start+i+1)
            row['page'] = start//3+1
            manifest.append(row)
        fig.subplots_adjust(left=.025,right=.985,top=.925,bottom=.09,hspace=.67,wspace=.15)
        fig.text(.5,.018,'공식 TXT와 저장 예측 그대로 표시. 확대는 설명용이며 학습 자료를 바꾸지 않음. IoU≥0.5·일대일 대응·고정 탐지 임계값으로 채점.',ha='center',fontsize=11)
        fig.savefig(folder/f'error_gallery_{start//3+1:02d}.png',dpi=150,facecolor='white')
        plt.close(fig)
    for num, case in enumerate(cases, 1):
        fig, axes = plt.subplots(1,5,figsize=(18,4.5))
        draw_row(axes,case,num)
        fig.suptitle('파랑 실선: 정답 박스 | 주황 실선: 채택 예측 | 보라 점선: 임계값 미달 예측',fontsize=14,y=.98)
        fig.subplots_adjust(left=.025,right=.985,top=.78,bottom=.22,wspace=.15)
        fig.savefig(folder/f'error_case_{num:02d}.png',dpi=150,facecolor='white')
        plt.close(fig)
    (OUT/'figure_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print('Rendered',len(cases),'actual cases on', (len(cases)+2)//3,'gallery pages; source hashes preserved.')


if __name__=='__main__':
    main()
