"""Show real validation predictions and a common magnified patch; images are not training inputs."""
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager,FontProperties
from matplotlib.patches import Rectangle
from xray_config import ROOT,DATA
from xray_eval import load_gt

REPORT=ROOT/'reports/roadmap_20261002'
CANDIDATES=[('v2_roadmap_yolo_official','YOLOv8n'),('v2_roadmap_faster_official_cpu','Faster R-CNN'),('v2_roadmap_rfdetr_official_oneclass','RF-DETR-S')]


def main():
    font='/System/Library/Fonts/AppleSDGothicNeo.ttc';fontManager.addfont(font)
    plt.rcParams.update({'font.family':FontProperties(fname=font).get_name(),'axes.unicode_minus':False})
    gt,_=load_gt('val');details=pd.read_csv(REPORT/'localization/v2_roadmap_yolo_official.csv')
    # A demanding example chosen mechanically from validation localization diagnostics.
    row=details.sort_values(['iou','stem']).iloc[0];stem=row.stem;target=gt.iloc[int(row.gt_index)]
    image=cv2.imread(str(DATA/'images/val'/f'{stem}.png'),0);h,w=image.shape
    available=[(r,n) for r,n in CANDIDATES if (ROOT/'reports'/f'preds_{r}_val_custom_ap_v2_eval.json').exists()]
    fig,axes=plt.subplots(2,len(available),figsize=(4.4*len(available),8.4),squeeze=False)
    for col,(run,name) in enumerate(available):
        score=json.loads((ROOT/'reports'/f'preds_{run}_val_custom_ap_v2_eval.json').read_text());pred=pd.read_csv(ROOT/'reports'/f'preds_{run}_val.csv')
        pred=pred[(pred.stem==stem)&(pred.score>=score['thr'])];truth=gt[gt.stem==stem]
        for zoom in [0,1]:
            ax=axes[zoom,col];ax.imshow(image,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
            for rows,color in [(truth,'#087bff'),(pred,'#ed8a00')]:
                for b in rows.itertuples():ax.add_patch(Rectangle((b.cx-b.w/2,b.cy-b.h/2),b.w,b.h,fill=False,ec=color,lw=1.4))
            if zoom:
                ax.set_xlim(max(0,target.cx-22),min(w,target.cx+22));ax.set_ylim(min(h,target.cy+22),max(0,target.cy-22))
                ax.set_title('같은 이물 주변을 확대',fontsize=12)
            else:ax.set_title(name+' · 실제 예측',fontsize=16)
            ax.axis('off')
    fig.suptitle('같은 검증 사진에서 모델이 그린 박스',fontsize=20,y=.975)
    fig.text(.04,.06,'파란색: 공식 TXT 정답   /   주황색: 검증 임계값을 넘긴 AI 예측',fontsize=12)
    fig.text(.04,.025,'학습 입력에는 이 설명용 박스를 넣지 않습니다. 한 장의 예시가 전체 모델 순위를 뜻하지 않습니다.',fontsize=10,color='#526476')
    fig.subplots_adjust(top=.90,bottom=.12,wspace=.12,hspace=.22)
    output=ROOT/'docs/figures/20_실제모델_예측과_정답.png';fig.savefig(output,dpi=160);plt.close(fig)
    (REPORT/'prediction_figure_sources.json').write_text(json.dumps({'stem':stem,'selection':'lowest YOLO matched IoU on validation',
        'gt_index':int(row.gt_index),'runs':[r for r,n in available],'figure':str(output.relative_to(ROOT)),
        'uses_test':False,'original_training_image_modified':False},ensure_ascii=False,indent=2))
    print(output.relative_to(ROOT))

if __name__=='__main__':main()
