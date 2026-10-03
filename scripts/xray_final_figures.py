"""Render actual final-model test errors from saved predictions; never select a model here."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager, FontProperties
from matplotlib.patches import Rectangle
from PIL import Image
from xray_config import ROOT, DATA
from xray_eval import evaluate, match, iou

REPORT=ROOT/'reports/roadmap_20261002'


def main():
    choice=json.loads((REPORT/'model_selection.json').read_text())
    run=choice['recommended_run']; threshold=choice['threshold']
    predpath=ROOT/'reports'/f'preds_{run}_test.csv'
    score,_,gt=evaluate(predpath,'test',thr=threshold,matching='iou50')
    pred=pd.read_csv(predpath);pred=pred[pred.score>=threshold]
    matched,_=match(pred,gt,iou_thr=.5,iou_only=True)
    font='/System/Library/Fonts/AppleSDGothicNeo.ttc';fontManager.addfont(font)
    plt.rcParams.update({'font.family':FontProperties(fname=font).get_name(),'axes.unicode_minus':False})
    misses=gt[~gt.detected].sort_values(['stem','cx'])
    examples=misses.head(3) if len(misses) else gt.sort_values(['stem','cx']).head(3)
    fig,axes=plt.subplots(2,len(examples),figsize=(5*len(examples),8.5),squeeze=False)
    records=[]
    for column,(idx,row) in enumerate(examples.iterrows()):
        with Image.open(DATA/'images/test'/f'{row.stem}.png') as im: gray=np.array(im.convert('L'))
        truth=gt[gt.stem==row.stem]; predictions=matched[matched.stem==row.stem]
        distances=np.hypot(predictions.cx-row.cx,predictions.cy-row.cy)
        close=predictions.loc[distances.idxmin()] if len(distances) else None
        best_iou=float(iou(row[['cx','cy','w','h']].to_numpy(float),close[['cx','cy','w','h']].to_numpy(float))) if close is not None else None
        for zoom in [0,1]:
            ax=axes[zoom,column];ax.imshow(gray,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
            for frame,color in [(truth,'#087BFF'),(predictions,'#ED8A00')]:
                for b in frame.itertuples():
                    ax.add_patch(Rectangle((b.cx-b.w/2,b.cy-b.h/2),b.w,b.h,fill=False,edgecolor=color,lw=1.6))
            if zoom:
                ax.set_xlim(max(0,row.cx-24),min(gray.shape[1],row.cx+24))
                ax.set_ylim(min(gray.shape[0],row.cy+24),max(0,row.cy-24))
                caption='채택한 예측 없음' if close is None else f'가장 가까운 예측 IoU {best_iou:.3f}'
                ax.set_title(caption,fontsize=14)
            else:
                ax.set_title(f'예시 {column+1} · '+('위치 기준 미충족' if not row.detected else '탐지 성공'),fontsize=17)
            ax.axis('off')
        records.append({'stem':row.stem,'gt_index':int(idx),'detected':bool(row.detected),'nearest_prediction_iou':best_iou})
    fig.suptitle('추천 모델의 실제 테스트 예측과 확대 사진',fontsize=22,y=.98)
    fig.text(.04,.066,'파랑: 공식 TXT 정답 박스   /   주황: 고정 탐지 임계값 이상 예측 박스',fontsize=14)
    fig.text(.04,.025,'가까이 예측했더라도 IoU 0.5 미만이면 해당 위치 기준에서 미탐·오탐이 됩니다. 모델 선택 후의 오류 설명입니다.',fontsize=11)
    fig.subplots_adjust(top=.90,bottom=.13,hspace=.17,wspace=.10)
    output=ROOT/'docs/figures/20_최종후보_테스트_오류예시.png';fig.savefig(output,dpi=150);plt.close(fig)
    (REPORT/'final_error_figure_sources.json').write_text(json.dumps({'run':run,'threshold':threshold,
        'selection':'first three unmatched GT sorted by filename then x; successes only if no misses',
        'examples':records,'used_for_model_selection':False,'metric_version':score['metric_version']},ensure_ascii=False,indent=2))
    print(output.relative_to(ROOT))


if __name__=='__main__':main()
