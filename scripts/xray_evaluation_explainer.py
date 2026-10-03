"""Draw reproducible evaluation diagrams and actual source/label/prediction examples."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.font_manager import FontProperties,fontManager
from xray_config import ROOT,DATA
from xray_eval import iou

BLUE='#1568c4';ORANGE='#d76c00';GREEN='#319675';GRAY='#405169'

def setup():
    font='/System/Library/Fonts/AppleSDGothicNeo.ttc';fontManager.addfont(font)
    plt.rcParams.update({'font.family':FontProperties(fname=font).get_name(),'axes.unicode_minus':False,'font.size':12})

def box(ax,x,y,w,h,color,**kw):ax.add_patch(Rectangle((x,y),w,h,fill=False,ec=color,lw=2.6,**kw))

def diagrams():
    fig,axes=plt.subplots(2,3,figsize=(15,9.5));fig.patch.set_facecolor('white')
    for ax in axes.flat:ax.set_xlim(0,12);ax.set_ylim(0,10);ax.set_aspect('equal');ax.axis('off')
    examples=[(0,'완전히 일치',1.),(2,'일부만 겹침',1/3),(5,'겹치지 않음',0.)]
    for ax,(shift,title,expected) in zip(axes[0],examples):
        a=(4,5,4,4);b=(4+shift,5,4,4);value=iou(a,b);assert np.isclose(value,expected)
        box(ax,2,3,4,4,BLUE);box(ax,2+shift,3,4,4,ORANGE,linestyle='--')
        intersection=max(0,4-shift)*4;union=32-intersection
        if intersection:ax.add_patch(Rectangle((2+shift,3),4-shift,4,fc=GREEN,alpha=.3))
        ax.set_title(title,fontsize=17,pad=5)
        ax.text(6,1.3,f'겹친 넓이 {intersection} ÷ 합친 넓이 {union}\nIoU = {value:.2f}',ha='center',va='center',fontsize=14)
    ax=axes[1,0];box(ax,3,4,4,4,BLUE);box(ax,6,4,4,4,ORANGE,linestyle='--')
    ax.plot([5,8],[6,6],'k.-');ax.text(6.5,6.3,'6픽셀',ha='center',fontsize=12)
    ax.text(6,2,'8×8 박스를 가로로 6픽셀 이동\nIoU = 16/112 ≈ 0.14\n기존 거리 기준: TP / IoU 0.5: 불일치',ha='center',va='center',fontsize=12)
    ax.set_title('중심이 가까워도 박스는 다를 수 있음',fontsize=14)
    ax=axes[1,1]
    for x in [2,8]:ax.plot(x,5,'ko',ms=6);box(ax,x-1,4,2,2,BLUE)
    box(ax,.9,3.9,2.2,2.2,ORANGE);box(ax,.6,3.6,2.8,2.8,ORANGE,linestyle='--')
    ax.text(2,7.6,'정답 A\n예측 ① 0.90\n예측 ② 0.80',ha='center',fontsize=12)
    ax.text(8,7.6,'정답 B\n예측 없음',ha='center',fontsize=12)
    ax.text(6,1.4,'두 예측 모두 A를 가리킴\n임계값 0.50 이상인 두 예측을 채점',ha='center',fontsize=12)
    ax.set_title('한 이물에 박스를 두 번 그렸다면?',fontsize=14)
    ax=axes[1,2]
    ax.text(.6,8.7,'① 높은 점수의 예측 → 정답 A: TP 1개',fontsize=12)
    ax.text(.6,6.5,'② 정답 A는 이미 찾았음 → 중복 FP 1개',fontsize=12)
    ax.text(.6,4.3,'③ 정답 B는 끝까지 못 찾음 → FN 1개',fontsize=12)
    ax.text(6,1.8,'정밀도 = 1/2, 재현율 = 1/2\nF1 = 0.50',ha='center',fontsize=16,color=GRAY)
    ax.set_title('정답 하나는 한 번만 맞혔다고 셈',fontsize=14)
    fig.suptitle('IoU와 일대일 채점 — 실제 실험 결과가 아닌 설명용 도식',fontsize=21,y=.99)
    fig.text(.03,.025,'파랑: 정답 박스  /  주황: 예측 박스  /  초록: 겹친 부분     ·     IoU = 교집합 넓이 ÷ 합집합 넓이',fontsize=13)
    fig.subplots_adjust(top=.91,bottom=.09,hspace=.25,wspace=.18)
    out=ROOT/'docs/figures/19_IoU와_중복예측_설명.png';fig.savefig(out,dpi=160);plt.close(fig);return str(out.relative_to(ROOT))

def actual():
    source=json.loads((ROOT/'reports/roadmap_20261002/prediction_figure_sources.json').read_text());stem=source['stem']
    row=pd.read_csv(DATA/'manifest.csv').set_index('stem').loc[stem]
    raw=Image.open(ROOT/row['path']).convert('RGB');clean=Image.open(DATA/'images/val'/f'{stem}.png').convert('RGB');w,h=clean.size
    labels=np.loadtxt(DATA/'labels/val'/f'{stem}.txt',ndmin=2)[:,1:]*[w,h,w,h]
    run='v2_roadmap_yolo_official';pred=pd.read_csv(ROOT/'reports'/f'preds_{run}_val.csv')
    threshold=json.loads((ROOT/'reports'/f'preds_{run}_val_iou50_v1_eval.json').read_text())['thr']
    pred=pred[(pred.stem==stem)&(pred.score>=threshold)]
    fig,axes=plt.subplots(1,3,figsize=(15,5.5))
    for ax,im,title in zip(axes,[raw,clean,clean],['① 원본에 이미 있던 색 네모','② TXT 좌표로 그린 정답 박스','③ AI 좌표로 그린 예측 박스']):
        ax.imshow(im);ax.axis('off');ax.set_title(title,fontsize=16,pad=14)
    for cx,cy,bw,bh in labels:box(axes[1],cx-bw/2,cy-bh/2,bw,bh,BLUE)
    for r in pred.itertuples():box(axes[2],r.cx-r.w/2,r.cy-r.h/2,r.w,r.h,ORANGE)
    fig.suptitle('색 네모·정답 박스·예측 박스는 서로 다른 정보입니다',fontsize=22,y=.99)
    fig.text(.045,.07,'① 검사기 표시: 제거 대상     ② 공식 TXT: 학습·채점 기준     ③ YOLOv8n 실제 예측: 채점 대상',fontsize=13)
    fig.text(.045,.025,'②·③의 선은 설명용으로 그렸습니다. 실제 학습 사진에는 없으며, 색깔 자체가 정답·예측을 결정하지 않습니다.',fontsize=12,color=GRAY)
    fig.subplots_adjust(top=.83,bottom=.16,wspace=.14)
    out=ROOT/'docs/figures/19_색네모_정답박스_예측박스.png';fig.savefig(out,dpi=160);plt.close(fig)
    return {'figure':str(out.relative_to(ROOT)),'stem':stem,'prediction_run':run,'threshold':threshold,'raw':row['path'],'source_images_unchanged':True}

def main():
    setup();info={'conceptual_figure':diagrams(),'actual_figure':actual()}
    (ROOT/'reports/roadmap_20261002/evaluation_explainer_sources.json').write_text(json.dumps(info,ensure_ascii=False,indent=2));print(json.dumps(info,ensure_ascii=False))
if __name__=='__main__':main()
