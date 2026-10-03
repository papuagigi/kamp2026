"""Render actual generated images and labels for the preprocessing document."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties,fontManager
from matplotlib.patches import Rectangle
import cv2
import pandas as pd
from xray_config import ROOT,DATA
from xray_prepare_roadmap import OUT,labels,xyxy


def main():
    font='/System/Library/Fonts/AppleSDGothicNeo.ttc';fontManager.addfont(font)
    plt.rcParams.update({'font.family':FontProperties(fname=font).get_name(),'axes.unicode_minus':False})
    aug=pd.read_csv(OUT/'augmentation.csv')
    complete=set(aug.loc[aug.kind=='remove_all_refined','parent'])
    partial=set(aug.loc[aug.kind=='remove_partial_refined','parent'])
    stem=sorted(complete & partial)[0]
    fig,axes=plt.subplots(2,3,figsize=(13,9))
    kinds=['original','crop','rotate_p','remove_all_refined','remove_partial_refined','offbar']
    titles=['① 색 네모 제거 기본 사진','② 사진 일부를 잘라 확대','③ 사진 전체를 7.5° 회전',
            '④ 이물 전부 제거','⑤ 이물 일부 제거','⑥ 다른 위치에 이물 무늬 추가']
    for ax,kind,title in zip(axes.flat,kinds,titles):
        ip=OUT/'images/clean'/f'{stem}.png' if kind=='original' else OUT/'images/augment'/f'{stem}__{kind}.png'
        lp=DATA/'labels/train'/f'{stem}.txt' if kind=='original' else OUT/'labels/augment'/f'{stem}__{kind}.txt'
        g=cv2.imread(str(ip),0);h,w=g.shape;b=labels(lp)
        ax.imshow(g,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
        for x,y,xx,yy in xyxy(b,w,h):ax.add_patch(Rectangle((x,y),xx-x,yy-y,ec='#0871c1',fill=False,lw=1))
        ax.set_title(title,fontsize=13,pad=12);ax.axis('off')
        ax.text(.5,-.05,f'저장된 라벨 {len(b)}개',transform=ax.transAxes,ha='center',fontsize=11)
    fig.suptitle('직접 생성한 전처리 결과 — 학습 효과는 별도 비교합니다',fontsize=20,y=.97)
    fig.subplots_adjust(left=.025,right=.975,top=.88,bottom=.10,wspace=.08,hspace=.3)
    fig.text(.04,.025,'파란 박스는 저장된 TXT를 그린 설명선입니다. ④는 합성 음성 사진이며 실제 정상 제품을 찍은 사진이 아닙니다.',fontsize=11,color='#526476')
    fig.savefig(ROOT/'docs/figures/19_실행한_전처리_사진예시.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
