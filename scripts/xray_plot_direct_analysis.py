"""Build inspectable learning figures and actual-image review sheets from CSVs."""
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from xray_config import ROOT,DATA
from xray_rescore_direct import OUT,RUNS

FONT='/System/Library/Fonts/AppleSDGothicNeo.ttc'
font_manager.fontManager.addfont(FONT)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=FONT).get_name(),'axes.unicode_minus':False,'font.size':11})


def curves():
    t=pd.read_csv(OUT/'learning_curves.csv');fig,axs=plt.subplots(2,2,figsize=(12,8.5),constrained_layout=True)
    for j,(model,rows) in enumerate(t.groupby('model',sort=False)):
        ax=axs[0,j];ax.plot(rows.epoch,rows.validation_f1,'o-',color='#226b9a',markersize=4)
        best=rows[rows.selected].iloc[0];ax.scatter([best.epoch],[best.validation_f1],s=100,color='#bd541a',zorder=3)
        ax.annotate(f"선택: {int(best.epoch)} epoch\nF1 {best.validation_f1:.4f}",(best.epoch,best.validation_f1),xytext=(6,.58),textcoords='data',arrowprops={'arrowstyle':'->'})
        ax.set(title=model+' — 검증 F1',ylabel='IoU ≥ 0.5 F1',ylim=(0,1),xlabel='epoch',xticks=[1,5,10,15,20]);ax.grid(alpha=.18)
        ax=axs[1,j];ax.plot(rows.epoch,rows.train_loss,label='학습 loss');ax.plot(rows.epoch,rows.val_loss,label='검증 loss')
        ax.set(title=model+' — 기록된 손실',xlabel='epoch',ylabel='각 모델의 loss',xticks=[1,5,10,15,20],ylim=(0,None));ax.legend();ax.grid(alpha=.18)
        if model=='RF-DETR-S':
            for ax in axs[:,j]:ax.axvline(14.5,color='#666',linestyle='--',alpha=.6)
    fig.suptitle('20 epoch 기록: 마지막 저장본보다 검증에서 고른 저장본을 사용',fontsize=16)
    fig.supxlabel('F1: 각 epoch의 검증 최적 임계값. 점선: RF-DETR의 MPS 전환. 모델 간 loss 크기는 비교하지 않음.',fontsize=10)
    fig.savefig(OUT/'learning_curves.png',dpi=170);plt.close(fig)
    a=pd.read_csv(OUT/'label_dimensions.csv');a=a[a.split.eq('train')]
    fig,ax=plt.subplots(figsize=(7.7,4.4),constrained_layout=True)
    groups=[a[a.origin.eq(k)].long_side for k in ['official','direct_visual']]
    ax.boxplot(groups,tick_labels=['라벨 제공 학습: 679개 박스','AI 라벨 학습: 3,268개 박스'],showfliers=False)
    for i,g in enumerate(groups,1):ax.text(i+.14,g.median()+.4,f'중앙값 {g.median():.0f}px',color='#b14815')
    ax.set(title='새 라벨은 공식 라벨보다 박스가 작은 경향',ylabel='박스 긴 변 (원본 사진 픽셀)',ylim=(0,24));ax.grid(axis='y',alpha=.15)
    fig.supxlabel('서로 다른 사진 집단의 분포 비교. 실제 이물 크기나 성능 저하 원인의 증명은 아님.',fontsize=10)
    fig.savefig(OUT/'label_dimensions.png',dpi=180);plt.close(fig)


def gallery():
    err=pd.read_csv(OUT/'fn_diagnostics.csv');obj=pd.read_csv(OUT/'objects.csv');meta=[]
    font=ImageFont.truetype(FONT,18);small=ImageFont.truetype(FONT,16)
    dest=OUT/'visual_review';dest.mkdir(exist_ok=True)
    for split in ['test','val']:
        indices=sorted(err[err.split.eq(split)].gt_index.unique())
        for page in range(math.ceil(len(indices)/6)):
            chosen=indices[page*6:(page+1)*6];canvas=Image.new('RGB',(1080,70+260*len(chosen)),'white');d=ImageDraw.Draw(canvas)
            d.text((12,8),f'{split} 오류 대상 전체 검토 {page+1} | 파랑: 공식 TXT · 주황: 고정 임계값 통과 예측',font=font,fill='black')
            d.text((12,35),'사진을 확대하여 표시한 분석 그림. 공식 IoU 채점값은 바꾸지 않습니다.',font=small,fill='black')
            for k,idx in enumerate(chosen):
                rows=obj[obj.split.eq(split)&obj.gt_index.eq(idx)];g=rows.iloc[0];top=70+260*k
                d.text((12,top),f'{split} GT {idx} | {g.stem}',font=font,fill='black')
                x0=int(g.cx)-20;y0=int(g.cy)-20
                source=Image.open(DATA/'images'/split/f'{g.stem}.png').convert('RGB')
                patch=source.crop((x0,y0,x0+40,y0+40)).resize((200,200),Image.Resampling.NEAREST)
                labels=['선 없는 사진','YOLO11s','RF-DETR-S']
                for j,label in enumerate(labels):
                    left=12+j*350;canvas.paste(patch,(left,top+48));d.text((left,top+25),label,font=small,fill='black')
                    if j==0:continue
                    q=rows[rows.model.eq(label)].iloc[0]
                    def rect(cx,cy,w,h,color):
                        d.rectangle([left+5*(cx-w/2-x0),top+48+5*(cy-h/2-y0),left+5*(cx+w/2-x0),top+48+5*(cy+h/2-y0)],outline=color,width=2)
                    rect(g.cx,g.cy,g.w,g.h,'#1567c1')
                    if pd.notna(q.pred_cx) and q.center_distance<18:rect(q.pred_cx,q.pred_cy,q.pred_w,q.pred_h,'#d56400')
                    text=f'IoU {q.best_iou:.3f}\n'+('TP' if q.detected else 'FN')
                    if pd.notna(q.pred_score) and q.center_distance<18:text+=f'\n점수 {q.pred_score:.3f}'
                    else:text+='\n이 위치에\n채택 박스 없음'
                    d.text((left+207,top+85),text,font=small,fill='black')
                meta.append(dict(split=split,gt_index=int(idx),stem=g.stem,sheet=f'visual_review/{split}_{page+1:02d}.png',row=k))
            canvas.save(dest/f'{split}_{page+1:02d}.png')
    (dest/'manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    print('review cases',len(meta),'sheets',len(list(dest.glob('*.png'))),flush=True)


def examples():
    """Keep three traceable cases for the narrative, without changing pixels."""
    meta=json.loads((OUT/'visual_review/manifest.json').read_text())
    chosen=[('test',89),('test',12),('val',24)]
    canvas=Image.new('RGB',(1080,890),'white');d=ImageDraw.Draw(canvas)
    font=ImageFont.truetype(FONT,20);small=ImageFont.truetype(FONT,17)
    d.text((12,8),'실제 오류 사진: 박스 차이와 표시 누락을 구분',font=font,fill='black')
    d.text((12,38),'파랑: 공식 TXT · 주황: 고정 임계값 통과 예측. IoU 점수는 그대로 유지.',font=small,fill='black')
    for n,(split,idx) in enumerate(chosen):
        entry=next(m for m in meta if m['split']==split and m['gt_index']==idx)
        source=Image.open(OUT/entry['sheet']);top=70+260*entry['row']
        canvas.paste(source.crop((0,top,1080,top+260)),(0,80+260*n))
    canvas.save(OUT/'error_examples.png')
    fps=pd.read_csv(OUT/'fp_diagnostics.csv');p=fps[fps.distance>100].iloc[0]
    im=Image.open(DATA/'images/val'/f'{p.stem}.png').convert('RGB');can=Image.new('RGB',(1000,660),'white');d=ImageDraw.Draw(can)
    d.text((20,15),'검증 YOLO 추가 표시: '+p.stem,fill='black',font=font)
    view=im.copy();v=ImageDraw.Draw(view);v.rectangle((p.pred_cx-p.pred_w/2,p.pred_cy-p.pred_h/2,p.pred_cx+p.pred_w/2,p.pred_cy+p.pred_h/2),outline='#d56400',width=1)
    view.thumbnail((610,580));can.paste(view,(15,70))
    patch=im.crop((int(p.pred_cx)-20,int(p.pred_cy)-20,int(p.pred_cx)+20,int(p.pred_cy)+20)).resize((320,320),Image.Resampling.NEAREST);can.paste(patch,(655,125))
    d.text((650,80),'표시 주변 확대 (선 없음)',fill='black',font=font)
    d.text((650,470),f'탐지 점수 {p.score:.6f}\n공식 정답 박스 없음',fill='black',font=font);can.save(OUT/'extra_fp.png')


if __name__=='__main__':curves();gallery();examples()
