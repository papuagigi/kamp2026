"""Create Korean teaching figures from existing raw/shared images; no dataset edits."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
from xray_audit_shared_dataset import ROOT, FOLDER, labels, rect

AUDIT=ROOT/'reports/provided_preprocessing_audit_20261002'
OUT=ROOT/'reports/preprocessing_guide_20261002'
OUT.mkdir(parents=True,exist_ok=True)
font=FontProperties(fname='/System/Library/Fonts/AppleSDGothicNeo.ttc')
plt.rcParams.update({'font.family':font.get_name(),'font.size':12,'axes.unicode_minus':False})
from matplotlib import font_manager
font_manager.fontManager.addfont('/System/Library/Fonts/AppleSDGothicNeo.ttc')
files=pd.read_csv(AUDIT/'files.csv')
mapping=pd.read_csv(AUDIT/'original_mapping.csv').set_index('name')
parent='h1_002_20200622_203053(2)'
basepath=FOLDER/'학습'

def load(name):
    p=basepath/'images'/f'{name}.png'
    return np.asarray(Image.open(p).convert('RGB')), labels(basepath/'labels'/f'{name}.txt')

raw=np.asarray(Image.open(ROOT/mapping.loc[parent,'raw_path']).convert('RGB'))
base,pb=load(parent);h,w=base.shape[:2]
# Common viewport only for the explanatory figure: saved dataset images are untouched.
viewport=(82,64,347,237)
blue='#1164bb';orange='#cc6418';grey='#676d77'

def box(ax,b,color,style='-',lw=1.35):
    x1,y1,x2,y2=rect(b,w,h)
    ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor=color,lw=lw,linestyle=style))

def eq(a,b): return np.allclose(a,b,atol=1.1e-6,rtol=0)

panels=[
 ('A','장비가 저장한 원본','전체 서로 다른 사진 2,532장',None,'원래 이물 3곳 · 장비 색 네모 있음'),
 ('B','색 네모 제거','500장: 학습 340 + 검증 80 + 테스트 80',parent,'이물 3곳 유지 · 색 선 자리를 주변 무늬로 채움'),
 ('C','이물 조각을 다른 곳에 합성','추가 680장',parent+'_syn_random0','이 예시: 기존 3곳 + 새 이물 2곳'),
 ('D','조각 변형 + 경계 부근 합성','추가 680장',parent+'_syn_edge_rot0','이 예시: 기존 3곳 + 새 이물 2곳'),
 ('E','이물 전부 제거','추가 340장',parent+'_clean','이 예시: 3곳을 모두 지움 → 정답 0개'),
 ('F','이물 일부 제거','추가 452장',parent+'_partial_xoo','이 예시: 맨 위만 지움 → 정답 2개')]
fig,axes=plt.subplots(3,2,figsize=(12.0,11.2))
fig.subplots_adjust(left=.045,right=.955,top=.89,bottom=.10,hspace=.53,wspace=.13)
manifest=[]
for ax,(letter,title,count,name,sub) in zip(axes.flat,panels):
    arr,b=(raw,pb) if name is None else load(name)
    ax.imshow(arr,interpolation='nearest')
    if name is not None:
        for bb in b:box(ax,bb,blue if any(eq(bb,x) for x in pb) else orange)
        for bb in pb:
            if not any(eq(bb,x) for x in b):box(ax,bb,grey,'--')
    x1,y1,x2,y2=viewport;ax.set_xlim(x1,x2);ax.set_ylim(y2,y1);ax.axis('off')
    ax.set_title(f'{letter}  {title}\n{count}',fontproperties=font,fontsize=14,pad=10,loc='left',color='#202b39')
    ax.text(0,-.075,sub,transform=ax.transAxes,fontproperties=font,fontsize=12,color='#404854',va='top')
    path=ROOT/mapping.loc[parent,'raw_path'] if name is None else basepath/'images'/f'{name}.png'
    manifest.append({'panel':letter,'name':path.stem,'source':path.relative_to(ROOT).as_posix(),'title':title,'example_boxes':len(b)})
fig.suptitle('같은 사진이 어떻게 달라졌을까?',fontproperties=font,fontsize=24,color='#172538',y=.978)
fig.text(.5,.935,'A의 원본 한 장에서 만들어진 실제 파일을 비교했습니다.',ha='center',fontproperties=font,fontsize=13,color='#586475')
fig.text(.055,.057,'파란 실선: 원래 정답   /   주황 실선: 새로 합성한 정답   /   회색 점선: 지운 자리',fontproperties=font,fontsize=13,color='#283747')
fig.text(.055,.026,'B~F의 선은 설명용으로 덧그렸습니다. 실제 PNG에는 없습니다. 모든 칸은 같은 범위를 확대해 표시했습니다.',fontproperties=font,fontsize=11,color='#586475')
fig.savefig(OUT/'types_actual_examples.png',dpi=170,facecolor='white');plt.close(fig)

# Magnified before/after pairs show the actual pixels with no annotation boxes.
syn,_=load(parent+'_syn_random0');clean,_=load(parent+'_clean')
new=pd.read_csv(AUDIT/'synthetic_boxes.csv')
choice=new[(new.name==parent+'_syn_random0')].sort_values('max_darkening').iloc[-1]
newx,newy=float(choice.cx),float(choice.cy)
bb=pb[np.argsort(pb[:,2])][1];oldx,oldy=float(bb[1]*w),float(bb[2]*h)
fig,ax=plt.subplots(2,2,figsize=(9,8.2));fig.subplots_adjust(left=.10,right=.91,top=.86,bottom=.10,hspace=.36,wspace=.19)
for i,(first,second,cx,cy,tt,cap) in enumerate([(base,syn,newx,newy,'C: 빈 곳에 이물 추가','원래 없던 어두운 무늬가 생깁니다.'),(base,clean,oldx,oldy,'E: 원래 이물 제거','어두운 점을 지우고 주변 밝기·잡음으로 채웁니다.')]):
    x1,y1=max(0,int(cx)-15),max(0,int(cy)-15);x2,y2=min(w,x1+31),min(h,y1+31)
    for j,arr in enumerate([first,second]):
        ax[i,j].imshow(arr[y1:y2,x1:x2],interpolation='nearest');ax[i,j].axis('off')
        ax[i,j].set_title(tt+' · '+('처리 전' if j==0 else '처리 후'),fontproperties=font,fontsize=14,pad=10)
    ax[i,0].text(0,-.08,cap,transform=ax[i,0].transAxes,fontproperties=font,fontsize=12,color='#404854',va='top')
fig.suptitle('작은 이물의 실제 변화 확대',fontproperties=font,fontsize=23,y=.966)
fig.text(.5,.915,'같은 위치의 픽셀을 확대했습니다. 밝기나 대비는 바꾸지 않았습니다.',fontproperties=font,fontsize=12,ha='center',color='#586475')
fig.text(.10,.035,'이 그림의 잘라 보기·확대는 설명용입니다. 학습 데이터에 새 Crop 증강을 적용한 것이 아닙니다.',fontproperties=font,fontsize=11,color='#586475')
fig.savefig(OUT/'pixel_changes_zoom.png',dpi=160,facecolor='white');plt.close(fig)

# Recount current delivered files and annotations for the teaching table.
counts=[]
for kor,split in [('학습','train'),('검증','val'),('테스트','test')]:
    paths=list((FOLDER/kor/'images').glob('*.png'))
    for p in paths:
        rows=files[(files.name==p.stem)&(files.split==split)]
        assert len(rows)==1,p
        category=rows.iloc[0].category
        counts.append({'split':split,'category':category,'images':1,'boxes':len(labels(FOLDER/kor/'labels'/f'{p.stem}.txt'))})
counts=pd.DataFrame(counts).groupby(['split','category'])[['images','boxes']].sum().reset_index()
counts.to_csv(OUT/'counts.csv',index=False)
(OUT/'example_sources.json').write_text(json.dumps({'parent':parent,'display_crop':viewport,'panels':manifest,'source_audit':'reports/provided_preprocessing_audit_20261002'},ensure_ascii=False,indent=2))
print(counts.to_string(index=False))
print('Created actual-photo comparison and zoom figures:',OUT.relative_to(ROOT))
