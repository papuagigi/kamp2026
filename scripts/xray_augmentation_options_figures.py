"""Make research-only whole-image crop/rotation examples and explanatory figures."""
import json
from pathlib import Path
import numpy as np,pandas as pd,cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties,fontManager
from matplotlib.patches import Rectangle
from xray_prepare_v2 import ROOT
from xray_augmentation_options_study import xyxy,lab,OUT

fontfile='/System/Library/Fonts/AppleSDGothicNeo.ttc';fontManager.addfont(fontfile);font=FontProperties(fname=fontfile)
plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False})
name='002_20200622_203053(2)';man=pd.read_csv(ROOT/'data/xray_v2/manifest.csv');row=man[man.stem==name].iloc[0];assert row.split=='train'
p=ROOT/'data/xray_v2/images/train'/f'{name}.png';g=np.asarray(Image.open(p).convert('L'));h,w=g.shape
b=lab(ROOT/'data/xray_v2/labels/train'/f'{name}.txt');boxes=np.array([xyxy(bb,w,h) for bb in b]);rng=np.random.default_rng(0)
f=.8;lo=np.maximum(0,boxes[:,2:].max(0)-[f*w,f*h]);hi=np.minimum(boxes[:,:2].min(0),[(1-f)*w,(1-f)*h]);assert (hi>=lo).all();x0,y0=rng.uniform(lo,hi)
cm=np.array([[1/f,0,-x0/f],[0,1/f,-y0/f]],float)
rm=cv2.getRotationMatrix2D((w/2,h/2),10,1)

def transform(matrix):
    arr=cv2.warpAffine(g,matrix,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=210)
    bb=[]
    for x1,y1,x2,y2 in boxes:
        points=np.array([[x1,y1,1],[x2,y1,1],[x2,y2,1],[x1,y2,1]])@matrix.T
        bb.append(np.r_[points.min(0),points.max(0)])
    return arr,np.array(bb)

crop,cb=transform(cm);rot,rb=transform(rm)
assert (cb[:,:2]>=0).all() and (cb[:,2:]<=[w,h]).all()
assert (rb[:,:2]>=0).all() and (rb[:,2:]<=[w,h]).all()
Image.fromarray(crop).save(OUT/'prototype_whole_image_crop.png');Image.fromarray(rot).save(OUT/'prototype_whole_image_rotate10.png')
(OUT/'prototype_transforms.json').write_text(json.dumps({'source':p.relative_to(ROOT).as_posix(),'scope':'research example only; not a training dataset','seed':0,'crop_side_fraction':f,'crop_origin':[x0,y0],'crop_affine':cm.tolist(),'rotation_degrees':10,'rotation_affine':rm.tolist(),'original_pixel_xyxy':boxes.tolist(),'crop_pixel_xyxy':cb.tolist(),'rotation_pixel_xyxy':rb.tolist(),'interpolation':'bilinear','fill_gray':210},ensure_ascii=False,indent=2))
fig,axes=plt.subplots(1,3,figsize=(12,4.8));fig.subplots_adjust(left=.025,right=.98,top=.79,bottom=.14,wspace=.08)
for ax,img,bs,title in zip(axes,[g,crop,rot],[boxes,cb,rb],['색 네모 제거 원본','사진 전체 안전 Crop','사진 전체 10° 회전']):
    ax.imshow(img,cmap='gray',vmin=0,vmax=255,interpolation='nearest');ax.axis('off');ax.set_title(title,fontproperties=font,fontsize=14,pad=10)
    for x1,y1,x2,y2 in bs:ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor='#0863b7',lw=1.1))
fig.suptitle('사진을 잘라 확대하거나 돌려도, 점은 막대 끝에 남아 있습니다.',fontproperties=font,fontsize=17,y=.96)
fig.text(.035,.055,'모두 같은 정답 3개입니다. 파란 선은 설명용이며 실제 입력에는 없습니다. 새 학습에는 아직 반영하지 않았습니다.',fontproperties=font,fontsize=11,color='#415165')
fig.savefig(OUT/'whole_image_augmentation_examples.png',dpi=165);plt.close(fig)

# Illustrate original equipment rectangles versus human TXT box sizes on the same location.
raw=np.asarray(Image.open(ROOT/row.path).convert('RGB'));x1,y1,x2,y2=boxes[1];view=(int(x1)-12,int(y1)-12,int(x2)+12,int(y2)+12)
fig,ax=plt.subplots(1,2,figsize=(7.6,4.3));fig.subplots_adjust(top=.75,bottom=.10,wspace=.10)
for aa,img,title in zip(ax,[raw,g],['장비가 표시한 색 네모','공식 TXT가 표시한 정답']):
    aa.imshow(img,cmap='gray',vmin=0,vmax=255,interpolation='nearest');aa.set_xlim(view[0],view[2]);aa.set_ylim(view[3],view[1]);aa.axis('off');aa.set_title(title,fontproperties=font,fontsize=14)
ax[1].add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor='#0863b7',lw=2))
fig.suptitle('색 네모는 위치 후보로 쓸 수 있지만, 정답 크기와 다릅니다.',fontproperties=font,fontsize=16,y=.96)
fig.savefig(OUT/'equipment_vs_txt_example.png',dpi=165);plt.close(fig)
print('Research prototypes: whole-image safe crop and 10 degree rotation; 3 transformed boxes retained each.')
