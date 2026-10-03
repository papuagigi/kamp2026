"""Render visual evidence from the read-only shared-dataset audit tables."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from xray_audit_shared_dataset import ROOT, OUT, FOLDER, labels, rect

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
M=pd.read_csv(OUT/'original_mapping.csv').set_index('name')
F=pd.read_csv(OUT/'files.csv').set_index('name')
K={'train':'학습','val':'검증','test':'테스트'}

def read(name):
    p=FOLDER/K[F.loc[name,'split']]/'images'/f'{name}.png'
    return np.asarray(Image.open(p)),labels(p.parent.parent/'labels'/f'{name}.txt')

def draw(ax,img,b=None,crop=None,title='',color='#10c680'):
    ax.imshow(img,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
    if b is not None:
        h,w=img.shape[:2]
        for bb in b:
            x1,y1,x2,y2=rect(bb,w,h)
            ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor=color,lw=1))
    if crop:
        x1,y1,x2,y2=crop;ax.set_xlim(x1,x2);ax.set_ylim(y2,y1)
    ax.set_title(title);ax.axis('off')


def crop_labels(b,w,h,pad=15):
    a=[rect(bb,w,h) for bb in b]
    return max(0,min(x[0] for x in a)-pad),max(0,min(x[1] for x in a)-pad),min(w,max(x[2] for x in a)+pad),min(h,max(x[3] for x in a)+pad)

names=['h1_002_20200622_203053(2)','h2_002_20200623_123055(5)','h3_001_20200624_162816(3)']
fig,ax=plt.subplots(3,3,figsize=(9,10),layout='constrained')
for i,name in enumerate(names):
    p=ROOT/M.loc[name,'raw_path']; im=Image.open(p);raw=np.asarray(im.convert('RGB'));idx=np.asarray(im);shared,b=read(name);h,w=shared.shape[:2]
    box=crop_labels(b,w,h,18)
    if i==1:
        import cv2
        cc,hh=cv2.findContours((idx>=244).astype(np.uint8),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        holes=[c for c,hr in zip(cc,hh[0]) if hr[3]!=-1 and cv2.contourArea(c)>=9]
        xys=np.concatenate(holes).reshape(-1,2)
        box=(max(0,int(xys[:,0].min())-15),max(0,int(xys[:,1].min())-15),min(w,int(xys[:,0].max())+15),min(h,int(xys[:,1].max())+15))
    draw(ax[i,0],raw,crop=box,title=f'Machine {i+1}: raw marks')
    draw(ax[i,1],shared,crop=box,title='Processed pixels')
    draw(ax[i,2],shared,b,crop=box,title='TXT label overlay')
fig.suptitle('Equipment marks removed; TXT labels are separate annotations',fontsize=13)
fig.savefig(OUT/'marks_before_after.png',dpi=180);plt.close(fig)

name='h2_002_20200623_123055(5)';shared,b=read(name);raw=np.asarray(Image.open(ROOT/M.loc[name,'raw_path']).convert('RGB'))
idx=np.asarray(Image.open(ROOT/M.loc[name,'raw_path']));h,w=idx.shape
import cv2
cc,hh=cv2.findContours((idx>=244).astype(np.uint8),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
holes=[c for c,hr in zip(cc,hh[0]) if hr[3]!=-1 and cv2.contourArea(c)>=9]
centers=b[:,1:3]*[w,h]
missing=[c for c in holes if not any(cv2.pointPolygonTest(c,tuple(map(float,p)),False)>=0 for p in centers)]
xys=np.concatenate(holes).reshape(-1,2)
box=(int(xys[:,0].min())-15,int(xys[:,1].min())-15,int(xys[:,0].max())+15,int(xys[:,1].max())+15)
fig,ax=plt.subplots(1,3,figsize=(8,6),layout='constrained')
draw(ax[0],raw,crop=box,title='Raw: 3 equipment detections')
draw(ax[1],shared,crop=box,title='After preprocessing')
draw(ax[2],shared,b,crop=box,title='Validation TXT: only 2 labels')
for c in missing:
    x,y,bw,bh=cv2.boundingRect(c)
    ax[2].add_patch(Rectangle((x,y),bw,bh,fill=False,edgecolor='#ffad00',lw=2,linestyle='--'))
    ax[2].text(x+bw+3,y+bh/2,'Missing\nlabel',color='#996000',fontsize=10)
fig.suptitle(name+' (validation)',fontsize=11)
fig.savefig(OUT/'missing_validation_label.png',dpi=170);plt.close(fig)

names=['h3_001_20200623_123214(3)_clean','h3_001_20200623_163209(3)_partial_xxo']
fig,ax=plt.subplots(2,3,figsize=(10,6),layout='constrained')
for i,name in enumerate(names):
    row=F.loc[name]; parent=row.parent;im,b=read(name);pg,pb=read(parent); bb=pb[np.argsort(pb[:,2])][0];h,w=pg.shape[:2];cx,cy=bb[1]*w,bb[2]*h;crop=(cx-18,cy-18,cx+18,cy+18)
    draw(ax[i,0],pg,crop=crop,title='Before removal')
    draw(ax[i,1],im,crop=crop,title=f'After: {row.category}')
    ax[i,2].imshow(cv2.morphologyEx(np.asarray(Image.fromarray(im).convert('L')),cv2.MORPH_BLACKHAT,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))),cmap='magma',vmin=0,vmax=60)
    ax[i,2].set_xlim(crop[0],crop[2]);ax[i,2].set_ylim(crop[3],crop[1]);ax[i,2].set_title('Residual dark response (0-60)');ax[i,2].axis('off')
fig.suptitle('Two strongest residuals among 1,468 erased labeled objects',fontsize=12)
fig.savefig(OUT/'erasure_residuals.png',dpi=180);plt.close(fig)

parent='h1_002_20200622_203053(2)';names=[parent,parent+'_syn_random0',parent+'_syn_edge_rot0',parent+'_clean']
part=next(n for n in F.index if n.startswith(parent+'_partial_'));names.append(part)
fig,ax=plt.subplots(1,5,figsize=(14,5),layout='constrained')
for a,n,t in zip(ax,names,['Original','Random added objects','Edge + rotated patches','All objects removed','Some objects removed']):
    img,b=read(n);draw(a,img,b,title=t,color='#00d090')
fig.suptitle('The same training parent; green boxes are drawn only for this audit figure',fontsize=12)
fig.savefig(OUT/'augmentation_examples.png',dpi=170);plt.close(fig)
print('Wrote 4 audit figures')
