"""Prepare visual annotation aids from cleaned pixels; no detector/old label inputs.

Candidate points are measurement aids, never accepted without a recorded visual choice.
All review outputs are versioned separately from official and existing pseudo labels.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/direct_labeling_20261006'
DATA=ROOT/'data/xray_direct_labels_20261006'
FONT='/System/Library/Fonts/Helvetica.ttc'

def candidate_points(g):
    f=cv2.GaussianBlur(g.astype(np.float32),(3,3),.6)
    bh=cv2.morphologyEx(f,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)))-f
    dil=cv2.dilate(bh,np.ones((7,7),np.uint8))
    yy,xx=np.where((bh>=dil-1e-5)&(bh>=6)&(f<208))
    seeds=sorted(zip(xx.tolist(),yy.tolist()), key=lambda p:float(bh[p[1],p[0]]),reverse=True)
    keep=[]
    h,w=g.shape
    for x,y in seeds:
        if min(x,y,w-1-x,h-1-y)<5:continue
        if any((x-q['cx'])**2+(y-q['cy'])**2<196 for q in keep):continue
        patch=bh[max(0,y-10):y+11,max(0,x-10):x+11]
        binary=(patch>=max(2.5,float(bh[y,x])*.55)).astype('uint8')
        n,labs,stats,cent=cv2.connectedComponentsWithStats(binary,8)
        k=int(labs[y-max(0,y-10),x-max(0,x-10)])
        if k==0:continue
        px,py,pw,ph,area=map(int,stats[k]);
        if area>100 or max(pw,ph)>16 or max(pw,ph)>5*max(1,min(pw,ph)):continue
        x0=max(0,x-10)+px;y0=max(0,y-10)+py
        # Dot extent plus two-pixel margin; bounds are adjustable during direct review.
        x1=max(0,x0-2);y1=max(0,y0-2);x2=min(w,x0+pw+2);y2=min(h,y0+ph+2)
        keep.append({'cx':x,'cy':y,'xyxy':[x1,y1,x2,y2],'pixel_contrast':round(float(bh[y,x]),2)})
    # Limit visual clutter; reviewer must inspect full image and add missed points manually.
    keep=sorted(keep,key=lambda q:q['pixel_contrast'],reverse=True)[:12]
    keep=sorted(keep,key=lambda q:(q['cy'],q['cx']))
    for i,q in enumerate(keep,1):q['id']=i
    return keep

def prepare():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'sheets').mkdir(exist_ok=True)
    rows=list(csv.DictReader((ROOT/'data/xray_roadmap_20261002/inventory.csv').open()))
    rows=sorted([r for r in rows if r['status']=='pseudo_candidate'],key=lambda r:r['stem'])
    assert len(rows)==1900
    records=[]
    for idx,r in enumerate(rows):
        p=ROOT/r['image'];g=cv2.imread(str(p),0);h,w=g.shape
        records.append({'index':idx,'stem':r['stem'],'image':r['image'],'group':int(r['group']),
                        'width':w,'height':h,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                        'candidates':candidate_points(g)})
    (OUT/'candidates.json').write_text(json.dumps(records,ensure_ascii=False,separators=(',',':'))+'\n')
    # Separate source-free index: no old labels, device marks, or prediction scores shown to reviewer.
    (OUT/'index.csv').write_text('index,stem,image,width,height,group\n'+''.join(f"{r['index']},{r['stem']},{r['image']},{r['width']},{r['height']},{r['group']}\n" for r in records))
    print(json.dumps({'count':len(records),'candidate_counts_percentiles':np.percentile([len(r['candidates']) for r in records],[0,25,50,75,95,100]).tolist()}))

def render(start,end,per=12):
    records=json.loads((OUT/'candidates.json').read_text());font=ImageFont.truetype(FONT,16);tiny=ImageFont.truetype(FONT,13)
    for s in range(start,end,per):
        batch=records[s:min(end,s+per)]
        # Every source pixel remains visible, resized only when large. Coordinates remain source units.
        W,H=470,420;sheet=Image.new('RGB',(W*4,H*math.ceil(len(batch)/4)),'#eef1f4');d=ImageDraw.Draw(sheet)
        for j,r in enumerate(batch):
            im=Image.open(ROOT/r['image']).convert('RGB');scale=min(1,450/r['width'],380/r['height']);im=im.resize((round(r['width']*scale),round(r['height']*scale)),Image.Resampling.LANCZOS)
            ox=(j%4)*W+(W-im.width)//2;oy=(j//4)*H+30;sheet.paste(im,(ox,oy))
            d.text(((j%4)*W+7,(j//4)*H+5),f"{r['index']:04d}  {r['width']}x{r['height']}  {len(r['candidates'])} points",font=font,fill='#111111')
            for q in r['candidates']:
                x=ox+q['cx']*scale;y=oy+q['cy']*scale
                # Keep target pixels unobscured. ID and thin outline outside the candidate point.
                x1,y1,x2,y2=q['xyxy'];d.rectangle((ox+x1*scale,oy+y1*scale,ox+x2*scale,oy+y2*scale),outline='#08b5d2',width=1)
                d.text((x+8,y-13),str(q['id']),font=tiny,fill='#b60074',stroke_width=1,stroke_fill='white')
        path=OUT/'sheets'/f'{s:04d}_{min(end,s+per)-1:04d}.png';sheet.save(path)
        print(str(path.relative_to(ROOT)))

def detail(indices):
    records=json.loads((OUT/'candidates.json').read_text());font=ImageFont.truetype(FONT,20)
    for i in indices:
        r=records[i];im=Image.open(ROOT/r['image']).convert('RGB');s=2
        enlarged=im.resize((im.width*s,im.height*s),Image.Resampling.NEAREST)
        out=Image.new('RGB',(enlarged.width*2,enlarged.height+35),'white');out.paste(enlarged,(0,35));out.paste(enlarged,(enlarged.width,35));d=ImageDraw.Draw(out)
        d.text((8,5),f"{i:04d} unmarked / coordinate aids (2x source)",font=font,fill='black')
        for q in r['candidates']:
            x=enlarged.width+q['cx']*s;y=35+q['cy']*s
            d.rectangle((x-12,y-12,x+12,y+12),outline='#00a4bf',width=1);d.text((x+14,y-16),str(q['id']),font=font,fill='#bc005e')
        p=OUT/'sheets'/f'detail_{i:04d}.png';out.save(p);print(str(p.relative_to(ROOT)))

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','render','detail']);p.add_argument('--start',type=int,default=0);p.add_argument('--end',type=int,default=1900);p.add_argument('--per',type=int,default=12);p.add_argument('--indices',type=int,nargs='*',default=[]);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='render':render(a.start,a.end,a.per)
    else:detail(a.indices)
if __name__=='__main__':main()
