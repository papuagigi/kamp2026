"""Render a fixed small diagnostic sample; do not assign visual correctness labels."""
import json
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageFont
from xray_config import ROOT,DATA
from xray_eval import load_gt
from xray_recovery import atomic_json


def main():
    base=ROOT/'reports/review_roles_20261005';out=base/'visual';out.mkdir(exist_ok=True)
    gt,_=load_gt('val');first=gt.sort_values(['stem','cx','cy']).head(6)
    columns=[('clean',None),('TXT',None),('RF full',ROOT/'runs/v2_common20_rfdetr_mps_20261004/validation/epoch_008.csv'),
             ('RF crop',base/'base_rf/candidate.csv')]
    data={name:pd.read_csv(p) for name,p in columns if p}
    sheet=Image.new('RGB',(800,6*236),'white');font=ImageFont.load_default(size=13);records=[]
    for n,g in enumerate(first.itertuples()):
        im=Image.open(DATA/'images/val'/f'{g.stem}.png').convert('RGB')
        side=40;x0=int(g.cx-side/2);y0=int(g.cy-side/2)
        for c,(name,p) in enumerate(columns):
            patch=im.crop((x0,y0,x0+side,y0+side)).resize((200,200),Image.Resampling.NEAREST)
            d=ImageDraw.Draw(patch)
            boxes=[]
            if name=='TXT':boxes=[g]
            elif p:
                f=data[name];boxes=list(f[(f.stem==g.stem)&(f.score>=.70370466)&((f.cx-g.cx).abs()<20)&((f.cy-g.cy).abs()<20)].itertuples())
            for b in boxes:
                d.rectangle([(b.cx-b.w/2-x0)*5,(b.cy-b.h/2-y0)*5,(b.cx+b.w/2-x0)*5,(b.cy+b.h/2-y0)*5],outline='blue' if name=='TXT' else '#ec8b00',width=2)
            sheet.paste(patch,(c*200,n*236+30));ImageDraw.Draw(sheet).text((c*200+5,n*236+6),f'{n+1} {name}',font=font,fill='black')
        records.append(dict(stem=g.stem,cx=g.cx,cy=g.cy,w=g.w,h=g.h,source='val',sampling='first six official GT sorted by stem/cx/cy; diagnostic only'))
    sheet.save(out/'base_crop_comparison.png');atomic_json(out/'cases.json',records)

if __name__=='__main__':main()
