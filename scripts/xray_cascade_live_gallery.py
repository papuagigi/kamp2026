"""Render targeted inspection evidence without assigning visual judgments."""
import json
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageFont
from xray_cascade_audit import save,sha
from xray_cascade_live import OUT
from xray_config import ROOT,DATA
from xray_eval import load_gt


def main():
    scope=json.loads((OUT/'visual_scope.json').read_text());gt,_=load_gt('val')
    gt=gt[gt.stem.isin(scope['stems'])]
    policy=json.loads((OUT/'policy.json').read_text())
    first=pd.read_csv(OUT/'original_repeat0_val.csv');first=first[first.score>=policy['thresholds'][policy['first']]]
    second=pd.read_csv(OUT/'second_repeat0_val.csv');second=second[second.score>=policy['thresholds'][policy['second']]]
    calls={r['stem']:r for r in json.loads((OUT/'execution.json').read_text())['calls'] if r['repeat']==0}
    font=ImageFont.load_default(size=14);folder=OUT/'visual';folder.mkdir(exist_ok=True)
    cases=[];sheetrows=[]
    for number,(ix,g) in enumerate(gt.iterrows(),1):
        cid=f'L{number:03}';source=DATA/'images/val'/f'{g.stem}.png'
        im=Image.open(source).convert('RGB');group=gt[gt.stem==g.stem]
        panels={};half=max(18,g.w,g.h)
        for name,preds in [('YOLO',first),('RF',second)]:
            preds=preds[preds.stem==g.stem].copy()
            selected=[]
            for pi,r in preds.iterrows():
                closest=((group.cx-r.cx)**2+(group.cy-r.cy)**2).idxmin()
                if closest==ix:
                    box=[float(r[k]) for k in ['cx','cy','w','h']]
                    selected.append(dict(pred_index=int(pi),box=box,score=float(r.score)))
                    half=max(half,abs(r.cx-g.cx)+r.w/2+3,abs(r.cy-g.cy)+r.h/2+3)
            panels[name]=selected
        side=int(np.ceil(2*half));x0,y0=int(g.cx-side/2),int(g.cy-side/2)
        base=im.crop((x0,y0,x0+side,y0+side)).resize((180,180),Image.Resampling.NEAREST)
        row=Image.new('RGB',(540,214),'white');d=ImageDraw.Draw(row)
        for col,name in enumerate(['clean','YOLO','RF']):
            patch=base.copy();dd=ImageDraw.Draw(patch)
            if name!='clean':
                for j,p in enumerate(panels[name],1):
                    cx,cy,w,h=p['box'];box=[(cx-w/2-x0)*180/side,(cy-h/2-y0)*180/side,(cx+w/2-x0)*180/side,(cy+h/2-y0)*180/side]
                    dd.rectangle(box,outline='#f58b00',width=2);dd.text((max(0,box[0]),max(0,box[1]-14)),f'p{j}',font=font,fill='#cc5500')
            row.paste(patch,(col*180,28))
            label=cid+' clean' if name=='clean' else name+(' not run' if name=='RF' and not calls[g.stem]['referred'] else f' {len(panels[name])} boxes')
            d.text((col*180+3,4),label,font=font,fill='black')
        dest=folder/f'{cid}.png';row.save(dest);sheetrows.append(row)
        cases.append(dict(case=cid,stem=g.stem,gt_index=int(ix),reference=[float(g[k]) for k in ['cx','cy','w','h']],
            panels=panels,rf_executed=calls[g.stem]['referred'],state=calls[g.stem]['state'],
            source=str(source.relative_to(ROOT)),source_sha256=sha(source),figure=str(dest.relative_to(ROOT)),figure_sha256=sha(dest)))
    for off in range(0,len(sheetrows),6):
        rows=sheetrows[off:off+6];s=Image.new('RGB',(540,len(rows)*214),'white')
        for n,row in enumerate(rows):s.paste(row,(0,n*214))
        s.save(folder/f'sheet{off//6+1:02}.png')
    save(folder/'cases.json',cases)
    print(json.dumps(dict(images=len(scope['stems']),targets=len(cases),sheets=int(np.ceil(len(cases)/6))),indent=2))


if __name__=='__main__':main()
