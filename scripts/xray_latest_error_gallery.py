"""Display actual latest model localization misses for supplementary visual review."""
import json
import pandas as pd
from PIL import Image,ImageDraw,ImageFont
from xray_config import ROOT,DATA
from xray_recovery import atomic_json,sha256


def main():
    b=ROOT/'reports/design_review_20261004';out=b/'latest_visual';out.mkdir(exist_ok=True)
    val=json.loads((b/'six_model_validation.json').read_text())['models']
    test=json.loads((b/'latest_test_descriptive.json').read_text())['models']
    font=ImageFont.load_default(size=15);records=[]
    for split,sources in [('val',[r for r in val if r['name'] in ['YOLO11n','YOLO26n']]),('test',test)]:
        cases={}
        for s in sources:
            for g in s['fn_boxes']:cases[(g['stem'],g['cx'],g['cy'])]=g
        frames=[]
        for i,g in enumerate(cases.values(),1):
            src=DATA/'images'/split/f'{g["stem"]}.png';im=Image.open(src).convert('RGB')
            x0,y0=int(g['cx'])-22,int(g['cy'])-22
            patch=im.crop((x0,y0,x0+44,y0+44)).resize((220,220),Image.Resampling.NEAREST)
            row=Image.new('RGB',(660,265),'white');d=ImageDraw.Draw(row)
            row.paste(patch,(0,25));d.text((3,3),f'{split} {i} clean',fill='black',font=font)
            panels=[]
            for col,s in enumerate(sources,1):
                path=ROOT/(s['path'] if 'path' in s else s['predictions'])
                t=pd.read_csv(path);thr=s['thr'] if 'thr' in s else s['threshold'];t=t[(t.stem==g['stem'])&(t.score>=thr)]
                t=t[((t.cx-g['cx'])**2+(t.cy-g['cy'])**2)**.5<18]
                p=patch.copy();dr=ImageDraw.Draw(p)
                for r,color in [(g,'#006dcc')]+[(r,'#f58b00') for r in t.to_dict(orient='records')]:
                    dr.rectangle([(r['cx']-r['w']/2-x0)*5,(r['cy']-r['h']/2-y0)*5,(r['cx']+r['w']/2-x0)*5,(r['cy']+r['h']/2-y0)*5],outline=color,width=2)
                name=s.get('name','YOLO11n' if 'yolo11n' in s['run'] else 'YOLO26n')
                row.paste(p,(col*220,25));d.text((col*220+3,3),name,fill='black',font=font)
                panels.append(dict(model=name,predictions=t.to_dict(orient='records')))
            d.text((3,247),'Blue = official TXT. Orange = model. Clean image at left.',fill='black',font=font)
            dest=out/f'{split}_{i:02}.png';row.save(dest);frames.append(row)
            records.append(dict(split=split,id=i,gt=g,source=str(src.relative_to(ROOT)),source_sha256=sha256(src),figure=str(dest.relative_to(ROOT)),panels=panels))
        for off in range(0,len(frames),4):
            sheet=Image.new('RGB',(660,265*len(frames[off:off+4])),'white')
            for j,row in enumerate(frames[off:off+4]):sheet.paste(row,(0,265*j))
            sheet.save(out/f'{split}_sheet{off//4+1}.png')
    atomic_json(out/'cases.json',records);print(dict(cases=len(records),folder=str(out.relative_to(ROOT))))


if __name__=='__main__':main()
