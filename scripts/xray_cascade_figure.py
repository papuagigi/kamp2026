"""Render actual validation localization failures and frozen model boxes."""
import json

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from xray_cascade_audit import OUT, NAMES
from xray_config import DATA, ROOT
from xray_eval import iou
from xray_selected_common import REPORT, RUNS


def main():
    summary = json.loads((OUT/'summary.json').read_text())
    policy = json.loads((OUT/'policy.json').read_text())
    preds = {r:pd.read_csv(REPORT/'diagnostics'/f'{r}_benchmark_validation.csv') for r in RUNS}
    rows = summary['first_fn_details']
    font = ImageFont.truetype('/System/Library/Fonts/AppleSDGothicNeo.ttc', 18)
    small = ImageFont.truetype('/System/Library/Fonts/AppleSDGothicNeo.ttc', 15)
    canvas = Image.new('RGB',(1220,70+len(rows)*320),'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((12,8),'실제 검증 사진 | 파랑: 공식 TXT 정답 박스 · 주황: 모델 예측 박스',font=font,fill='black')
    draw.text((12,35),'화면은 4배 확대. IoU 0.5 위치 오류와 실제 이물 표시 누락은 다릅니다.',font=font,fill='black')
    details=[]
    for rowidx,g in enumerate(rows):
        top=70+rowidx*320
        draw.text((12,top),g['stem'],font=font,fill='black')
        x0,y0=int(g['cx'])-30,int(g['cy'])-30
        source=Image.open(DATA/'images/val'/f'{g["stem"]}.png').convert('RGB')
        patch=source.crop((x0,y0,x0+60,y0+60)).resize((240,240),Image.Resampling.NEAREST)
        for col,label in enumerate(['선 없는 사진']+[NAMES[r] for r in RUNS]):
            left=col*244
            canvas.paste(patch,(left,top+49))
            draw.text((left+6,top+27),label,font=font,fill='black')
            if not col:continue
            run=RUNS[col-1]
            p=preds[run]
            selected=p[(p.stem==g['stem'])&(p.score>=policy['thresholds'][run])].copy()
            selected['distance']=((selected.cx-g['cx'])**2+(selected.cy-g['cy'])**2)**.5
            selected=selected[selected.distance<=30]
            def rect(cx,cy,w,h,color):
                draw.rectangle((left+4*(cx-w/2-x0),top+49+4*(cy-h/2-y0),
                                left+4*(cx+w/2-x0),top+49+4*(cy+h/2-y0)),outline=color,width=2)
            rect(g['cx'],g['cy'],g['w'],g['h'],'#1264c5')
            for p in selected.itertuples():rect(p.cx,p.cy,p.w,p.h,'#dd6800')
            if len(selected):
                p=selected.sort_values('distance').iloc[0]
                overlap=iou([g[k] for k in ['cx','cy','w','h']],p[['cx','cy','w','h']].values)
                draw.text((left+5,top+292),f'점수 {p.score:.3f} / IoU {overlap:.3f}',font=small,fill='black')
                details.append(dict(stem=g['stem'],model=NAMES[run],score=float(p.score),iou=float(overlap)))
    path=ROOT/'docs/figures/common20_cascade_validation_examples.png'
    canvas.save(path)
    (OUT/'illustrated_examples.json').write_text(json.dumps(details,ensure_ascii=False,indent=2)+'\n')
    print(path)


if __name__=='__main__':main()
