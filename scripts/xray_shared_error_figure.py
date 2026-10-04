"""Show the validation example that flip agreement fails to flag."""
import json
import pandas as pd
from PIL import Image,ImageDraw,ImageFont
from xray_config import ROOT,DATA
from xray_shared_error_audit import OUT
from xray_selected_common import REPORT,RUNS
from xray_recovery import atomic_json,sha256


def main():
    summary=json.loads((OUT/'summary.json').read_text())
    case=next(x for x in summary['common_fn_cases'] if x['skipped_by_flip_gate'])
    frozen=json.loads((REPORT/'frozen_selection.json').read_text())['runs']
    directory=REPORT/'cascade_audit/consistency_native_csv'
    inputs=[('Original YOLO',directory/'original_val.csv',frozen[RUNS[0]]['thresholds']['iou50']),
            ('Flipped YOLO restored',directory/'hflip_val.csv',frozen[RUNS[0]]['thresholds']['iou50']),
            ('Faster R-CNN',REPORT/'diagnostics'/f'{RUNS[1]}_benchmark_validation.csv',frozen[RUNS[1]]['thresholds']['iou50']),
            ('RF-DETR-S',REPORT/'diagnostics'/f'{RUNS[2]}_benchmark_validation.csv',frozen[RUNS[2]]['thresholds']['iou50'])]
    path=DATA/'images/val'/f'{case["stem"]}.png';im=Image.open(path).convert('RGB')
    left,top=int(case['cx'])-20,int(case['cy'])-20;scale=5
    base=im.crop((left,top,left+40,top+40)).resize((200,200),Image.Resampling.NEAREST)
    canvas=Image.new('RGB',(1050,255),'white');draw=ImageDraw.Draw(canvas);font=ImageFont.load_default(size=14)
    draw.text((5,5),'Clean image',fill='black',font=font);canvas.paste(base,(5,29))
    boxes=[]
    for col,(name,source,threshold) in enumerate(inputs,1):
        t=pd.read_csv(source);t=t[(t.stem==case['stem'])&(t.score>=threshold)]
        t=t[((t.cx-case['cx'])**2+(t.cy-case['cy'])**2)**.5<15]
        patch=base.copy();d=ImageDraw.Draw(patch)
        def rect(cx,cy,w,h,color):
            d.rectangle([(cx-w/2-left)*scale,(cy-h/2-top)*scale,(cx+w/2-left)*scale,(cy+h/2-top)*scale],outline=color,width=2)
        rect(case['cx'],case['cy'],case['w'],case['h'],'#006dcc')
        for r in t.itertuples():rect(r.cx,r.cy,r.w,r.h,'#f58b00')
        draw.text((col*210+5,5),name,fill='black',font=font);canvas.paste(patch,(col*210+5,29))
        boxes.append(dict(model=name,predictions=t.to_dict(orient='records'),source=str(source.relative_to(ROOT)),sha256=sha256(source)))
    draw.text((5,234),'Blue: official TXT box. Orange: model prediction. Same physical area in every panel.',fill='black',font=font)
    target=OUT/'stable_error_example.png';canvas.save(target)
    atomic_json(OUT/'figure.json',dict(case=case,source=str(path.relative_to(ROOT)),source_sha256=sha256(path),
                                    figure_sha256=sha256(target),panels=boxes))
    print(target.relative_to(ROOT))


if __name__=='__main__':main()
