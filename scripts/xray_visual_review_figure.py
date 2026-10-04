"""Show inspected no-box and duplicate examples with decoded panel names."""
import json
from PIL import Image,ImageDraw,ImageFont
from xray_selected_common import REPORT,ROOT,RUNS


def main():
    folder=REPORT/'visual_review'
    cases={r['case']:r for r in json.loads((folder/'case_manifest.json').read_text())}
    names=dict(zip(RUNS,['YOLOv8n','Faster R-CNN','RF-DETR-S','D-FINE-S']))
    font=ImageFont.truetype('/System/Library/Fonts/AppleSDGothicNeo.ttc',18)
    out=Image.new('RGB',(1100,900),'white');d=ImageDraw.Draw(out)
    for i,(cid,caption) in enumerate([
        ('T221','RF-DETR-S는 채택된 박스가 없음. 다른 세 모델은 검은 중심을 표시함.'),
        ('T225','YOLOv8n은 채택된 박스가 없음. 다른 세 모델은 검은 중심을 표시함.'),
        ('T191','D-FINE-S는 같은 이물에 박스 두 개. 성공 1개와 중복 1개로 기록함.')]):
        r=cases[cid]
        with Image.open(ROOT/r['figure']) as im:im=im.crop((0,28,880,204)).resize((1100,220),Image.Resampling.NEAREST)
        y=300*i;d.text((8,y+2),cid+' | '+caption,font=font,fill='black')
        for j,label in enumerate(['선 없는 실제 사진']+[names[r['panels'][code]['run']] for code in 'ABCD']):
            d.text((220*j+8,y+32),label,font=font,fill='black')
        out.paste(im,(0,y+65))
    path=ROOT/'docs/figures/common20_visual_review_examples.png';out.save(path)
    print(str(path.relative_to(ROOT)))


if __name__=='__main__':main()
