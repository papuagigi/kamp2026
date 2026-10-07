"""Draw the approved full-frame inspection flow for the report.

This script only creates an explanatory figure. It does not change decisions.
"""
from pathlib import Path
from math import atan2, cos, sin
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FONT = Path('/System/Library/Fonts/AppleSDGothicNeo.ttc')
OUT = ROOT / 'docs/figures/20_전체사진_이중검사.png'


def main():
    image = Image.new('RGB', (1600, 1100), 'white')
    draw = ImageDraw.Draw(image)

    def text(x, y, value, size=34):
        font = ImageFont.truetype(str(FONT), size)
        for line in value.split('\n'):
            draw.text((x, y), line, font=font, fill='#162832', anchor='mt')
            y += size * 1.35

    def box(bounds, value, fill='#EDF3F6'):
        draw.rounded_rectangle(bounds, radius=16, fill=fill,
                               outline='#90A4AE', width=3)
        text((bounds[0] + bounds[2]) / 2, bounds[1] + 18, value)

    def arrow(points):
        draw.line(points, fill='#56727F', width=4)
        x, y = points[-1]
        px, py = points[-2]
        angle = atan2(y - py, x - px)
        base_x, base_y = x - 16 * cos(angle), y - 16 * sin(angle)
        draw.polygon([(x, y),
                      (base_x + 10 * sin(angle), base_y - 10 * cos(angle)),
                      (base_x - 10 * sin(angle), base_y + 10 * cos(angle))],
                     fill='#56727F')

    text(800, 15, '같은 전체 사진을 두 모델이 각각 검사', 43)
    box((380, 85, 1220, 165), '① 입력 | 색 네모를 제거한 전체 사진')
    arrow([(600,165),(600,190),(405,190),(405,220)])
    arrow([(1000,165),(1000,190),(1195,190),(1195,220)])
    box((55,225,755,350), '② 모델 | YOLO11s\n사진 전체에서 이물 후보 탐색')
    box((845,225,1545,350), '② 모델 | RF-DETR-S\n사진 전체에서 이물 후보 탐색')
    arrow([(405,350),(405,380),(800,380),(800,410)])
    arrow([(1195,350),(1195,380),(800,380),(800,410)])
    box((370,415,1230,495), '③ 프로그램 | 두 결과 보존')
    arrow([(800,495),(800,540)])
    box((220,545,1380,635), '④ 두 예측·검사 상태·현장 통과 조건 확인')
    for x in [275,800,1325]:
        arrow([(800,635),(800,680),(x,680),(x,730)])
    palettes=[('#E8F4ED','#2F7D3C'),('#FFF3E2','#D85D00'),('#FCEBED','#C62F38')]
    values=[('PASS','두 모델 모두 이물 미검출\n+ 현장 조건 모두 통과\n→ 제품 통과'),
            ('RE-INSPECTION','후보·불일치·불확실·오류\n또는 현장 조건 미충족\n→ 제품 보류·추가 확인'),
            ('REJECT','이물 확인\n→ 제품 분리·격리')]
    for x,(fill,line),(title,body) in zip([275,800,1325],palettes,values):
        draw.rounded_rectangle((x-245,735,x+245,945),radius=16,fill=fill,outline=line,width=4)
        font=ImageFont.truetype(str(FONT),40)
        draw.text((x,752),title,font=font,fill=line,anchor='mt')
        text(x,813,body,30)
    arrow([(800,945),(800,985)])
    box((490,990,1110,1075), '재촬영 또는 담당자 재검사', '#FFF9EF')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUT)
    print(OUT.relative_to(ROOT))


if __name__ == '__main__':
    main()
