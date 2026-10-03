"""Draw source-grounded model explanations. No inference or dataset writes.

The diagrams are original explanatory schematics, not paper figure copies,
network activation maps, attention measurements, or model predictions.
Source identifiers are defined in docs/20, section 2.7.
"""
from pathlib import Path
import hashlib
import json
import os

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/xray-model-guide-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/figures'
AUDIT = ROOT / 'reports/model_explanation_20261003'
PHOTO = ROOT / 'data/xray_v2/images/train/002_20200622_203053(2).png'
FONT = Path('/System/Library/Fonts/AppleSDGothicNeo.ttc')
if FONT.exists():
    fontManager.addfont(str(FONT))
    family = FontProperties(fname=str(FONT)).get_name()
else:
    family = 'sans-serif'
plt.rcParams.update({'font.family': family, 'font.size': 13,
                     'axes.unicode_minus': False, 'svg.fonttype': 'path'})
INK, MUTED = '#192E42', '#526779'
BLUE, TEAL, ORANGE, PURPLE = '#DCEBFA', '#DDF2EE', '#FBE8CF', '#EAE2F7'
STROKE = '#9BAFBD'
items = []


def text(ax, x, y, value, size=14, color=INK, ha='left', va='top', weight='normal'):
    return ax.text(x, y, value, fontsize=size, color=color, ha=ha, va=va,
                   weight=weight, linespacing=1.5, transform=ax.transAxes)


def box(ax, x, y, w, h, title, body='', face=BLUE, title_size=17, body_size=13):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.009',
                              facecolor=face, edgecolor=STROKE, linewidth=.8,
                              transform=ax.transAxes))
    if body:
        text(ax, x+w/2, y+h*.76, title, title_size, ha='center', weight='bold')
        text(ax, x+w/2, y+h*.42, body, body_size, ha='center')
    else:
        text(ax, x+w/2, y+h/2, title, title_size, ha='center', va='center', weight='bold')


def arrow(ax, start, end, color=MUTED, style='-|>', lw=1.8):
    ax.annotate('', xy=end, xytext=start, xycoords='axes fraction',
                arrowprops={'arrowstyle': style, 'lw': lw, 'color': color,
                            'shrinkA': 2, 'shrinkB': 2})


def path_arrow(ax, points):
    ax.plot([p[0] for p in points[:-1]], [p[1] for p in points[:-1]],
            color=MUTED, lw=1.8, transform=ax.transAxes)
    arrow(ax, points[-2], points[-1])


def canvas(title, subtitle, h=8.5):
    fig, ax = plt.subplots(figsize=(15.5, h))
    fig.subplots_adjust(left=.025, right=.975, bottom=.045, top=.965)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    text(ax, .025, .97, title, 25, weight='bold')
    text(ax, .025, .907, subtitle, 14, MUTED)
    return fig, ax


def photo(fig, ax, x, y, w, h, patches=False):
    p = ax.inset_axes([x, y, w, h])
    p.imshow(Image.open(PHOTO), cmap='gray', interpolation='nearest')
    p.axis('off')
    if patches:
        for t in (.25, .5, .75):
            p.plot([t,t], [0,1], transform=p.transAxes, color='#24B6D0', lw=1.4)
            p.plot([0,1], [t,t], transform=p.transAxes, color='#24B6D0', lw=1.4)


def footer(ax, source, note='출처를 참고해 재구성한 개념도입니다. 실제 모델 출력은 아닙니다.'):
    text(ax, .025, .082, note, 12, MUTED)
    text(ax, .025, .045, source, 11, MUTED)


def save(fig, stem, sources):
    paths = []
    for ext in ('png', 'svg'):
        p = OUT / (stem + '.' + ext)
        fig.savefig(p, dpi=145, facecolor='white')
        paths.append(str(p.relative_to(ROOT)))
    plt.close(fig)
    items.append({'files': paths, 'kind': 'original explanatory schematic', 'sources': sources})


def overview():
    fig, ax = canvas('세 모델은 이물을 어떻게 찾는가', '같은 X-ray 사진을 입력합니다. 특징을 처리하는 방식이 다릅니다.', 8.3)
    rows = [
        (.67, 'YOLOv8n', 'CNN · 한 단계 탐지', ['CNN 특징 추출','여러 크기 특징 결합','박스·점수 예측','중복 박스 정리']),
        (.43, 'Faster R-CNN', 'CNN · 두 단계 탐지', ['ResNet50 + FPN','후보 영역 생성','RoIAlign + 재판단','중복 박스 정리']),
        (.19, 'RF-DETR-S', 'Vision Transformer', ['DINOv2 ViT 특징','특징 변환·후보 준비','탐지 decoder','박스·점수 선택']),
    ]
    for y, name, sub, labels in rows:
        text(ax, .028, y+.125, name, 20, weight='bold')
        text(ax, .028, y+.067, sub, 12, MUTED)
        for i, title in enumerate(labels):
            x = .23 + i*.19
            box(ax, x, y, .165, .16, title, face=[BLUE,TEAL,ORANGE,PURPLE][i],title_size=15)
            if i < 3: arrow(ax, (x+.169,y+.08), (x+.185,y+.08))
    footer(ax, '근거: [S2] Ultralytics · [S3–S6] Faster R-CNN / ResNet / FPN / torchvision · [S9] RF-DETR')
    save(fig, '20_구조비교_출처기반', ['S2','S3','S4','S5','S6','S9'])


def yolo():
    fig, ax = canvas('YOLOv8n: 특징을 합치고 박스와 점수를 예측합니다', 'n은 nano 크기입니다. 이번 프로젝트의 가벼운 기준 모델입니다.')
    photo(fig, ax, .02, .60, .15, .21)
    text(ax, .095, .575, '색 네모 제거 사진', 12, ha='center')
    for x, title, body, face in [
        (.215,'Backbone','Conv · C2f · SPPF\n사진의 특징을 추출',BLUE),
        (.425,'Neck','위·아래 경로로\n여러 크기 특징 결합',TEAL),
        (.635,'Detect head','위치 분기 + 종류 분기\n박스와 이물 점수 예측',ORANGE),
        (.845,'NMS','겹친 후보를 정리\n최종 예측 박스',PURPLE),
    ]:
        box(ax,x,.625,.135 if x>.8 else .16,.18,title,body,face=face,title_size=17,body_size=12)
    for a,b in [(.175,.205),(.38,.415),(.59,.625),(.80,.835)]: arrow(ax,(a,.715),(b,.715))
    box(ax,.035,.17,.29,.31,'C2f · SPPF','C2f: 여러 경로의 특징을 합칩니다.\nSPPF: 주변 범위가 다른 특징을\n효율적으로 모읍니다.',BLUE,17,13)
    box(ax,.355,.17,.29,.31,'세 해상도에서 탐지','512 입력 설정의 정사각형 예시\nP3: 64×64 / P4: 32×32\nP5: 16×16 특징 위치',TEAL,17,13)
    box(ax,.675,.17,.29,.31,'Anchor-free','미리 정한 폭·높이의 기준 박스 없이\n특징 위치에서 박스를 예측합니다.\n격자 위치 자체가 없다는 뜻은 아닙니다.',ORANGE,17,12)
    footer(ax,'근거: [S2] Ultralytics 공식 구조·구현. 실제 입력 사진: 기본 학습 데이터셋의 1장.')
    save(fig,'20_YOLOv8n_구조설명',['S2'])


def faster():
    fig, ax = canvas('Faster R-CNN: 후보 영역을 찾고 다시 판단합니다', 'ResNet50-FPN v2는 CNN 기반의 두 단계 객체 탐지 모델입니다.')
    photo(fig, ax,.025,.625,.14,.19)
    text(ax,.095,.605,'색 네모 제거 사진',12,ha='center')
    boxes=[(.205,'ResNet50 + FPN','사진 특징 추출\n여러 해상도 특징 결합',BLUE),
           (.435,'RPN','이물이 있을 법한\n후보 영역 생성',TEAL),
           (.665,'RoIAlign','후보 좌표에 맞춰\n특징을 같은 크기로 모음',ORANGE)]
    for x,t,b,c in boxes:box(ax,x,.64,.195,.17,t,b,c,17,12)
    path_arrow(ax,[(.302,.822),(.302,.86),(.762,.86),(.762,.822)])
    text(ax,.53,.884,'같은 특징을 재사용',11,ha='center')
    for a,b in [(.17,.195),(.405,.425),(.635,.655)]:arrow(ax,(a,.724),(b,.724))
    path_arrow(ax,[(.862,.724),(.92,.724),(.92,.49),(.785,.49)])
    box(ax,.565,.37,.22,.22,'영역별 판단','이물 / 배경 분류\n박스 위치·크기 보정',ORANGE,18,14)
    arrow(ax,(.555,.48),(.455,.48))
    box(ax,.23,.37,.22,.22,'NMS + 점수 선택','중복 후보 정리\n최종 예측 박스 출력',PURPLE,18,14)
    text(ax,.033,.275,'1단계: RPN이 후보를 만듭니다.   2단계: 모은 특징으로 후보를 다시 판단합니다.',16,weight='bold')
    text(ax,.033,.213,'RoIAlign은 원본 사진을 다시 자르는 증강이 아닙니다. 특징 지도에서 후보 영역의 정보를 모읍니다.',13)
    text(ax,.033,.167,'현재 코드의 anchor 기준 크기: 8 · 16 · 32 · 64 · 128픽셀. 작은 이물을 고려한 설정입니다.',13)
    footer(ax,'근거: [S3] Ren et al. · [S5] Lin et al. · [S6] torchvision 0.23 · scripts/xray_model_io.py')
    save(fig,'20_FasterRCNN_구조설명',['S3','S5','S6'])


def resnet_fpn():
    fig, ax = canvas('ResNet50과 FPN은 서로 다른 일을 합니다', 'ResNet50은 특징을 추출합니다. FPN은 해상도가 다른 특징을 결합합니다.', 9)
    ax.plot([.505,.505],[.15,.84],color='#D6E0E7',lw=1,transform=ax.transAxes)
    text(ax,.035,.82,'ResNet: 입력을 더하는 지름길',20,weight='bold')
    box(ax,.045,.58,.105,.12,'입력 x',face=BLUE,title_size=16)
    box(ax,.22,.545,.17,.19,'잔차 F(x)','1×1 → 3×3 → 1×1\n합성곱으로 새 특징 계산',BLUE,16,11)
    arrow(ax,(.155,.64),(.21,.64))
    ax.add_patch(Circle((.443,.64),.025,fc=TEAL,ec=STROKE,transform=ax.transAxes))
    text(ax,.443,.64,'+',21,ha='center',va='center',weight='bold')
    arrow(ax,(.393,.64),(.416,.64))
    path_arrow(ax,[(.098,.71),(.098,.765),(.443,.765),(.443,.67)])
    text(ax,.237,.775,'입력을 직접 전달',12,ha='center').set_bbox(
        dict(facecolor='white',edgecolor='none',pad=1.5))
    text(ax,.03,.465,'출력의 핵심 계산: x + F(x)',20,weight='bold')
    text(ax,.03,.397,'추가 변화를 학습할 경로를 만듭니다.\n깊은 신경망의 학습을 돕습니다.\n크기가 다르면 입력 변환이 추가됩니다.',14)
    text(ax,.03,.235,'그림은 ResNet50의 병목 블록을 단순화했습니다.\n정규화·활성화 층은 생략했습니다.',12,MUTED)
    text(ax,.54,.82,'FPN: 세부 위치와 넓은 문맥을 결합',20,weight='bold')
    text(ax,.565,.76,'ResNet 특징',13,weight='bold')
    text(ax,.827,.76,'FPN 특징',13,weight='bold')
    for idx,(c,p,y,w) in enumerate([('C5','P5',.65,.08),('C4','P4',.515,.105),('C3','P3',.38,.13),('C2','P2',.245,.155)]):
        x=.635-w/2; px=.865-w/2
        box(ax,x,y,w,.073,c,face=BLUE,title_size=15)
        box(ax,px,y,w,.073,p,face=TEAL,title_size=15)
        arrow(ax,(x+w+.006,y+.036),(px-.008,y+.036))
        if idx<3:
            arrow(ax,(.635,y-.043),(.635,y-.003))
            arrow(ax,(.865,y-.006),(.865,y-.057))
    text(ax,.713,.568,'옆 연결',11,ha='center')
    text(ax,.952,.52,'확대\n+\n결합',12,ha='center')
    text(ax,.54,.18,'P2는 더 촘촘한 위치 정보를 가집니다.\n그림의 네모는 사진이 아닌 특징 지도를 뜻합니다.',12)
    footer(ax,'근거: [S4] He et al., Fig. 2 / Table 1 · [S5] Lin et al., Fig. 3 / §3')
    save(fig,'20_ResNet50_FPN_역할',['S4','S5'])


def rfdetr():
    fig, ax = canvas('RF-DETR-S: 패치의 관계로 특징을 만들고 이물을 찾습니다', 'DINOv2로 사전학습한 Vision Transformer를 이물 탐지에 맞춰 추가 학습합니다.',9)
    photo(fig,ax,.027,.615,.14,.20,patches=True)
    text(ax,.095,.595,'패치 개념도',12,ha='center')
    for x,t,b,c in [(.21,'패치 표현','패치를 숫자 벡터로 변환\n위치 정보 추가',BLUE),
                    (.465,'DINOv2 ViT','작은 영역 안의 관계와\n넓은 영역의 관계를 처리',TEAL),
                    (.72,'Projector','탐지 decoder가 사용할\n특징 형식으로 변환',TEAL)]:
        box(ax,x,.64,.225,.17,t,b,c,18,13)
    for a,b in [(.174,.20),(.44,.455),(.695,.71)]:arrow(ax,(a,.725),(b,.725))
    path_arrow(ax,[(.84,.63),(.84,.545),(.64,.545),(.64,.495)])
    box(ax,.485,.285,.30,.20,'탐지 decoder','후보 표현(query)이 특징을 참고\n이물 위치와 점수를 단계적으로 보정',ORANGE,18,13)
    arrow(ax,(.79,.385),(.833,.385))
    box(ax,.845,.30,.13,.17,'출력','박스 좌표\n이물 점수',PURPLE,17,13)
    box(ax,.028,.285,.37,.20,'후보 표현(query)','물체 후보를 나타내는 숫자 벡터입니다.\n사용자가 입력하는 문장 질문이 아닙니다.',BLUE,17,13)
    arrow(ax,(.405,.385),(.475,.385))
    text(ax,.035,.217,'현재 설정: 패치 16×16픽셀 · decoder 3층 · 기본 query 300개 · 이물 1종',14,weight='bold')
    text(ax,.035,.168,'패치 크기는 탐지 가능한 이물의 최소 크기가 아닙니다. 그림의 격자 수는 설명을 위해 줄였습니다.',12)
    footer(ax,'근거: [S7] ViT · [S8] DINOv2 · [S9] RF-DETR Fig. 2 · 저장된 RFDETRSmall 설정')
    save(fig,'20_RFDETRS_구조설명',['S7','S8','S9'])


def main():
    OUT.mkdir(parents=True,exist_ok=True); AUDIT.mkdir(parents=True,exist_ok=True)
    original_hash=hashlib.sha256(PHOTO.read_bytes()).hexdigest()
    overview(); yolo(); faster(); resnet_fpn(); rfdetr()
    assert hashlib.sha256(PHOTO.read_bytes()).hexdigest()==original_hash
    evidence={'input_photo':str(PHOTO.relative_to(ROOT)),'input_sha256':original_hash,
              'input_unchanged':True,'model_inference_executed':False,'figures':items}
    (AUDIT/'figure_manifest.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'figure_count':len(items),'formats':['png','svg'],'input_unchanged':True},ensure_ascii=False))


if __name__=='__main__':
    main()
