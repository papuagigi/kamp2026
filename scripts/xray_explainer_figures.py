"""Render explanatory figures from real images and saved illustration transforms.

No dataset or label files are modified. Hypothetical review boxes are labeled.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/figures'
EVIDENCE = ROOT / 'reports/docs_consolidation_20261002'
FONT = '/System/Library/Fonts/AppleSDGothicNeo.ttc'
fontManager.addfont(FONT)
plt.rcParams.update({'font.family': FontProperties(fname=FONT).get_name(),
                     'axes.unicode_minus': False, 'font.size': 12})
BLUE, ORANGE, DARK = '#1469aa', '#b95018', '#26374a'


def save(fig, name):
    fig.savefig(OUT / name, dpi=170, facecolor='white')
    plt.close(fig)


def photo(ax, arr, title, caption='', boxes=None):
    ax.imshow(arr, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
    ax.set_axis_off()
    ax.set_title(title, fontsize=15, pad=13, color=DARK)
    if boxes is not None:
        for x1, y1, x2, y2 in boxes:
            ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False,
                                   edgecolor=BLUE, lw=1.5))
    if caption:
        ax.text(.5, -.07, caption, transform=ax.transAxes, ha='center',
                va='top', fontsize=12, color=DARK, linespacing=1.45)


def box(ax, x, y, w, h, text, face='#edf3fa', size=13):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.01',
                              fc=face, ec='none'))
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=size,
            color=DARK, linespacing=1.5)


def arrow(ax, a, b):
    ax.annotate('', xy=b, xytext=a,
                arrowprops={'arrowstyle': '->', 'color': '#657b8c', 'lw': 2})


def main():
    EVIDENCE.mkdir(exist_ok=True, parents=True)
    name = '002_20200622_203053(2)'
    row = pd.read_csv(ROOT/'data/xray_v2/manifest.csv').set_index('stem').loc[name]
    assert row['split'] == 'train'
    raw_path = ROOT/row['path']
    masked_path = ROOT/'data/xray_v2/images/train'/f'{name}.png'
    label_path = ROOT/'data/xray_v2/labels/train'/f'{name}.txt'
    before_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in [raw_path, masked_path, label_path]}
    raw = np.asarray(Image.open(raw_path).convert('RGB'))
    clean = np.asarray(Image.open(masked_path).convert('L'))
    height, width = clean.shape
    proto = ROOT/'reports/augmentation_options_study_20261002'
    trans = json.loads((proto/'prototype_transforms.json').read_text())
    boxes = np.array(trans['original_pixel_xyxy'])
    crop = np.asarray(Image.open(proto/'prototype_whole_image_crop.png'))
    rot = np.asarray(Image.open(proto/'prototype_whole_image_rotate10.png'))
    cb, rb = np.array(trans['crop_pixel_xyxy']), np.array(trans['rotation_pixel_xyxy'])

    # Exact same region and display scale in all panels; no contrast enhancement.
    fig, axs = plt.subplots(1, 3, figsize=(12, 5.4))
    fig.subplots_adjust(left=.035, right=.97, top=.78, bottom=.26, wspace=.13)
    x1, y1, x2, y2 = boxes[1]
    cx, cy = (x1+x2)/2, (y1+y2)/2
    for ax, arr, title, caption, bs in zip(
        axs, [raw, clean, clean],
        ['① 원본 사진 확대', '② 빨간 선만 지운 사진', '③ TXT 위치를 그려 본 모습'],
        ['빨간 테두리: 검사기 표시\n가운데 검은 점: 찾을 이물',
         '테두리 자리는 주변 밝기로 메움\n가운데 검은 점은 남아 있음',
         '파란 네모: TXT의 위치 설명\n실제 학습 사진에는 그리지 않음'],
        [None, None, boxes[1:2]]):
        photo(ax, arr, title, caption, bs)
        ax.set_xlim(cx-13, cx+13); ax.set_ylim(cy+13, cy-13)
    fig.suptitle('사진을 고치는 일과 정답 위치를 기록하는 일은 별개입니다',
                 fontsize=19, y=.96, color=DARK)
    fig.text(.045, .035, '같은 사진의 같은 부분입니다. 지운 선 아래의 원래 밝기는 알 수 없어 주변으로 추정해 메웁니다.',
             fontsize=11, color='#526476')
    save(fig, '19_색표시와_정답의_차이.png')

    # Show the chosen crop rectangle before its actual saved transformation.
    fig, axs = plt.subplots(1, 3, figsize=(12, 6.2))
    fig.subplots_adjust(left=.025, right=.975, top=.80, bottom=.25, wspace=.1)
    photo(axs[0], clean, '① 색 네모를 지운 사진', '검은 점 3개를 찾는 사진', boxes)
    photo(axs[1], clean, '② 남길 범위를 고릅니다', '노란 범위 안에 점 3개가 모두 있음\n범위 바깥의 여백·제품 일부는 잘라냄', boxes)
    ox, oy = trans['crop_origin']; f = trans['crop_side_fraction']
    cw, ch = f*width, f*height
    for x, y, w, h in [(0, 0, width, oy), (0, oy+ch, width, height-oy-ch),
                       (0, oy, ox, ch), (ox+cw, oy, width-ox-cw, ch)]:
        axs[1].add_patch(Rectangle((x,y),w,h,fc='white',alpha=.68,ec='none'))
    axs[1].add_patch(Rectangle((ox,oy),cw,ch,fill=False,ec='#c88200',lw=2.5,ls='--'))
    photo(axs[2], crop, '③ 잘라낸 사진을 확대합니다', '사진 속 점이 더 크게 보임\n점 위치를 적은 라벨 좌표도 함께 바꿈', cb)
    fig.suptitle('Crop: 이물이 모두 남도록 사진 일부를 잘라낸 뒤 확대하기', fontsize=18, y=.96, color=DARK)
    fig.text(.04,.035,'이 예시는 가로·세로의 80%가 남는 범위를 골랐습니다. 파란 네모·노란 선은 설명용입니다. 학습에는 아직 적용 전입니다.',fontsize=10.5,color='#526476')
    save(fig,'19_사진을_자르는_과정.png')

    fig, axs = plt.subplots(1, 2, figsize=(9, 5.8))
    fig.subplots_adjust(left=.04,right=.96,top=.80,bottom=.23,wspace=.14)
    photo(axs[0],clean,'① 회전 전','제품·막대·검은 점이 한 사진에 있음',boxes)
    photo(axs[1],rot,'② 사진 전체를 10° 회전','제품·막대·검은 점이 함께 돌아감\n파란 정답 박스도 새 위치로 이동',rb)
    fig.suptitle('회전: 이물 조각만 돌리는 것이 아니라 사진 전체를 돌립니다',fontsize=17,y=.97,color=DARK)
    fig.text(.045,.035,'설명용 변환 예시입니다. 원래 세 점을 유지하며 새 이물을 추가하지 않습니다. 파란 박스는 설명용입니다.',fontsize=10.5,color='#526476')
    save(fig,'19_사진전체_회전.png')

    # Deliberately illustrative review states, not detector outputs.
    fig, axs=plt.subplots(1,3,figsize=(12,6.0))
    fig.subplots_adjust(left=.025,right=.975,top=.77,bottom=.23,wspace=.1)
    bad=boxes.copy();bad[1]=boxes[1]+[-13,-10,13,10]
    captions=['점 3개에 박스 3개\n여러 AI 검사도 통과하면 자동 채택',
              '가운데 점에 박스가 빠져 있음\n누락 의심으로 사용자에게 전달',
              '가운데 박스가 막대까지 크게 포함\n크기 의심으로 사용자에게 전달']
    for ax,bs,title,cap in zip(axs,[boxes,boxes[[0,2]],bad],['맞게 붙인 경우의 예','하나를 놓친 경우의 예','너무 크게 붙인 경우의 예'],captions):
        photo(ax,clean,title,cap,bs)
        ax.set_xlim(boxes[:,0].min()-22,boxes[:,2].max()+55)
        ax.set_ylim(boxes[:,3].max()+12,boxes[:,1].min()-12)
    fig.suptitle('사람은 AI가 의심스럽다고 분류한 사진만 확인합니다',fontsize=19,y=.97,color=DARK)
    fig.text(.035,.89,'검수 방식 설명용 가상 박스입니다. 실제 AI가 예측한 결과가 아닙니다.',fontsize=12,color=ORANGE)
    fig.text(.04,.035,'검수 방법 설명용 예시입니다. 실제 처리 결과는 별도 검수 화면에서 확인합니다.',fontsize=11,color='#526476')
    save(fig,'19_AI라벨_검수예시.png')

    fig,ax=plt.subplots(figsize=(11,4.8));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
    ax.text(.03,.94,'비슷한 연속 사진은 한 묶음으로 함께 나눕니다',fontsize=19,color=DARK)
    for y,lhs,rhs,fc in [(.65,'같은 검사기·짧은 간격\nA1 · A2 · A3','학습용\nAI가 정답을 보며 배우는 자료','#e8f2fb'),
                        (.37,'다른 촬영 묶음\nB1 · B2','검증용\n학습 방법과 기준을 고르는 자료','#e6f2ea'),
                        (.09,'또 다른 촬영 묶음\nC1 · C2','테스트용\n선택한 모델의 결과를 확인하는 자료','#fff0df')]:
        box(ax,.03,y,.40,.20,lhs,fc);box(ax,.55,y,.42,.20,rhs,fc);arrow(ax,(.45,y+.1),(.53,y+.1))
    save(fig,'19_촬영묶음으로_나누기.png')

    fig,ax=plt.subplots(figsize=(12,6.4));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
    ax.text(.02,.955,'같은 사진을 보고 이물 위치를 찾는 세 가지 방식',fontsize=20,color=DARK)
    rows=[(.66,'YOLOv8n  |  CNN',['사진의 무늬·경계에서\n특징을 찾음','여러 위치에서\n이물 박스와 점수를 예측','빠른 기준 모델로 비교\n우리 데이터에서 성능 확인'],'#e8f2fb'),
          (.38,'Faster R-CNN  |  CNN',['사진에서\n의심 구역을 먼저 제안','구역별 특징을 모아\n이물인지 다시 판단','작은 점·배경 오탐을\n더 잘 구분하는지 비교'],'#e6f2ea'),
          (.10,'RF-DETR-S  |  Transformer',['사진 조각의 특징을\n읽어 들임','여러 위치의 관계를\n참고해 이물을 찾음','배경·촬영 조건 변화에\n도움이 되는지 비교'],'#fff0df')]
    for y,title,parts,fc in rows:
        ax.text(.025,y+.215,title,fontsize=15,color=DARK)
        for x,t in zip([.025,.365,.705],parts):box(ax,x,y,.27,.18,t,fc,size=12.5)
        arrow(ax,(.307,y+.09),(.352,y+.09));arrow(ax,(.647,y+.09),(.692,y+.09))
    ax.text(.025,.025,'이해를 위한 구조 요약입니다. 어떤 모델이 더 좋은지는 같은 데이터로 학습·평가한 뒤 정합니다.',fontsize=11,color='#526476')
    save(fig,'20_모델세가지_쉬운도식.png')

    for p,hsh in before_hashes.items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==hsh
    (EVIDENCE/'figure_sources.json').write_text(json.dumps({'source_files':before_hashes,
        'transforms':str((proto/'prototype_transforms.json').relative_to(ROOT)),
        'review_example':'hypothetical boxes on actual image; no inference',
        'original_txt_example':label_path.read_text().splitlines()[1],
        'generated_figures':['19_색표시와_정답의_차이.png','19_사진을_자르는_과정.png',
            '19_사진전체_회전.png','19_AI라벨_검수예시.png','19_촬영묶음으로_나누기.png',
            '20_모델세가지_쉬운도식.png']},ensure_ascii=False,indent=2))
    print('Generated 6 explanatory figures. Source image, v2 image and TXT hashes unchanged.')


if __name__=='__main__':
    main()
