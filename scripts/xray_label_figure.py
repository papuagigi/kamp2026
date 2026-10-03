"""Explain one official YOLO TXT file with its actual image and box coordinates.

Writes documentation assets only; source images, labels, and splits stay unchanged.
"""
import csv
import hashlib
import json
from pathlib import Path
import unicodedata

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
STEM = '002_20200622_203053(2)'
COLORS = ['#1469aa', '#b95018', '#187147']
FONT = '/System/Library/Fonts/AppleSDGothicNeo.ttc'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    with (ROOT/'data/xray_v2/manifest.csv').open(encoding='utf-8-sig') as f:
        row = next(r for r in csv.DictReader(f) if r['stem'] == STEM)
    assert row['split'] == 'train'
    source_dir = next(p for p in ROOT.iterdir()
                      if unicodedata.normalize('NFC', p.name) == '제조AI데이터셋')
    official = next(p for p in source_dir.rglob(STEM+'.txt')
                    if unicodedata.normalize('NFC', p.parent.parent.name) == '라벨링 6종 세트')
    clean_path = ROOT/'data/xray_v2/images/train'/f'{STEM}.png'
    train_label = ROOT/'data/xray_v2/labels/train'/f'{STEM}.txt'
    raw_path = ROOT/row['path']
    sources = [official, clean_path, train_label, raw_path, ROOT/'data/xray_v2/split.csv']
    before = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    assert official.read_bytes() == train_label.read_bytes()
    names = (official.parents[2]/'test1/yolov3/classes.names').read_text().splitlines()
    assert names == ['defect']
    lines = official.read_text().splitlines()
    values = np.array([[float(v) for v in line.split()] for line in lines])
    assert values.shape == (3, 5) and np.all(values[:, 0] == 0)
    clean = np.array(Image.open(clean_path).convert('L'))
    height, width = clean.shape
    assert (width, height) == (352, 332)
    pixels = values[:, 1:] * np.array([width, height, width, height])
    xyxy = np.column_stack([pixels[:, :2]-pixels[:, 2:]/2,
                           pixels[:, :2]+pixels[:, 2:]/2])

    examples = ROOT/'docs/examples'
    examples.mkdir(parents=True, exist_ok=True)
    examples.joinpath(official.name).write_bytes(official.read_bytes())
    examples.joinpath(clean_path.name).write_bytes(clean_path.read_bytes())
    assert digest(examples/official.name) == digest(official)
    assert digest(examples/clean_path.name) == digest(clean_path)

    fontManager.addfont(FONT)
    plt.rcParams.update({'font.family': FontProperties(fname=FONT).get_name(),
                         'axes.unicode_minus': False, 'font.size': 12})
    fig = plt.figure(figsize=(14, 8.5), facecolor='white')
    fig.text(.04, .95, '사진 속 이물 3개 = TXT 파일의 3줄', fontsize=23, weight='bold', color='#26374a')
    fig.text(.04, .91, STEM + '  |  실제 사진과 공식 TXT를 그대로 짝지은 예시', fontsize=12, color='#526476')
    whole = fig.add_axes([.04, .43, .29, .42])
    zoom = fig.add_axes([.35, .43, .18, .42])
    for ax in [whole, zoom]:
        ax.imshow(clean, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
        for i, ((x1, y1, x2, y2), color) in enumerate(zip(xyxy, COLORS), 1):
            ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, ec=color, lw=1.6))
            ax.text(238 if ax is zoom else 235, (y1+y2)/2, ['①','②','③'][i-1],
                    color=color, fontsize=15, va='center',
                    bbox={'facecolor':'white', 'alpha':.92, 'edgecolor':'none', 'pad':1})
        ax.set_axis_off()
    whole.set_title('색 표시를 지운 실제 사진\n가로 352 × 세로 332픽셀', fontsize=13, pad=12)
    zoom.set_title('이물 부분 확대\n번호 = TXT의 줄 순서', fontsize=13, pad=12)
    zoom.set_xlim(198, 258)
    zoom.set_ylim(179, 102)
    fig.text(.57, .82, '② 둘째 줄을 사진 위에 표시하면', fontsize=17, color=COLORS[1], weight='bold')
    fig.text(.57, .765, '0  →  이물(defect)의 종류 번호', fontsize=14, color='#26374a')
    fig.text(.57, .705, '중심 가로: 0.619318… × 352 = 218픽셀\n중심 세로: 0.418674… × 332 = 139픽셀',
             fontsize=14, color='#26374a', linespacing=1.8, va='top')
    fig.text(.57, .58, '박스 너비: 0.022727… × 352 = 8픽셀\n박스 높이: 0.024096… × 332 = 8픽셀',
             fontsize=14, color='#26374a', linespacing=1.8, va='top')
    fig.text(.57, .435, '왼쪽 위가 출발점입니다.\n가로는 오른쪽으로, 세로는 아래쪽으로 잽니다.',
             fontsize=12, color='#526476', linespacing=1.5)

    fig.text(.04, .355, '실제 TXT 내용 — 숫자의 순서는 아래와 같습니다', fontsize=16, weight='bold', color='#26374a')
    fig.text(.04, .313, '종류 번호   중심 가로 위치   중심 세로 위치   박스 너비   박스 높이', fontsize=13, color='#526476')
    for i, (line, color) in enumerate(zip(lines, COLORS)):
        y = .25 - i*.062
        fig.text(.04, y, ['①','②','③'][i], color=color, fontsize=16)
        fig.text(.075, y, line, family='DejaVu Sans Mono', fontsize=11.4, color=color)
    fig.text(.04, .045, '번호·색 박스는 설명용이며 실제 TXT와 학습 사진에는 없습니다. TXT의 위치·크기는 사진 전체에 대한 비율(0~1)입니다.',
             fontsize=11.5, color='#526476')
    out = ROOT/'docs/figures/19_TXT라벨_사진과_좌표.png'
    fig.savefig(out, dpi=170, facecolor='white')
    plt.close(fig)
    assert all(digest(ROOT/p) == h for p, h in before.items())
    evidence = {
        'source_sha256': before, 'width': width, 'height': height,
        'class_names': names, 'txt_lines': lines,
        'pixel_cx_cy_width_height': pixels.tolist(), 'pixel_xyxy': xyxy.tolist(),
        'figure': str(out.relative_to(ROOT)),
        'txt_copy': str((examples/official.name).relative_to(ROOT)),
        'image_copy': str((examples/clean_path.name).relative_to(ROOT)),
        'source_unchanged': True, 'copies_byte_identical': True,
        'annotation_note': 'Numbered boxes are rendered from official TXT, not model predictions.',
    }
    report = ROOT/'reports/docs_consolidation_20261002/label_example.json'
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(evidence, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(evidence, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
