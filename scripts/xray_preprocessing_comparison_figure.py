"""Display existing files from the same source; never change dataset pixels."""
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
FONT = Path('/System/Library/Fonts/AppleSDGothicNeo.ttc')
fontManager.addfont(str(FONT))
plt.rcParams.update({'font.family': FontProperties(fname=str(FONT)).get_name(),
                     'font.size': 12, 'axes.unicode_minus': False})


def boxes(p):
    return np.array([list(map(float, line.split())) for line in p.read_text().splitlines()
                     if line.strip()]).reshape(-1, 5)


def main():
    folder = next(p for p in (ROOT/'preprocessed_data').iterdir()
                  if p.is_dir() and unicodedata.normalize('NFC', p.name) == '전처리된데이터셋')/'X-ray_데이터셋'
    stem = '002_20200622_203053(2)'
    teamstem = 'h1_' + stem
    team = folder/'학습'
    ours = ROOT/'data/xray_roadmap_20261002'
    inputs = [
        (team/'images'/f'{teamstem}.png', team/'labels'/f'{teamstem}.txt', '전달 자료 · 색 네모 제거'),
        (team/'images'/f'{teamstem}_syn_random0.png', team/'labels'/f'{teamstem}_syn_random0.txt', '전달 자료 · 이물 추가(C)'),
        (team/'images'/f'{teamstem}_syn_edge_rot0.png', team/'labels'/f'{teamstem}_syn_edge_rot0.txt', '전달 자료 · 조각 변형 후 추가(D)'),
        (ours/'images/clean'/f'{stem}.png', ROOT/'data/xray_v2/labels/train'/f'{stem}.txt', '현재 자료 · 색 네모 제거'),
        (ours/'images/augment'/f'{stem}__crop.png', ours/'labels/augment'/f'{stem}__crop.txt', '현재 자료 · 사진 일부 자르고 확대'),
        (ours/'images/augment'/f'{stem}__rotate_p.png', ours/'labels/augment'/f'{stem}__rotate_p.txt', '현재 자료 · 사진 전체 +7.5° 회전'),
    ]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest()
              for image, label, _ in inputs for p in [image, label]}
    base_boxes = boxes(inputs[0][1])
    fig, axes = plt.subplots(2, 3, figsize=(12, 9))
    fig.subplots_adjust(left=.03, right=.98, top=.86, bottom=.095, hspace=.28, wspace=.07)
    for i, (ax, (image, label, title)) in enumerate(zip(axes.ravel(), inputs)):
        pixels = np.asarray(Image.open(image).convert('L'))
        h, w = pixels.shape
        ax.imshow(pixels, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
        for b in boxes(label):
            original = any(np.allclose(b, q, atol=1.1e-6, rtol=0) for q in base_boxes)
            color = '#df7200' if i in [1, 2] and not original else '#147fc1'
            ax.add_patch(Rectangle(((b[1]-b[3]/2)*w, (b[2]-b[4]/2)*h),
                                   b[3]*w, b[4]*h, fill=False, edgecolor=color, linewidth=1.4))
        ax.set_title(title, pad=7)
        ax.axis('off')
    fig.suptitle('같은 원본에서 출발해, 무엇을 바꾸었는가?', fontsize=22, y=.97)
    fig.text(.5, .925, '위: 제품은 그대로 두고 새 이물을 합성   /   아래: 제품·막대·이물을 함께 자르거나 회전',
             ha='center', fontsize=12)
    fig.text(.5, .045, '모두 실제 저장된 파일입니다. 파란 선·주황 선은 TXT를 보여 주기 위한 설명선입니다.\n'
             '전달 자료의 학습 안내에는 사진 전체 ±10° 회전도 있습니다. 이 그림은 저장 PNG만 비교합니다.',
             ha='center', fontsize=11)
    output = ROOT/'docs/figures/19_전달자료와_현재자료_동일원본비교.png'
    fig.savefig(output, dpi=150)
    plt.close(fig)
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in before.items())
    record = dict(figure=str(output.relative_to(ROOT)), same_source_stem=stem,
                  source_files=[dict(path=str(p.relative_to(ROOT)), sha256=d) for p, d in before.items()],
                  source_files_unchanged=True, only_display_overlays_added=True)
    out = ROOT/'reports/preprocessing_comparison_20261003'
    out.mkdir(parents=True, exist_ok=True)
    (out/'figure_sources.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
