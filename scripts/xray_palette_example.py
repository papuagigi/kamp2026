#!/usr/bin/env python3
"""
팔레트 BMP 설명용 예시와 라벨 사진 500장의 밝기 통계를 출력한다 (docs/06 2-1, 3장에 쓰는 숫자).
- 예시 사진 한 장: 파일 크기 구성, 팔레트 표 일부, 금속구와 검사기 표시 주변의 픽셀 번호
- 라벨 사진 500장: 팔레트가 0~243 회색인지, 사진별 가장 밝은 값과 가장 흔한 값, 쓰인 표시 색
출력: reports/palette_example.txt (화면 출력과 같은 내용)
사용: .venv/bin/python scripts/xray_palette_example.py
"""
import io, contextlib
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
from xray_prepare import RAW, LAB

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "palette_example.txt"
STEM = "001_20200623_043227(1)"           # 3호기, 금속구 대비 46(라벨 사진 중앙값과 같음)
ROWS, COLS = (150, 161), (336, 350)       # 왼쪽 표시 선과 금속구가 함께 보이는 범위 (행, 열)
COLOR_NAME = {(96, 255, 96): "연두", (255, 127, 255): "분홍", (255, 0, 255): "자홍", (255, 255, 0): "노랑",
              (255, 0, 0): "빨강", (0, 255, 255): "하늘", (0, 255, 0): "초록", (0, 0, 255): "파랑"}

def first_copies():
    raw = {}
    for p in sorted(RAW.rglob("*.bmp")): raw.setdefault(p.stem, p)
    return raw

def main():
    raw = first_copies(); p = raw[STEM]
    im = Image.open(p); idx = np.asarray(im); H, W = idx.shape
    pal = np.array(im.getpalette()[:768]).reshape(-1, 3)
    size = p.stat().st_size
    print(f"[예시 사진] {STEM}  모드 {im.mode}, {W}×{H}")
    print(f"파일 크기 {size:,}바이트 = 픽셀 {W * H:,}개 × 1바이트 + 팔레트 표 {256 * 4:,}바이트 + 머리말 {size - W * H - 1024}바이트")
    print(f"같은 사진을 24비트 RGB BMP로 저장하면 {W * H * 3 + 54:,}바이트")
    print("팔레트 표 일부 (번호 → R, G, B):")
    for i in (0, 1, 2, 42, 88, 120, 210, 242, 243):
        print(f"  {i:3d} → {tuple(int(v) for v in pal[i])}")
    for i in range(244, 256):
        c = tuple(int(v) for v in pal[i]); print(f"  {i:3d} → {c} {COLOR_NAME.get(c, '')}")
    g = idx[idx < 244]; vals, cnt = np.unique(idx, return_counts=True)
    print(f"이 사진의 회색 값 범위 {g.min()}~{g.max()}, 가장 흔한 값 {vals[cnt.argmax()]} ({cnt.max():,}픽셀, 제품 밖 빈 곳)")
    print(f"이 사진에 쓰인 표시 번호: {', '.join(f'{v}({c}픽셀)' for v, c in zip(vals, cnt) if v >= 244)}")
    r0, r1 = ROWS; c0, c1 = COLS
    print(f"\n픽셀 번호 (행 {r0}~{r1 - 1}, 열 {c0}~{c1 - 1}). 254는 파랑 표시, 나머지는 회색 밝기")
    print("     " + " ".join(f"{c:>3d}" for c in range(c0, c1)))
    for r in range(r0, r1):
        print(f"{r:>4d} " + " ".join(f"{v:>3d}" for v in idx[r, c0:c1]))

    rows = []
    for s in sorted(q.stem for q in LAB.glob("*.txt")):
        im = Image.open(raw[s]); a = np.asarray(im); pl = np.array(im.getpalette()[:768]).reshape(-1, 3)
        gray = a[a < 244]; v, c = np.unique(gray, return_counts=True)
        rows.append(dict(stem=s, mode=im.mode,
                         gray_pal=bool((pl[:244] == np.arange(244)[:, None]).all()),
                         gmin=int(gray.min()), gmax=int(gray.max()), gmode=int(v[c.argmax()]),
                         marks=" ".join(str(x) for x in np.unique(a[a >= 244]))))
    df = pd.DataFrame(rows)
    print(f"\n[라벨 사진 {len(df)}장]")
    print(f"팔레트 모드 사진 {int((df['mode'] == 'P').sum())}장, 0~243번이 모두 (번호, 번호, 번호) 회색인 사진 {int(df.gray_pal.sum())}장")
    q = lambda s: f"최소 {s.min()}, 중앙값 {int(s.median())}, 최대 {s.max()}"
    print(f"사진별 가장 밝은 회색 값: {q(df.gmax)}")
    print(f"사진별 가장 흔한 회색 값(제품 밖 빈 곳): {q(df.gmode)}")
    print(f"사진별 가장 어두운 회색 값: {q(df.gmin)}")
    print(f"243번(가장 밝은 회색)을 쓴 사진: {int((df.gmax == 243).sum())}장")
    used = pd.Series(" ".join(df.marks).split()).astype(int).value_counts().sort_index()
    print("표시 번호별로 그 번호가 나온 사진 수: " + ", ".join(f"{k}번 {v}장" for k, v in used.items()))

if __name__ == "__main__":
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): main()
    text = buf.getvalue(); print(text, end="")
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(text, encoding="utf-8")
