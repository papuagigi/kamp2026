#!/usr/bin/env python3
"""
색 네모가 없는 라벨 사진 사본이 데이터 폴더 어딘가에 있는지 찾는다.
- 데이터 폴더의 모든 그림 파일(bmp, jpg, png)을 열어 보고, 공식 라벨 500장과 같은 사진인지 확인한다.
  같은 사진인지는 이름이 같거나, 크기가 같고 표시 밖 픽셀이 99% 넘게 같은 것으로 판단한다.
- 사본마다 검사기 표시 픽셀 수를 센다. 팔레트 이미지는 244~255번, 그 밖의 이미지는 RGB 차이 40 기준.
출력: reports/data_audit/copy_markers.csv, 화면에 요약
사용: .venv/bin/python scripts/xray_clean_copy_check.py
"""
import collections
from pathlib import Path
import numpy as np, cv2, pandas as pd
from PIL import Image
from xray_prepare import colored_mask, DS, RAW, LAB

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "data_audit"

def role(rel):
    if rel.startswith("test1/yolov3/X선이물검출기"): return "원본 사진"
    if rel.startswith("라벨링 6종 세트/images"): return "실습용 사본"
    if rel.startswith("test1/yolov3/images"): return "가이드북 예제 사진"
    if rel.startswith("test1/yolov3/result"): return "가이드북 탐지 결과"
    if rel.startswith("OpenLabeling-master"): return "라벨링 도구"
    return "기타"

def read(path):
    """(팔레트 번호 또는 회색 배열, 표시 픽셀 수, 모드). 그림이 아니면 None."""
    try:
        im = Image.open(path); im.load()
    except Exception:
        return None
    if im.mode == "P":
        idx = np.asarray(im)
        return idx, int((idx >= 244).sum()), "P"
    rgb = np.asarray(im.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), int(colored_mask(rgb).sum()), im.mode

def main():
    labeled = sorted(p.stem for p in LAB.glob("*.txt"))
    raw = {}
    for p in sorted(RAW.rglob("*.bmp")):
        raw.setdefault(p.stem, p)                      # 폴더 간 같은 사본은 첫 것만 기준으로
    ref = {}                                           # 라벨 사진: 팔레트 번호 배열
    for s in labeled:
        ref[s] = np.asarray(Image.open(raw[s]))
    by_shape = collections.defaultdict(list)
    for s, a in ref.items():
        by_shape[a.shape].append(s)

    rows = []
    files = sorted(p for p in DS.rglob("*") if p.is_file() and p.suffix.lower() in {".bmp", ".jpg", ".png"} and not p.name.startswith("._"))
    for p in files:
        rel = str(p.relative_to(DS)); r = read(p)
        if r is None:
            rows.append(dict(rel=rel, role=role(rel), is_image=False)); continue
        arr, mark_px, mode = r
        stem, match, same = p.stem, "", np.nan
        if stem in ref and arr.shape == ref[stem].shape:
            match = stem
        else:                                          # 이름이 달라도 내용이 같은 사진인지
            for s in by_shape.get(arr.shape, []):
                keep = ref[s] < 244
                if (arr[keep] == ref[s][keep]).mean() > 0.99:
                    match = s; break
        if match:
            keep = ref[match] < 244
            same = float((arr[keep] == ref[match][keep]).mean()) if mode == "P" else float((np.abs(arr[keep].astype(int) - ref[match][keep]) <= 2).mean())
        rows.append(dict(rel=rel, role=role(rel), is_image=True, mode=mode, width=arr.shape[1], height=arr.shape[0],
                         mark_px=mark_px, labeled_match=match, same_px_frac=same,
                         identical=bool(match and mode == "P" and np.array_equal(arr, ref[match]))))
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "copy_markers.csv", index=False, encoding="utf-8-sig")

    img = df[df.is_image]
    print(f"그림으로 열리는 파일 {len(img)}개 (확장자만 그림인 코드·설정 파일 {int((~df.is_image).sum())}개 제외)")
    print(img.groupby("role").agg(파일=("rel", "size"), 표시없음=("mark_px", lambda x: int((x == 0).sum()))).to_string())
    m = img[img.labeled_match != ""]
    print(f"\n공식 라벨 500장과 같은 사진인 파일 {len(m)}개, 해당하는 라벨 사진 {m.labeled_match.nunique()}장")
    print(m.groupby("role").agg(파일=("rel", "size"), 원본과_완전히_같음=("identical", "sum"), 표시없음=("mark_px", lambda x: int((x == 0).sum()))).to_string())
    clean = m[m.mark_px == 0]
    print(f"\n색 표시가 전혀 없는 라벨 사진 사본: {len(clean)}개")
    for r in clean.itertuples():
        print(f"  {r.rel}  (라벨 사진 {r.labeled_match}, 표시 밖 픽셀 일치 {r.same_px_frac:.3f})")
    per = m.groupby("labeled_match").mark_px.min()
    print(f"라벨 사진 가운데 표시 없는 사본이 하나라도 있는 사진: {int((per == 0).sum())} / {len(labeled)}장")

if __name__ == "__main__":
    main()
