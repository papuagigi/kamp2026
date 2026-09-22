#!/usr/bin/env python3
"""
X-ray 정식 데이터셋 준비 (v1).
- 정답 500장(라벨링 6종 세트/labels)을 마스킹해 PNG로 변환
- 촬영 묶음(같은 호기 60초 이내 연속) ID, 시편 여부, 촬영 시각, 이미지 크기를 태깅
- 촬영 묶음 단위로 학습/검증/평가 분할 (호기별로 고르게), 결과를 data/xray_v1/manifest.csv 에 저장
사용: .venv/bin/python scripts/xray_prepare.py [--rebuild]
"""
import argparse, re, shutil, collections, random
from pathlib import Path
from datetime import datetime
import numpy as np, cv2, pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "제조AI데이터셋" / "4. X-ray 검사장비 AI 데이터셋" / "dataset"
RAW = DS / "test1" / "yolov3" / "X선이물검출기(06.23_09.22)"
LAB = DS / "라벨링 6종 세트" / "labels"
OUT = ROOT / "data" / "xray_v1"
PAT = re.compile(r"^(\d{3})_(\d{8})_(\d{6})\((\d+)\)$")
ROUTINE_HOURS = {0, 4, 8, 12, 16, 20}

def colored_mask(rgb, thr=40):
    a = rgb.astype(int)
    return ((np.abs(a[:,:,0]-a[:,:,1]) + np.abs(a[:,:,1]-a[:,:,2]) + np.abs(a[:,:,0]-a[:,:,2])) > thr).astype(np.uint8)

def mask_image(path):
    """색 네모를 지우고 흑백 이미지를 돌려준다. (마스킹된 픽셀 수도 함께)"""
    rgb = np.asarray(Image.open(path).convert("RGB"))
    m = cv2.dilate(colored_mask(rgb), np.ones((5, 5), np.uint8))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return cv2.inpaint(gray, m, 3, cv2.INPAINT_TELEA), int(m.sum())

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--rebuild", action="store_true"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if OUT.exists() and not a.rebuild and (OUT / "manifest.csv").exists():
        print("[skip] 이미 있음:", OUT); return
    if OUT.exists(): shutil.rmtree(OUT)
    raw_map = {}
    for p in RAW.rglob("*.bmp"):
        raw_map.setdefault(p.stem, p)          # 폴더 간 중복은 첫 것만
        machine_folder = p.relative_to(RAW).parts[0][:3]
    rows = []
    for t in sorted(LAB.glob("*.txt")):
        m = PAT.match(t.stem)
        if not m or t.stem not in raw_map: continue
        p = raw_map[t.stem]
        dt = datetime.strptime(m.group(2) + m.group(3), "%Y%m%d%H%M%S")
        boxes = [list(map(float, l.split())) for l in t.read_text().splitlines() if l.strip()]
        with Image.open(p) as im: W, H = im.size
        rows.append(dict(stem=t.stem, path=str(p), machine=p.relative_to(RAW).parts[0][:3], prefix=m.group(1), dt=dt,
                         hour=dt.hour, n_boxes=len(boxes), width=W, height=H,
                         box_w_px=np.median([b[3]*W for b in boxes]) if boxes else 0))
    df = pd.DataFrame(rows).sort_values(["machine", "dt"]).reset_index(drop=True)
    # 촬영 묶음: 같은 호기에서 이전 사진과 60초 이내면 같은 묶음
    burst, cur = [], 0
    for i in range(len(df)):
        if i == 0 or df.machine[i] != df.machine[i-1] or (df.dt[i] - df.dt[i-1]).total_seconds() > 60: cur += 1
        burst.append(cur)
    df["burst_id"] = burst
    # 시편 여부: 박스 3개이거나, 4시간 루틴 시각(정각 ±20분)에 찍힌 사진
    minute_ok = df.dt.dt.minute <= 20
    df["is_testpiece"] = (df.n_boxes == 3) | (df.hour.isin(ROUTINE_HOURS) & minute_ok)
    # 촬영 묶음 단위 분할: 호기별로 묶음을 섞어 60/20/20
    rng = random.Random(a.seed); split = {}
    for mch, g in df.groupby("machine"):
        ids = sorted(g.burst_id.unique()); rng.shuffle(ids)
        n = len(ids); n_tr, n_va = int(n * 0.6), int(n * 0.2)
        for k, b in enumerate(ids): split[b] = "train" if k < n_tr else "val" if k < n_tr + n_va else "test"
    df["split"] = df.burst_id.map(split)
    # 이미지 변환
    masked_px = []
    for s in ["train", "val", "test"]:
        (OUT / "images" / s).mkdir(parents=True, exist_ok=True); (OUT / "labels" / s).mkdir(parents=True, exist_ok=True)
    for r in df.itertuples():
        img, n = mask_image(Path(r.path)); masked_px.append(n)
        cv2.imwrite(str(OUT / "images" / r.split / f"{r.stem}.png"), img)
        shutil.copy(LAB / f"{r.stem}.txt", OUT / "labels" / r.split / f"{r.stem}.txt")
    df["masked_px"] = masked_px
    df.to_csv(OUT / "manifest.csv", index=False, encoding="utf-8-sig")
    (OUT / "data.yaml").write_text(f"path: {OUT}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: defect\n")
    print("사진 수:", len(df), "| 촬영 묶음 수:", df.burst_id.nunique())
    print("분할별 사진 수:", df.split.value_counts().to_dict())
    print("분할별 시편/비시편:", df.groupby(["split", "is_testpiece"]).size().to_dict())
    print("분할별 호기:", df.groupby(["split", "machine"]).size().to_dict())
    print("마스킹 픽셀 중앙값:", int(np.median(masked_px)), "| 저장:", OUT)

if __name__ == "__main__":
    main()
