#!/usr/bin/env python3
"""
'마스킹 흔적을 배우는가' 검증용 사진 만들기.
val 원본 사진의 빈자리(정답 이물에서 25px 이상 떨어진 제품 안쪽)에 가짜 색 네모를 그린 뒤,
학습 때와 똑같은 마스킹을 적용한다. 결과 사진에는 진짜 이물 자리와 가짜 자리 모두에 같은 흔적이 남는다.
탐지기가 가짜 자리를 이물로 잡으면 흔적을 배운 것이다.
출력: data/<데이터>/images/val_fakemask/*.png, data/<데이터>/fake_positions.csv (stem, cx, cy, kind=fake|control)
데이터는 환경변수 XRAY_DATA(기본 xray_v2). v2에서는 가짜 네모를 팔레트 255번(빨강)으로 그리고 v2 방식(팔레트 마스킹+나비에-스토크스)으로 지운다.
"""
import random, csv, os
from pathlib import Path
import numpy as np, cv2, pandas as pd
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
from xray_config import DATA, select_device
V2 = DATA.name != "xray_v1"
from xray_prepare import mask_image, colored_mask
from xray_eval import product_mask
from xray_prepare_v2 import load_palette, mask_palette

def main():
    rng = random.Random(0)
    man = pd.read_csv(DATA / "manifest.csv", encoding="utf-8-sig"); man = man[man.split == "val"]
    out_dir = DATA / "images" / "val_fakemask"; out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in man.itertuples():
        if V2:   # 팔레트 번호 배열 그대로 (0~243 회색, 244~255 표시)
            idx = load_palette(ROOT / r.path).copy(); H, W = idx.shape; gray = idx.astype(np.uint8)
        else:
            rgb = np.asarray(Image.open(r.path).convert("RGB")).copy(); H, W = rgb.shape[:2]; gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        pm = cv2.erode(product_mask(gray), np.ones((31, 31), np.uint8))
        gts = [(float(l.split()[1]) * W, float(l.split()[2]) * H) for l in (DATA / "labels" / "val" / f"{r.stem}.txt").read_text().splitlines() if l.strip()]
        ys, xs = np.where(pm > 0); cand = list(zip(xs.tolist(), ys.tolist())); rng.shuffle(cand)
        placed = []
        for x, y in cand:
            if all(((x - gx)**2 + (y - gy)**2) ** 0.5 > 25 for gx, gy in gts) and all(((x - px)**2 + (y - py)**2) ** 0.5 > 30 for px, py in placed):
                placed.append((x, y))
            if len(placed) == 6: break
        # 앞 3개: 가짜 색 네모를 그림 (색 네모와 같은 20px 빨간 테두리), 뒤 3개: 아무것도 안 그림 (대조군)
        for k, (x, y) in enumerate(placed):
            kind = "fake" if k < 3 else "control"
            if kind == "fake" and V2: cv2.rectangle(idx, (x - 10, y - 10), (x + 10, y + 10), 255, 2)
            elif kind == "fake": cv2.rectangle(rgb, (x - 10, y - 10), (x + 10, y + 10), (255, 0, 0), 2)
            rows.append([r.stem, x, y, kind])
        if V2: masked, _ = mask_palette(idx)
        else:
            tmp = out_dir / f"{r.stem}_tmp.png"; Image.fromarray(rgb).save(tmp); masked, _ = mask_image(tmp); tmp.unlink()
        cv2.imwrite(str(out_dir / f"{r.stem}.png"), masked)
    with open(DATA / "fake_positions.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["stem", "cx", "cy", "kind"]); w.writerows(rows)
    print("사진", len(man), "장, 가짜 자리", sum(1 for r in rows if r[3] == "fake"), "개, 대조 자리", sum(1 for r in rows if r[3] == "control"), "개 →", out_dir)

if __name__ == "__main__":
    main()
