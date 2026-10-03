#!/usr/bin/env python3
"""
정식 데이터셋 v2 준비. v1(scripts/xray_prepare.py → data/xray_v1)과 다른 점은 세 가지다.
- 검사기 표시 지우기: 팔레트 244~255번 픽셀만 고르고(넓히지 않음) 나비에-스토크스 인페인팅(반경 3)으로 메운다.
  v1은 같은 픽셀을 5×5로 넓혀 텔레아로 메웠다. 근거는 scripts/xray_mask_compare.py의 옮겨 놓기 시험.
- 점검 묶음: 라벨 여부와 관계없이 서로 다른 원본 사진 전체를 같은 호기·60초 규칙으로 묶는다(한 번의 점검 = 한 묶음).
- 분할: 묶음 단위로, 호기 × 시험편 종류(막대3/막대1)마다 사진 수 기준 60:20:20을 맞춘다.
  라벨 누락이 의심되는 사진이 든 묶음은 학습 세트로 고정한다(평가에 넣으면 제대로 찾아도 오경보로 채점되기 때문).
manifest 열: stem, path(프로젝트 루트 기준 상대경로), md5, machine, dt, hour, routine, n_boxes, card_type,
            practice_set, label_issue, width, height, burst_id, split, masked_px
출력: data/xray_v2/{images,labels}/{train,val,test}, manifest.csv, split.csv, data.yaml, summary.json
사용: .venv/bin/python scripts/xray_prepare_v2.py [--rebuild]
"""
import argparse, hashlib, json, random, re, shutil
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd, cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "제조AI데이터셋" / "4. X-ray 검사장비 AI 데이터셋" / "dataset"
RAW = DS / "test1" / "yolov3" / "X선이물검출기(06.23_09.22)"
SETS = DS / "라벨링 6종 세트"; LAB = SETS / "labels"
OUT = ROOT / "data" / "xray_v2"
PAT = re.compile(r"^(\d{3})_(\d{8})_(\d{6})\((\d+)\)$")
MARK_MIN = 244                      # 팔레트 244~255번 = 검사기 표시 색, 0~243번 = 같은 번호의 회색
ROUTINE_HOURS = {0, 4, 8, 12, 16, 20}
RATIO = {"train": 0.6, "val": 0.2, "test": 0.2}

def load_palette(path):
    """원본 BMP를 팔레트 번호 배열로 연다. 0~243번이 같은 번호의 회색인지 확인한다."""
    im = Image.open(path)
    if im.mode != "P": raise ValueError(f"팔레트 이미지가 아님: {path}")
    pal = np.array(im.getpalette()[:768]).reshape(-1, 3)
    if not all(pal[i][0] == pal[i][1] == pal[i][2] == i for i in range(MARK_MIN)): raise ValueError(f"팔레트 구성이 다름: {path}")
    return np.asarray(im)

def mask_palette(idx):
    """검사기 표시(팔레트 244~255번)만 지우고 나비에-스토크스로 메운 흑백 이미지와 지운 픽셀 수를 돌려준다."""
    m = (idx >= MARK_MIN).astype(np.uint8)
    gray = idx.astype(np.uint8)
    return (cv2.inpaint(gray, m, 3, cv2.INPAINT_NS) if m.any() else gray.copy()), int(m.sum())

def label_issue(idx, boxes, W, H):
    """라벨과 검사기 네모(표시 테두리가 둘러싼 안쪽)를 맞춰 본다. 검사기미표시: 네모 밖 라벨, 라벨누락의심: 라벨 없는 네모."""
    mk = (idx >= MARK_MIN).astype(np.uint8)
    cnts, hier = cv2.findContours(mk, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    holes = [c for c, h in zip(cnts, hier[0]) if h[3] != -1 and cv2.contourArea(c) >= 9] if hier is not None else []
    centers = [(b[1] * W, b[2] * H) for b in boxes]
    inside = lambda c, pt: cv2.pointPolygonTest(c, pt, False) >= 0
    issues = []
    if not all(any(inside(c, pt) for c in holes) for pt in centers): issues.append("검사기미표시")
    if any(not any(inside(c, pt) for pt in centers) for c in holes): issues.append("라벨누락의심")
    return "+".join(issues)

def to_py(o):
    if isinstance(o, dict): return {str(k): to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [to_py(v) for v in o]
    return o.item() if hasattr(o, "item") else o

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--rebuild", action="store_true"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if (OUT / "manifest.csv").exists() and not a.rebuild:
        print("[skip] 이미 있음:", OUT); return
    if OUT.exists(): shutil.rmtree(OUT)

    # 1) 서로 다른 원본 사진 = (이름, 내용) 쌍. 같은 파일 사본은 경로 정렬 순서로 첫 파일만 쓴다
    rows, seen = [], set()
    for p in sorted(RAW.rglob("*.bmp")):
        m = PAT.match(p.stem)
        if not m: continue
        md5 = hashlib.md5(p.read_bytes()).hexdigest()
        if (p.stem, md5) in seen: continue
        seen.add((p.stem, md5))
        rows.append(dict(stem=p.stem, path=p.relative_to(ROOT).as_posix(), md5=md5, machine=p.relative_to(RAW).parts[0][:3],
                         dt=datetime.strptime(m.group(2) + m.group(3), "%Y%m%d%H%M%S")))
    allimg = pd.DataFrame(rows).sort_values(["machine", "dt", "stem"]).reset_index(drop=True)
    gap = allimg.groupby("machine").dt.diff().dt.total_seconds()
    allimg["burst_id"] = (gap.isna() | (gap > 60)).cumsum()

    # 2) 라벨 사진 500장과 태그
    labels = {p.stem: p for p in LAB.glob("*.txt")}
    lab = allimg[allimg.stem.isin(labels)].copy()
    if not (lab.stem.is_unique and len(lab) == len(labels)): raise RuntimeError("라벨 사진과 원본 사진이 1:1로 맞지 않음")
    sets = {n: {q.stem for q in (SETS / f"images {n}").glob("*")} for n in [15, 50, 100, 200, 300, 400]}
    extra = []
    for r in lab.itertuples():
        idx = load_palette(ROOT / r.path); H, W = idx.shape
        boxes = [list(map(float, l.split())) for l in labels[r.stem].read_text().splitlines() if l.strip()]
        extra.append(dict(stem=r.stem, width=W, height=H, n_boxes=len(boxes), label_issue=label_issue(idx, boxes, W, H)))
    lab = lab.merge(pd.DataFrame(extra), on="stem")
    lab["hour"] = lab.dt.dt.hour
    lab["routine"] = lab.hour.isin(ROUTINE_HOURS) & lab.dt.dt.minute.between(15, 50)
    lab["card_type"] = np.where(lab.n_boxes >= 2, "막대3", "막대1")      # 박스 2개짜리 1장은 라벨이 하나 빠진 막대 3개짜리
    lab["practice_set"] = lab.stem.map(lambda s: next((f"images {n}" for n in sorted(sets) if s in sets[n]), "라벨만"))

    # 3) 분할: 묶음 단위, 호기 × 시험편 종류마다 사진 수 기준 60:20:20 (모자란 쪽부터 채움)
    rng = random.Random(a.seed)
    bursts = lab.groupby("burst_id").agg(machine=("machine", "first"), card=("card_type", lambda s: s.mode()[0]), n=("stem", "size"),
                                         mixed=("card_type", "nunique"), forced=("label_issue", lambda s: s.str.contains("라벨누락").any()))
    split = {}
    for _, g in bursts.groupby(["machine", "card"]):
        total, cnt = g.n.sum(), {k: 0 for k in RATIO}
        ids = sorted(g.index); rng.shuffle(ids)
        for b in sorted(ids, key=lambda b: not g.loc[b, "forced"]):     # 고정할 묶음을 먼저 배정 (나머지 순서는 섞은 그대로)
            s = "train" if g.loc[b, "forced"] else max(RATIO, key=lambda k: RATIO[k] * total - cnt[k])
            split[b] = s; cnt[s] += g.loc[b, "n"]
    lab["split"] = lab.burst_id.map(split)

    # 4) 표시 지우기와 저장
    for s in RATIO:
        (OUT / "images" / s).mkdir(parents=True, exist_ok=True); (OUT / "labels" / s).mkdir(parents=True, exist_ok=True)
    masked = {}
    for r in lab.itertuples():
        idx = load_palette(ROOT / r.path); img, n = mask_palette(idx); masked[r.stem] = n
        if not np.array_equal(img[idx < MARK_MIN], idx[idx < MARK_MIN]): raise RuntimeError(f"표시 밖 픽셀이 바뀜: {r.stem}")
        cv2.imwrite(str(OUT / "images" / r.split / f"{r.stem}.png"), img)
        shutil.copy(labels[r.stem], OUT / "labels" / r.split / f"{r.stem}.txt")
    lab["masked_px"] = lab.stem.map(masked)
    cols = ["stem", "path", "md5", "machine", "dt", "hour", "routine", "n_boxes", "card_type", "practice_set", "label_issue",
            "width", "height", "burst_id", "split", "masked_px"]
    lab = lab.sort_values(["split", "machine", "dt"])[cols]
    lab.to_csv(OUT / "manifest.csv", index=False, encoding="utf-8-sig")
    lab[["stem", "burst_id", "split"]].sort_values("stem").to_csv(OUT / "split.csv", index=False, encoding="utf-8")
    (OUT / "data.yaml").write_text(f"path: {OUT}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: defect\n")

    # 5) 요약
    order = list(RATIO)
    share = lambda col: {k: round(float(v), 3) for k, v in col.reindex(order).items()}
    summ = dict(
        images=len(lab), boxes=int(lab.n_boxes.sum()), bursts=int(lab.burst_id.nunique()),
        mixed_card_bursts=int((bursts.mixed > 1).sum()),
        split_images=lab.split.value_counts().reindex(order).to_dict(),
        split_share=share(lab.split.value_counts() / len(lab)),
        split_bursts=lab.groupby("split").burst_id.nunique().reindex(order).to_dict(),
        split_boxes=lab.groupby("split").n_boxes.sum().reindex(order).to_dict(),
        single_card_share=share(lab.assign(v=lab.card_type == "막대1").groupby("split").v.mean()),
        stratum_split={f"{m} {c}": g.split.value_counts().reindex(order, fill_value=0).to_dict() for (m, c), g in lab.groupby(["machine", "card_type"])},
        machine_split={m: g.split.value_counts().reindex(order, fill_value=0).to_dict() for m, g in lab.groupby("machine")},
        label_only_split=lab[lab.practice_set == "라벨만"].split.value_counts().reindex(order, fill_value=0).to_dict(),
        label_issues=lab[lab.label_issue != ""][["stem", "label_issue", "split"]].to_dict("records"),
        masked_px_total=int(lab.masked_px.sum()), inpaint="팔레트 244~255번, 팽창 없음, cv2.INPAINT_NS 반경 3", seed=a.seed,
        split_csv_md5=hashlib.md5((OUT / "split.csv").read_bytes()).hexdigest())
    summ = to_py(summ)
    (OUT / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=1)); print("saved", OUT)

if __name__ == "__main__":
    main()
