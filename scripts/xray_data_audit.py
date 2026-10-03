#!/usr/bin/env python3
"""
원본 X-ray 데이터셋 점검. 원본 폴더는 읽기만 한다.
- 원본 폴더의 모든 파일을 역할별로 목록화한다 (확장자와 실제 형식이 다른 파일도 표시)
- 원본 사진(BMP)마다 호기, 촬영 시각, 크기, 중복, 라벨 여부, 색 네모(팔레트 244~255번 색)를 기록한다
- 공식 라벨 박스마다 크기, 장비 네모 안에 있는지, 대비, 제품 가장자리까지 거리, 마스킹이 박스를 건드리는지를 기록한다
- 촬영 시각으로 연속 촬영 묶음(60초 이내)과 하루 중 촬영 시각 분포를 본다
출력: reports/data_audit/{inventory.csv, raw_images.csv, label_boxes.csv, summary.json}
사용: .venv/bin/python scripts/xray_data_audit.py
"""
import re, json, hashlib, collections
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd, cv2
from PIL import Image
from xray_eval import product_mask

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "제조AI데이터셋" / "4. X-ray 검사장비 AI 데이터셋" / "dataset"
RAW = DS / "test1" / "yolov3" / "X선이물검출기(06.23_09.22)"
LABSET = DS / "라벨링 6종 세트"
OUT = ROOT / "reports" / "data_audit"
PAT = re.compile(r"^(\d{3})_(\d{8})_(\d{6})\((\d+)\)$")
MARK_MIN = 244            # 팔레트 244~255번은 색 네모 색, 0~243번은 X-ray 회색
PRACTICE = [15, 50, 100, 200, 300, 400]

GROUPS = [  # (상대경로 앞부분, 역할)
    ("test1/yolov3/X선이물검출기(06.23_09.22)/", "원본 사진: 검사기가 저장한 BMP"),
    ("라벨링 6종 세트/labels/", "공식 정답 라벨 TXT"),
    ("라벨링 6종 세트/images", "실습용 사진 사본 (15~400장 묶음)"),
    ("실습별 가중치파일/", "실습 묶음별로 학습한 YOLOv3 가중치"),
    ("test1/yolov3/weights/", "가이드북 YOLOv3 가중치"),
    ("test1/yolov3/images/", "가이드북 예제 사진 15장"),
    ("test1/yolov3/labels/", "가이드북 예제 라벨 15개"),
    ("test1/yolov3/result/", "가이드북 예제 탐지 결과 그림"),
    ("test1/yolov3/utils/", "가이드북 YOLOv3 코드"),
    ("test1/yolov3/", "가이드북 YOLOv3 코드·설정·학습 기록"),
    ("OpenLabeling-master/main/input/", "라벨링 도구 예제 입력 사진"),
    ("OpenLabeling-master/main/output/YOLO_darknet/", "라벨링 도구 출력 (YOLO 형식)"),
    ("OpenLabeling-master/main/output/PASCAL_VOC/", "라벨링 도구 출력 (VOC XML)"),
    ("OpenLabeling-master/", "라벨링 도구 OpenLabeling 코드"),
    ("yolov3_20201200.ipynb", "가이드북 실습 노트북"),
]

def real_type(p):
    with open(p, "rb") as f: head = f.read(8)
    if head[:2] == b"BM": return "bmp"
    if head[:3] == b"\xff\xd8\xff": return "jpeg"
    if head[:8] == b"\x89PNG\r\n\x1a\n": return "png"
    if head[:2] == b"PK": return "zip(pt)"
    if head[:1] in (b"{", b"["): return "json"
    try:
        p.read_bytes()[:4096].decode("utf-8"); return "text"
    except UnicodeDecodeError:
        return "binary"

def inventory():
    rows = []
    for p in sorted(DS.rglob("*")):
        if not p.is_file() or p.name == ".DS_Store" or "__pycache__" in p.parts: continue
        rel = p.relative_to(DS).as_posix()
        role = next((r for pre, r in GROUPS if rel.startswith(pre)), "기타")
        ext = p.suffix.lower().lstrip(".") or p.name.lstrip(".")
        rt = real_type(p)
        expect = {"bmp": "bmp", "jpg": "jpeg", "png": "png", "pt": "zip(pt)", "ipynb": "json"}.get(ext)
        rows.append(dict(rel=rel, role=role, ext=ext, real_type=rt, size=p.stat().st_size,
                         ext_mismatch=bool(expect and expect != rt)))
    return pd.DataFrame(rows)

def clean(o):
    """numpy 숫자를 json에 쓸 수 있는 파이썬 값으로 바꾼다."""
    if isinstance(o, dict): return {str(k.item() if hasattr(k, "item") else k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    return o.item() if hasattr(o, "item") else o

def mark_regions(idx):
    """색 네모 픽셀과, 네모 테두리가 둘러싼 안쪽 영역(구멍) 목록을 돌려준다."""
    mk = (idx >= MARK_MIN).astype(np.uint8)
    holes = []
    if mk.any():
        cnts, hier = cv2.findContours(mk, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        for c, h in zip(cnts, hier[0]):
            if h[3] != -1 and cv2.contourArea(c) >= 9: holes.append(c)
    n, _, st, _ = cv2.connectedComponentsWithStats(mk, 8)
    lines = sum(1 for i in range(1, n) if (st[i, 2] <= 3 and st[i, 3] >= 30) or (st[i, 3] <= 3 and st[i, 2] >= 30))
    return mk, holes, lines

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inv = inventory(); inv.to_csv(OUT / "inventory.csv", index=False, encoding="utf-8-sig")

    labels = {p.stem: p for p in (LABSET / "labels").glob("*.txt")}
    sets = {n: {p.stem for p in (LABSET / f"images {n}").glob("*")} for n in PRACTICE}
    demo = {p.stem: p for p in (DS / "test1/yolov3/labels").glob("*.txt")}

    # 1) 원본 사진 파일 하나하나
    rows = []
    for p in sorted(RAW.rglob("*.bmp")):
        m = PAT.match(p.stem); rel = p.relative_to(RAW)
        im = Image.open(p); idx = np.asarray(im)
        pal = np.array(im.getpalette()[:768]).reshape(-1, 3) if im.mode == "P" else None
        pal_ok = pal is not None and all(pal[i][0] == pal[i][1] == pal[i][2] == i for i in range(MARK_MIN)) and \
            all(not (pal[i][0] == pal[i][1] == pal[i][2]) for i in range(MARK_MIN, 256))
        mk, holes, lines = mark_regions(idx) if im.mode == "P" else (None, [], 0)
        colors = sorted(int(v) for v in np.unique(idx[idx >= MARK_MIN])) if im.mode == "P" else []
        dt = datetime.strptime(m.group(2) + m.group(3), "%Y%m%d%H%M%S")
        rows.append(dict(file=rel.as_posix(), stem=p.stem, machine=rel.parts[0][:3], serial=rel.parts[1].split("_")[0],
                         folder_date=rel.parts[1].split("_")[1], name_prefix=m.group(1), dt=dt, name_seq=int(m.group(4)),
                         width=im.size[0], height=im.size[1], mode=im.mode, palette_ok=pal_ok,
                         md5=hashlib.md5(p.read_bytes()).hexdigest(), mark_px=int(mk.sum()) if mk is not None else 0,
                         mark_boxes=len(holes), mark_lines=lines, mark_colors=" ".join(map(str, colors)),
                         labeled=p.stem in labels))
    raw = pd.DataFrame(rows)
    raw["copies"] = raw.groupby("stem").stem.transform("size")
    raw["copies_identical"] = raw.groupby("stem").md5.transform("nunique") == 1
    raw["first_copy"] = ~raw.duplicated(["stem", "md5"])      # 서로 다른 사진 = (이름, 내용) 쌍
    raw["folder_date_differs"] = raw.folder_date != raw.dt.dt.strftime("%Y%m%d")
    raw["mark_type"] = np.where(raw.mark_boxes > 0, "네모", np.where(raw.mark_lines > 0, "선", np.where(raw.mark_px > 0, "기타", "없음")))
    raw["practice_set"] = raw.stem.map(lambda s: next((f"images {n}" for n in PRACTICE if s in sets[n]), "라벨만" if s in labels else ""))
    # 연속 촬영 묶음: 같은 호기, 이전 사진과 60초 이내 (고유 사진 기준)
    u = raw[raw.first_copy].sort_values(["machine", "dt"]).copy()
    gap = u.groupby("machine").dt.diff().dt.total_seconds()
    u["burst_id"] = (gap.isna() | (gap > 60)).cumsum()
    raw = raw.merge(u[["stem", "md5", "burst_id"]], on=["stem", "md5"], how="left")
    raw.to_csv(OUT / "raw_images.csv", index=False, encoding="utf-8-sig")

    # 2) 공식 라벨 박스 하나하나
    first = raw[raw.first_copy & raw.labeled].set_index("stem")   # 라벨 사진은 이름이 겹치지 않는다
    brows = []
    for s, lp in sorted(labels.items()):
        r = first.loc[s]; p = RAW / r.file
        idx = np.asarray(Image.open(p)); H, W = idx.shape
        mk, holes, _ = mark_regions(idx)
        hole_map = np.zeros((H, W), np.int32)
        for k, c in enumerate(holes, 1): cv2.drawContours(hole_map, [c], -1, k, -1)
        gray = idx.astype(np.uint8).copy(); gblur = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 0.8)
        filled = cv2.inpaint(gray, mk, 3, cv2.INPAINT_TELEA) if mk.any() else gray
        dist = cv2.distanceTransform(product_mask(filled), cv2.DIST_L2, 3)
        dil = cv2.dilate(mk, np.ones((5, 5), np.uint8))          # xray_prepare.mask_image 과 같은 팽창
        mdist = cv2.distanceTransform((1 - mk).astype(np.uint8), cv2.DIST_L2, 3) if mk.any() else np.full((H, W), 1e3, np.float32)
        yy, xx = np.mgrid[0:H, 0:W]
        boxes = [list(map(float, l.split())) for l in lp.read_text().splitlines() if l.strip()]
        order = np.argsort([b[2] for b in boxes])                # 위에서부터 순서 (막대 위치)
        for rank, j in enumerate(order):
            c, cx, cy, w, h = boxes[j]; cx, cy, w, h = cx * W, cy * H, w * W, h * H
            xi, yi = int(min(max(cx, 0), W - 1)), int(min(max(cy, 0), H - 1))
            x0, x1, y0, y1 = max(0, int(cx - w / 2)), min(W, int(np.ceil(cx + w / 2))), max(0, int(cy - h / 2)), min(H, int(np.ceil(cy + h / 2)))
            hole = int(hole_map[yi, xi])
            color = -1
            if hole:
                ring = cv2.dilate((hole_map == hole).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & (mk > 0)
                if ring.any(): color = int(np.bincount(idx[ring]).argmax())
            # 금속구 위치: 라벨 중심 반경 4픽셀 안(표시 픽셀 제외)에서 살짝 흐린 영상의 가장 어두운 점 (라벨 중심이 조금 비껴 있을 수 있다)
            sub = np.where((np.hypot(xx - cx, yy - cy) <= 4) & (mk == 0), gblur, np.inf)
            by, bx = np.unravel_index(np.argmin(sub), sub.shape)
            rr = np.hypot(xx - bx, yy - by); ann = (rr >= 5) & (rr <= 9) & (mk == 0)
            center = float(gray[max(0, by - 1):by + 2, max(0, bx - 1):bx + 2].min())   # 깊이는 원래 픽셀 값으로
            bg = float(np.median(gray[ann])) if ann.any() else np.nan
            brows.append(dict(stem=s, machine=r.machine, n_boxes=len(boxes), pos_rank=rank, cx=cx, cy=cy, w=w, h=h, size=max(w, h),
                              inside_mark=bool(hole), mark_color=color, ball_x=int(bx), ball_y=int(by), ball_offset=float(np.hypot(bx - cx, by - cy)),
                              center_gray=center, bg_gray=bg, contrast=bg - center,
                              edge_dist=float(dist[yi, xi]), box_mark_px=int(mk[y0:y1, x0:x1].sum()),
                              box_masked_frac=float(dil[y0:y1, x0:x1].mean()) if y1 > y0 and x1 > x0 else np.nan,
                              mark_dist=float(mdist[yi, xi]), core_masked=bool((dil[(rr <= 3)] > 0).any())))
    lb = pd.DataFrame(brows); lb.to_csv(OUT / "label_boxes.csv", index=False, encoding="utf-8-sig")

    # 3) 다른 곳에 있는 라벨(가이드북 예제 15개)과 공식 라벨 비교
    def rd(p): return [tuple(round(float(v), 6) for v in l.split()) for l in p.read_text().splitlines() if l.strip()]
    demo_overlap = sorted(set(demo) & set(labels))
    demo_conflict = [s for s in demo_overlap if rd(demo[s]) != rd(labels[s])]

    # 4) 요약
    uq = raw[raw.first_copy]
    lab_u = uq[uq.labeled]
    labeled_marks = lab_u.set_index("stem")
    per_img = lb.groupby("stem").agg(n=("inside_mark", "size"), inside=("inside_mark", "sum"))
    extra_marks = int(sum(max(0, labeled_marks.loc[s, "mark_boxes"] - per_img.loc[s, "inside"]) for s in per_img.index))
    q = lambda s: {k: round(float(v), 2) for k, v in s.quantile([0.1, 0.5, 0.9]).items()}
    # 4시간 정기 점검: 0·4·8·12·16·20시의 15~50분에 찍힌 사진
    routine = uq.dt.dt.hour.isin([0, 4, 8, 12, 16, 20]) & uq.dt.dt.minute.between(15, 50)
    bu = uq.sort_values("dt").groupby("burst_id").agg(machine=("machine", "first"), start=("dt", "min"),
                                                     pattern=("mark_boxes", lambda v: "".join(map(str, v))))
    bu["routine"] = bu.start.dt.hour.isin([0, 4, 8, 12, 16, 20]) & bu.start.dt.minute.between(15, 50)
    kind = uq.mark_type.where(uq.mark_type != "네모", uq.mark_boxes.map({1: "막대1", 2: "표시2개", 3: "막대3"}).fillna("기타"))
    card = pd.crosstab([uq.machine, uq.dt.dt.strftime("%m")], kind).to_dict("index")
    bstat = uq.groupby("burst_id").agg(machine=("machine", "first"), start=("dt", "min"), end=("dt", "max"), n=("stem", "size"))
    bstat["dur"] = (bstat.end - bstat.start).dt.total_seconds()
    slot = bstat[bu.routine].assign(slot=lambda d: d.start.dt.strftime("%Y-%m-%d %H"))
    order = slot.sort_values("start").groupby("slot").machine.apply(lambda v: " > ".join(dict.fromkeys(v)))
    days = uq.groupby("machine").dt.apply(lambda v: frozenset(v.dt.date))
    import xml.etree.ElementTree as ET
    voc = sorted((DS / "OpenLabeling-master/main/output/PASCAL_VOC").glob("*.xml"))
    voc_objs = [len(ET.parse(v).getroot().findall("object")) for v in voc]
    # 참고: 우리 v1 분할과 실습 묶음의 관계 (가이드북 가중치가 우리 평가 사진을 이미 봤는지)
    v1 = ROOT / "data" / "xray_v1" / "manifest.csv"
    v1_split = {}
    if v1.exists():
        mv = pd.read_csv(v1, encoding="utf-8-sig")[["stem", "split"]].merge(lab_u[["stem", "practice_set"]], on="stem")
        mv["seen_by_practice"] = mv.practice_set != "라벨만"
        v1_split = {f"{k[0]} {'실습묶음' if k[1] else '라벨만'}": int(v) for k, v in mv.groupby(["split", "seen_by_practice"]).size().items()}
    card_l = pd.crosstab([uq[uq.labeled].machine, uq[uq.labeled].dt.dt.strftime("%m")], kind[uq.labeled]).to_dict("index")
    summ = dict(
        files=dict(total=int(len(inv)), by_role={k: dict(n=int(v.size), mb=round(v.sum() / 1e6, 1)) for k, v in inv.groupby("role")["size"]},
                   ext_mismatch=int(inv.ext_mismatch.sum()),
                   ext_mismatch_by_role={k: int(v) for k, v in inv[inv.ext_mismatch].role.value_counts().items()}),
        raw=dict(files=int(len(raw)), names=int(raw.stem.nunique()), distinct_images=int(raw.first_copy.sum()),
                 identical_copies=int(len(raw) - raw.first_copy.sum()),
                 identical_copy_same_machine=int(raw[~raw.first_copy].merge(raw[raw.first_copy][["stem", "md5", "machine"]], on=["stem", "md5"], suffixes=("", "_1")).eval("machine == machine_1").sum()),
                 name_collisions=sorted(raw[~raw.copies_identical].stem.unique().tolist()),
                 name_collisions_labeled=int(raw[~raw.copies_identical].labeled.sum()),
                 per_machine=uq.machine.value_counts().sort_index().to_dict(),
                 serial=uq.groupby("machine").serial.first().to_dict(),
                 name_prefix=uq.groupby("machine").name_prefix.agg(lambda s: sorted(s.unique())).to_dict(),
                 sizes={f"{a} {b}x{c}": int(v) for (a, b, c), v in uq.groupby(["machine", "width", "height"]).size().items()},
                 date_range=[str(uq.dt.min()), str(uq.dt.max())], days_with_images=int(uq.dt.dt.date.nunique()),
                 palette_mode_P=int((raw["mode"] == "P").sum()), palette_ok=int(raw.palette_ok.sum()),
                 folder_date_differs=int(raw.folder_date_differs.sum()),
                 folder_date_differs_hours=raw[raw.folder_date_differs].dt.dt.hour.value_counts().sort_index().to_dict()),
        marks=dict(mark_type=uq.mark_type.value_counts().to_dict(),
                   boxes_per_image=uq.mark_boxes.value_counts().sort_index().to_dict(),
                   line_images=dict(n=int((uq.mark_type == "선").sum()),
                                    machine=uq[uq.mark_type == "선"].machine.value_counts().to_dict(),
                                    sizes={f"{a}x{b}": int(v) for (a, b), v in uq[uq.mark_type == "선"].groupby(["width", "height"]).size().items()},
                                    dates=sorted(uq[uq.mark_type == "선"].dt.dt.strftime("%Y-%m-%d").unique().tolist()),
                                    time_range=[str(uq[uq.mark_type == "선"].dt.min()), str(uq[uq.mark_type == "선"].dt.max())],
                                    labeled=int(uq[uq.mark_type == "선"].labeled.sum())),
                   label_boxes_inside_mark=int(lb.inside_mark.sum()), label_boxes=int(len(lb)),
                   label_boxes_outside_mark=lb[~lb.inside_mark][["stem", "machine", "n_boxes", "pos_rank"]].to_dict("records"),
                   marks_without_label_in_labeled_images=extra_marks,
                   box_color_index=lb[lb.inside_mark].mark_color.value_counts().sort_index().to_dict()),
        labels=dict(images=int(len(labels)), boxes=int(len(lb)), n_boxes_per_image=lab_u.stem.map(per_img.n).value_counts().sort_index().to_dict(),
                    per_machine_images=lab_u.machine.value_counts().sort_index().to_dict(),
                    practice_membership=lab_u.practice_set.value_counts().to_dict(),
                    label_only_100=dict(machine=lab_u[lab_u.practice_set == "라벨만"].machine.value_counts().to_dict(),
                                        n_boxes=lab_u[lab_u.practice_set == "라벨만"].stem.map(per_img.n).value_counts().to_dict(),
                                        months=lab_u[lab_u.practice_set == "라벨만"].dt.dt.strftime("%Y-%m").value_counts().sort_index().to_dict()),
                    practice_nested=all(sets[a] <= sets[b] for a, b in zip(PRACTICE, PRACTICE[1:])),
                    demo_labels=len(demo), demo_in_official=len(demo_overlap), demo_conflict=demo_conflict,
                    labeled_share_by_machine=(lab_u.machine.value_counts() / uq.machine.value_counts()).round(3).sort_index().to_dict()),
        objects=dict(size_px=q(lb["size"]), size_px_by_machine={k: q(v) for k, v in lb.groupby("machine")["size"]},
                     ball_offset_from_label_center=q(lb.ball_offset), contrast=q(lb.contrast), contrast_by_machine={k: q(v) for k, v in lb.groupby("machine").contrast},
                     contrast_by_pos_3box={str(k): q(v) for k, v in lb[lb.n_boxes == 3].groupby("pos_rank").contrast},
                     edge_dist=q(lb.edge_dist), image_area_share_pct=round(float((lb.w * lb.h).sum() / sum(first.loc[s, "width"] * first.loc[s, "height"] for s in labels) * 100), 3),
                     boxes_touched_by_raw_mark=int((lb.box_mark_px > 0).sum()), box_masked_frac=q(lb.box_masked_frac),
                     boxes_masked_over_10pct=int((lb.box_masked_frac > 0.1).sum()),
                     mark_dist_from_center=q(lb.mark_dist), core_masked=int(lb.core_masked.sum())),
        time=dict(hour_all=uq.dt.dt.hour.value_counts().sort_index().to_dict(), hour_labeled=lab_u.dt.dt.hour.value_counts().sort_index().to_dict(),
                  minute_all_q=q(uq.dt.dt.minute.astype(float)),
                  bursts=int(uq.burst_id.nunique()), burst_size=q(uq.groupby("burst_id").size().astype(float)),
                  images_in_multi_bursts=round(float((uq.groupby("burst_id").stem.transform("size") > 1).mean()), 3),
                  labeled_bursts=int(lab_u.burst_id.nunique()),
                  routine_share=round(float(routine.mean()), 3), routine_share_labeled=round(float(routine[uq.labeled].mean()), 3),
                  burst_mark_pattern=bu.pattern.value_counts().head(6).to_dict(),
                  routine_bursts_per_machine_day=bu[bu.routine].groupby(["machine", bu[bu.routine].start.dt.date]).size().value_counts().sort_index().to_dict(),
                  card_by_machine_month={f"{k[0]} {k[1]}": v for k, v in card.items()},
                  card_by_machine_month_labeled={f"{k[0]} {k[1]}": v for k, v in card_l.items()},
                  burst_duration_sec=q(bstat.dur), burst_size_by_month={f"{k[0]} {k[1]}장": int(v) for k, v in bstat.groupby([bstat.start.dt.strftime("%m"), bstat.n]).size().items()},
                  routine_slots=int(len(order)), routine_slot_machine_order=order.value_counts().head(5).to_dict(),
                  days_per_machine={m: len(d) for m, d in days.items()}, same_days_all_machines=bool(len(set(days.values)) == 1)),
        extra=dict(two_box_images=int((uq.mark_boxes == 2).sum()), two_box_images_labeled=int(((uq.mark_boxes == 2) & uq.labeled).sum()),
                   edge_dist_min=round(float(lb.edge_dist.min()), 1), voc_xml=len(voc), voc_empty=int(sum(o == 0 for o in voc_objs)),
                   voc_tool_samples=int(sum(not PAT.match(v.stem) for v in voc)),
                   size_412_line=int(((uq.width == 412) & (uq.mark_type == "선")).sum()), size_412_box=int(((uq.width == 412) & (uq.mark_type == "네모")).sum()),
                   practice_set_by_v1_split=v1_split),
    )
    summ = clean(summ)
    (OUT / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=1, default=str))
    print("saved", OUT)

if __name__ == "__main__":
    main()
