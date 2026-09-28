#!/usr/bin/env python3
"""
가이드북(대회가 함께 준 YOLOv3 실습 자료) 점검. 원본 폴더는 읽기만 한다.
- 실습별 가중치파일(last15~last400.pt)과 test1/yolov3/weights/last.pt 안에 저장된 학습 기록(results.txt 형식)을 읽어
  학습 횟수, 학습 입력 크기 범위, 마지막 줄의 검증 성능(가이드북 자체 무작위 분할 기준)을 표로 만든다
- 확장자가 .jpg 등으로 바뀐 코드 사본이 원래 파일과 같은지 비교한다
가중치 파일은 피클이라, numpy 배열 복원 함수만 허용하는 안전한 방식으로 연다 (그 밖의 객체가 나오면 멈춘다).
출력: reports/guidebook_audit/{checkpoints.csv, renamed_copies.csv}
사용: .venv/bin/python scripts/xray_guidebook_audit.py
"""
import re, importlib, filecmp
from pathlib import Path
import pandas as pd, torch
ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "제조AI데이터셋" / "4. X-ray 검사장비 AI 데이터셋" / "dataset"
GUIDE = DS / "test1" / "yolov3"; OUT = ROOT / "reports" / "guidebook_audit"
COLS = ["epoch", "gpu_mem", "giou", "obj", "cls", "total", "targets", "img_size", "P", "R", "mAP50", "F1", "val_giou", "val_obj", "val_cls"]

def safe_load(p):
    for _ in range(20):
        try:
            return torch.load(p, map_location="cpu", weights_only=True)
        except Exception as e:
            s = str(e)
            m = re.search(r"Unsupported global: GLOBAL ([\w\.]+)", s) or re.search(r"but got <class '([\w\.]+)'>", s)
            if not m or not m.group(1).startswith("numpy."): raise
            mod, name = m.group(1).rsplit(".", 1)
            torch.serialization.add_safe_globals([getattr(importlib.import_module(mod), name)])
    raise RuntimeError("허용해야 할 객체가 너무 많다")

def main():
    OUT.mkdir(parents=True, exist_ok=True); rows = []
    files = [(f"실습 {n}장", DS / "실습별 가중치파일" / f"last{n}.pt") for n in [15, 50, 100, 200, 300, 400]] + [("예제 weights/last.pt", GUIDE / "weights" / "last.pt")]
    for name, p in files:
        ck = safe_load(p)
        lines = [l.split() for l in (ck.get("training_results") or "").splitlines() if l.strip()]
        rec = pd.DataFrame([l[:len(COLS)] for l in lines if len(l) >= len(COLS)], columns=COLS)
        for c in COLS[2:]: rec[c] = pd.to_numeric(rec[c], errors="coerce")
        last = rec.iloc[-1]
        rows.append(dict(weights=name, file=p.relative_to(DS).as_posix(), mb=round(p.stat().st_size / 1e6, 1),
                         params_M=round(sum(v.numel() for v in ck["model"].values() if hasattr(v, "numel")) / 1e6, 2),
                         saved_epoch=int(ck["epoch"]), record_lines=len(rec), img_size_min=int(rec.img_size.min()), img_size_max=int(rec.img_size.max()),
                         last_P=float(last.P), last_R=float(last.R), last_mAP50=float(last.mAP50), last_F1=float(last.F1),
                         best_F1=float(rec.F1.max()), optimizer_saved=ck.get("optimizer") is not None))
        print(rows[-1])
    pd.DataFrame(rows).to_csv(OUT / "checkpoints.csv", index=False, encoding="utf-8-sig")
    # 확장자가 바뀐 사본: 이름 앞부분이 같은 원래 파일과 비교
    cmp = []
    for p in sorted(GUIDE.iterdir()):
        if not p.is_file(): continue
        ext = p.name.rsplit(".", 1)[-1] if "." in p.name else ""        # 숨김 파일(.jpgignore 등)도 포함
        if not ext.startswith("jpg") or p.name.startswith("test_batch"): continue
        stem = p.name[: p.name.rfind(".")] if not p.name.startswith(".") else ""
        orig = [q for q in GUIDE.iterdir() if q.is_file() and q != p and (q.name[: q.name.rfind(".")] == stem if stem else False)]
        cmp.append(dict(copy=p.name, original=" ".join(q.name for q in orig) or "-", identical=any(filecmp.cmp(p, q, shallow=False) for q in orig)))
    pd.DataFrame(cmp).to_csv(OUT / "renamed_copies.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(cmp).to_string(index=False)); print("saved", OUT)

if __name__ == "__main__":
    main()
