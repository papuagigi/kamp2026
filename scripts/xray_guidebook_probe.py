#!/usr/bin/env python3
"""
가이드북(주최 측 실습 코드) YOLOv3-SPP 실습 가중치를, 어떤 실습 가중치도 학습에 쓰지 않은 사진에 돌려 예측 CSV로 저장한다.
- 대상: 정답 TXT(라벨링 6종 세트/labels)는 있지만 실습 폴더(images 15~400)에는 없는 사진 = "라벨 전용 100장"
  (목록은 reports/guidebook_probe/label_only_100.txt 로 저장)
- 입력 조건 raw: 장비가 그린 색 네모가 남은 원본(cv2.imread). 가이드북이 학습한 입력과 같다.
  입력 조건 masked: scripts/xray_prepare.py 의 mask_image 로 색 네모를 지운 흑백을 3채널로 바꾼 것
- 전처리·후처리는 가이드북 detect.py 와 같다: letterbox → BGR→RGB, CHW, /255 → model(img)[0]
  → non_max_suppression → scale_coords(원본 픽셀) → 반올림. 단 채점기가 문턱값을 훑도록 conf 0.001 로 뽑는다(detect.py 기본 0.3).
- 원본 폴더(제조AI데이터셋/)는 읽기만 한다. 가이드북 코드를 import 해도 .pyc 가 생기지 않게 막는다.
사용: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/xray_guidebook_probe.py --weights <실습별 가중치파일/last400.pt> --cond raw
      채점: .venv/bin/python scripts/xray_eval.py reports/preds_guidebook_last400_raw_labelonly100.csv --split train,val,test --stems reports/guidebook_probe/label_only_100.txt
      요약 표: .venv/bin/python scripts/xray_guidebook_probe.py --summary  (채점 JSON을 모아 reports/guidebook_probe/summary.md)
예측 CSV: reports/preds_guidebook_<가중치>_<조건>_labelonly100.csv (stem, cx, cy, w, h, score; 원본 픽셀), 실행 기록은 같은 이름 _run.json
"""
import sys
sys.dont_write_bytecode = True  # 가이드북 폴더에 __pycache__ 가 생기지 않게 (가이드북 import 전에 둔다)
import argparse, csv, importlib, importlib.machinery, json, os, random, re, time, types, warnings
from pathlib import Path
import numpy as np, cv2, torch

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "제조AI데이터셋" / "4. X-ray 검사장비 AI 데이터셋" / "dataset"
GB = DS / "test1" / "yolov3"; RAW = GB / "X선이물검출기(06.23_09.22)"   # 가이드북 코드와 원본 사진 (읽기만)
SET = DS / "라벨링 6종 세트"; WTS = DS / "실습별 가중치파일"
REPORTS = ROOT / "reports"; OUT = REPORTS / "guidebook_probe"
NS = [15, 50, 100, 200, 300, 400]
sys.path.insert(0, str(ROOT / "scripts"))
from xray_prepare import mask_image

def safe_load(p):
    """체크포인트(pickle)를 weights_only 로 연다. numpy 복원용 전역만 허용하고, 그 밖의 것이 나오면 멈춘다."""
    for _ in range(20):
        try:
            return torch.load(p, map_location="cpu", weights_only=True)
        except Exception as e:
            s = str(e)
            m = re.search(r"Unsupported global: GLOBAL ([\w\.]+)", s) or re.search(r"but got <class '([\w\.]+)'>", s)
            if not m or not m.group(1).startswith("numpy."): raise
            mod, name = m.group(1).rsplit(".", 1)
            torch.serialization.add_safe_globals([getattr(importlib.import_module(mod), name)])
    raise RuntimeError("too many globals")

def import_guidebook():
    """가이드북 코드(2020년 ultralytics/yolov3)를 읽기 전용으로 불러온다. 지금 환경에 맞춘 호환 처리 포함."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        for n, t in [("int", int), ("float", float), ("bool", bool)]:   # numpy 1.24+ 에서 사라진 별칭
            if not hasattr(np, n): setattr(np, n, t)
    try: import tqdm  # noqa: F401
    except ImportError:  # tqdm 미설치: 가이드북은 진행 막대로만 쓰므로 받은 것을 그대로 돌려주는 대체물 (spec은 torch._dynamo 가 찾음)
        m = types.ModuleType("tqdm"); m.tqdm = lambda it=None, *a, **k: it
        m.__spec__ = importlib.machinery.ModuleSpec("tqdm", None); sys.modules["tqdm"] = m
    sys.modules["thop"] = None  # 모델 요약용 FLOPs 계산(무작위 가중치로 추가 forward 1회)을 건너뜀
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(GB))
    from models import Darknet
    from utils.datasets import letterbox
    from utils.utils import non_max_suppression, scale_coords
    return Darknet, letterbox, non_max_suppression, scale_coords

def label_only_stems():
    """정답 TXT가 있는 500장 중 어느 실습 폴더(images 15~400)에도 없는 사진 = 어떤 실습 가중치도 못 본 사진"""
    seen = {p.stem for d in SET.glob("images *") for p in d.iterdir() if not p.name.startswith(".")}
    return sorted({p.stem for p in (SET / "labels").glob("*.txt")} - seen)

def raw_paths():
    """사진 이름 → 원본 BMP. 폴더 간 중복 사본(내용 같음)은 경로 정렬 순서의 첫 것"""
    m = {}
    for p in sorted(RAW.rglob("*.bmp")): m.setdefault(p.stem, p)
    return m

def load_image(path, cond):
    if cond == "raw": return cv2.imread(str(path))           # BGR 3채널, 색 네모 포함
    gray, _ = mask_image(path); return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

@torch.no_grad()
def detect(model, img0, gb, img_size, device, conf, iou):
    """detect.py 와 같은 순서. 반환: (x0, y0, x1, y1, score, cls) 원본 픽셀"""
    _, letterbox, nms, scale_coords = gb
    img = np.ascontiguousarray(letterbox(img0, new_shape=img_size)[0][:, :, ::-1].transpose(2, 0, 1))  # BGR→RGB, CHW
    x = torch.from_numpy(img).to(device).float(); x /= 255.0; x = x.unsqueeze(0)
    det = nms(model(x)[0], conf, iou, multi_label=False, classes=None, agnostic=False)[0]
    if det is None or not len(det): return np.zeros((0, 6), np.float32)
    det[:, :4] = scale_coords(x.shape[2:], det[:, :4], img0.shape).round()
    return det.cpu().numpy()

def run(a):
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stems: stems = Path(a.stems).read_text().split()
    else: stems = label_only_stems(); (OUT / "label_only_100.txt").write_text("\n".join(stems) + "\n")
    out = Path(a.out) if a.out else REPORTS / f"preds_guidebook_{Path(a.weights).stem}_{a.cond}_labelonly100.csv"
    if DS.resolve() in out.resolve().parents: sys.exit("원본 폴더(제조AI데이터셋/)에는 쓰지 않는다")
    t0 = time.time(); gb = import_guidebook(); device = torch.device(a.device)
    ck = safe_load(a.weights)
    model = gb[0](str(GB / "yolov3-spp.cfg"), a.img_size)
    model.load_state_dict(ck["model"]); model.to(device).eval()   # strict: 키가 하나라도 어긋나면 멈춤
    paths = raw_paths(); t1 = time.time(); scores, n_img_box = [], 0
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["stem", "cx", "cy", "w", "h", "score"])
        for s in stems:
            det = detect(model, load_image(paths[s], a.cond), gb, a.img_size, device, a.conf, a.iou)
            n_img_box += len(det) > 0
            for x0, y0, x1, y1, sc, _ in det:
                w.writerow([s, f"{(x0 + x1) / 2:.1f}", f"{(y0 + y1) / 2:.1f}", f"{x1 - x0:.1f}", f"{y1 - y0:.1f}", f"{sc:.4f}"]); scores.append(float(sc))
    t2 = time.time(); bf = ck.get("best_fitness")
    log = dict(weights=Path(a.weights).name, cond=a.cond, device=str(device), torch=torch.__version__, img_size=a.img_size, conf=a.conf, iou=a.iou,
               ckpt_epoch=ck.get("epoch"), ckpt_best_fitness=None if bf is None else float(np.asarray(bf).ravel()[0]),
               n_images=len(stems), n_images_with_box=int(n_img_box), n_boxes=len(scores), n_boxes_ge_0_3=int(sum(v >= 0.3 for v in scores)),
               max_score=max(scores) if scores else None, sec_load=round(t1 - t0, 1), sec_infer=round(t2 - t1, 1), sec_total=round(t2 - t0, 1))
    out.with_name(out.stem + "_run.json").write_text(json.dumps(log, ensure_ascii=False, indent=2))
    print(json.dumps(log, ensure_ascii=False)); print("saved", out)

def write_summary():
    """xray_eval.py 가 남긴 채점 JSON과 실행 기록(_run.json)만으로 reports/guidebook_probe/summary.md 를 만든다."""
    J = lambda s: json.loads((REPORTS / s).read_text(encoding="utf-8"))
    f = lambda v: "-" if v is None else f"{v:.4f}"
    ta, runs = [], []
    for n in NS:
        for c in ["raw", "masked"]:
            s = f"preds_guidebook_last{n}_{c}_labelonly100"
            e, t, r = J(f"{s}_eval.json"), J(f"{s}_thr03_eval.json"), J(f"{s}_run.json"); runs.append(r)
            assert t["thr"] == 0.3 and e["n_images"] == t["n_images"] == r["n_images"]
            ta.append(f"| last{n} | {c} | {e['n_images']} | {e['n_gt']} | {f(e['precision'])} | {f(e['recall'])} | {f(e['f1'])} | {f(e['ap'])} | {f(e['fp_per_image'])} | {f(e['thr'])} "
                      f"| {f(t['precision'])} | {f(t['recall'])} | {f(t['f1'])} |")
    tb = []
    for name, inp, s in [("가이드북 last400", "raw", "preds_guidebook_last400_raw_labelonly100"), ("가이드북 last400", "masked", "preds_guidebook_last400_masked_labelonly100"),
                         ("YOLOv8n 640", "masked", "preds_v1_yolov8n_640_test"), ("YOLOv8s 1024", "masked", "preds_v1_yolov8s_1024_test"),
                         ("고전 DoG", "masked", "preds_classical_dog_test")]:
        e = J(f"{s}_common29_eval.json")
        tb.append(f"| {name} | {inp} | {e['n_images']} | {e['n_gt']} | {f(e['precision'])} | {f(e['recall'])} | {f(e['f1'])} | {f(e['ap'])} | {f(e['fp_per_image'])} | {f(e['thr'])} |")
    dev = ", ".join(sorted({r["device"] for r in runs})); sec = sum(r["sec_total"] for r in runs); r0 = runs[0]
    head = "| 가중치 | 입력 | 사진 | 정답 | 정밀도 | 재현율 | F1 | AP | 사진당 오탐 | 최고 F1 문턱값 | 정밀도@0.3 | 재현율@0.3 | F1@0.3 |\n|---|---|" + "---:|" * 11
    md = ["# 가이드북 실습 가중치를 처음 보는 사진에 돌려 본 결과", "",
          "주최 측 가이드북의 YOLOv3-SPP 실습 가중치(last15~last400)를, 어느 실습 폴더에도 없는 라벨 전용 100장에 돌렸다.",
          "raw는 장비가 그린 색 네모가 남은 원본(가이드북이 학습한 입력), masked는 우리 마스킹 함수로 색 네모를 지운 사진이다.",
          "채점은 scripts/xray_eval.py, 표의 숫자는 채점 JSON 값을 소수 넷째 자리로 반올림한 것이다.", "",
          "- 정밀도: 모델이 이물이라고 한 것 중 맞은 비율. 재현율: 실제 이물 중 찾아낸 비율. AP: 문턱값을 훑은 정밀도-재현율 곡선 아래 넓이.",
          "- 사진당 오탐: 문턱값에서 정답과 맞지 않은 박스 수 / 사진 수. 최고 F1 문턱값: F1이 가장 높은 점수 문턱값. @0.3: detect.py 기본 문턱값 0.3으로 고정.", "",
          "## 1. 가중치 × 입력 조건 (라벨 전용 100장)", "", head, *ta, "",
          "## 2. 공통 평가 29장 (라벨 전용 100장 중 우리 test 분할, 두 쪽 모두 학습에 안 쓴 사진)", "",
          "| 모델 | 입력 | 사진 | 정답 | 정밀도 | 재현율 | F1 | AP | 사진당 오탐 | 최고 F1 문턱값 |\n|---|---|" + "---:|" * 8, *tb, "",
          "## 실행 조건", "",
          f"- 장치 {dev}, torch {r0['torch']}, img-size {r0['img_size']}, conf {r0['conf']}, NMS iou {r0['iou']}",
          f"- 추론 12회(가중치 6개 × 입력 2개) 실행 시간 합계 {sec:.0f}초 (모델 불러오기 포함, 채점 시간 제외)",
          "- 숫자 출처: reports/preds_guidebook_*_labelonly100{,_thr03,_common29}_eval.json, reports/preds_*_test_common29_eval.json, 실행 기록 *_run.json"]
    (OUT / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8"); print("saved", OUT / "summary.md")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(WTS / "last400.pt")); ap.add_argument("--cond", choices=["raw", "masked"], default="raw")
    ap.add_argument("--stems", default=None, help="사진 이름 목록 파일(한 줄에 하나). 없으면 라벨 전용 100장")
    ap.add_argument("--img-size", type=int, default=512); ap.add_argument("--device", default="cpu")
    ap.add_argument("--conf", type=float, default=0.001); ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--out", default=None); ap.add_argument("--summary", action="store_true", help="채점 JSON을 모아 요약 표만 만든다")
    a = ap.parse_args()
    write_summary() if a.summary else run(a)

if __name__ == "__main__":
    main()
