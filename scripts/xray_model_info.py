#!/usr/bin/env python3
"""
모델 후보의 크기와 속도 비교. 설계 파일(YAML)로 클래스 1개짜리 모델을 만들어 파라미터 수와
640×640 입력 한 장의 순전파 시간(중앙값)을 잰다. 가중치는 내려받지 않는다(무작위 가중치로도 시간은 같다).
순전파 시간에는 전처리와 NMS(겹친 박스 정리)가 빠져 있다. 다른 무거운 작업이 돌 때 재면 값이 흔들린다.
출력: reports/model_info.csv
사용: .venv/bin/python scripts/xray_model_info.py [--runs 30]
"""
import argparse, time, platform
from pathlib import Path
import numpy as np, pandas as pd, torch
ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = [  # (이름, 설계 파일, 설명)
    ("YOLOv3-SPP", "yolov3-spp.yaml", "가이드북과 같은 계열(Ultralytics 재구현, 머리 부분은 최신식)"),
    ("YOLOv3-tiny", "yolov3-tiny.yaml", "YOLOv3의 경량판"),
    ("YOLOv8n", "yolov8n.yaml", "지금 쓰는 경량 1단계 탐지 모델"),
    ("YOLOv8s", "yolov8s.yaml", "YOLOv8의 한 단계 큰 판"),
    ("YOLOv8n-P2", "yolov8n-p2.yaml", "작은 물체용 고해상도(4배 축소) 탐지층 추가"),
    ("YOLO11n", "yolo11n.yaml", "YOLOv8 후속 경량 모델"),
    ("YOLO26n", "yolo26n.yaml", "설치된 Ultralytics의 최신 경량 모델"),
    ("RT-DETR-L", "rtdetr-l.yaml", "트랜스포머 기반 탐지 모델"),
]

def build(cfg):
    from ultralytics.nn.tasks import DetectionModel, RTDETRDetectionModel
    m = (RTDETRDetectionModel if "rtdetr" in cfg else DetectionModel)(cfg, nc=1, verbose=False)
    return m.eval()

def latency(model, device, runs):
    model = model.to(device); x = torch.zeros(1, 3, 640, 640, device=device); ts = []
    with torch.no_grad():
        for i in range(runs + 5):
            if device == "mps": torch.mps.synchronize()
            t0 = time.perf_counter(); model(x)
            if device == "mps": torch.mps.synchronize()
            if i >= 5: ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", type=int, default=30); a = ap.parse_args()
    torch.manual_seed(0); rows = []
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []) + (["cuda"] if torch.cuda.is_available() else [])
    for name, cfg, desc in CANDIDATES:
        m = build(cfg); params = sum(p.numel() for p in m.parameters()) / 1e6
        row = dict(model=name, cfg=cfg, params_M=round(params, 2), desc=desc)
        for d in devices: row[f"ms_{d}"] = round(latency(m, d, a.runs), 1)
        rows.append(row); print(row)
    df = pd.DataFrame(rows); out = ROOT / "reports" / "model_info.csv"; df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"장치: {devices}, CPU: {platform.processor() or platform.machine()}, torch {torch.__version__}\nsaved", out)

if __name__ == "__main__":
    main()
