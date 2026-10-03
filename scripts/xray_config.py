"""공통 실행 기준: v2 기본 데이터, 고정 분할 확인값, 학습 장치 선택."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_NAME = os.environ.get("XRAY_DATA", "xray_v2")
if DATA_NAME not in {"xray_v1", "xray_v2"}:
    raise ValueError("XRAY_DATA는 xray_v2(기본) 또는 xray_v1이어야 합니다")
DATA = ROOT / "data" / DATA_NAME
V2_SPLIT_MD5 = "8e58184ae24dfd1b48da2e9dfd88fedc"


def select_device(requested="auto"):
    """명시한 장치는 유지하고 auto는 mps → cuda → cpu 순으로 선택한다."""
    if requested != "auto":
        return requested
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"
