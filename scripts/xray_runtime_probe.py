#!/usr/bin/env python3
"""현재 기기의 PyTorch 학습 장치와 작은 순전파·역전파 연산을 점검한다.

데이터셋이나 모델 가중치를 읽지 않고, 프로그램을 설치하지 않는다.
사용: .venv/bin/python scripts/xray_runtime_probe.py --out reports/decision_review_20261001/runtime.json
"""
import argparse
import json
from pathlib import Path
import platform
import subprocess

import torch

ROOT = Path(__file__).resolve().parents[1]


def command(args):
    result = subprocess.run(args, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else {"error": result.stderr.strip()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.manual_seed(0)
    hardware = {key: command(["/usr/sbin/sysctl", "-n", key]) for key in ["hw.model", "hw.memsize", "machdep.cpu.brand_string", "hw.physicalcpu", "hw.logicalcpu"]}
    displays = command(["/usr/sbin/system_profiler", "SPDisplaysDataType", "-json"])
    if isinstance(displays, str):
        hardware["gpu"] = [{k: item[k] for k in ["sppci_model", "spdisplays_cores", "spdisplays_metal", "spdisplays_vendor"] if k in item} for item in json.loads(displays).get("SPDisplaysDataType", [])]
    else:
        hardware["gpu"] = displays
    report = {
        "hardware": hardware,
        "macos": platform.mac_ver()[0], "arch": platform.machine(),
        "python": platform.python_version(), "torch": torch.__version__,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "cuda_built": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
        "device_checks": {},
    }
    devices = ["cpu"] + (["mps"] if report["mps_available"] else []) + (["cuda"] if report["cuda_available"] else [])
    for device in devices:
        result = {}
        try:
            network = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3), torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(1)).to(device)
            values = network(torch.rand(2, 3, 32, 32, device=device))
            loss = values.square().mean()
            loss.backward()
            if device == "mps":
                torch.mps.synchronize()
            result["conv_forward_backward_finite"] = bool(torch.isfinite(loss).item() and all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in network.parameters()))
            result["tensor_device"] = str(values.device)
        except Exception as exc:
            result["conv_error"] = f"{type(exc).__name__}: {exc}"
        try:
            from torchvision.ops import nms
            boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]], device=device)
            selected = nms(boxes, torch.tensor([0.9, 0.8], device=device), 0.5)
            result["torchvision_nms"] = selected.cpu().tolist()
        except Exception as exc:
            result["torchvision_nms_error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
        report["device_checks"][device] = result
    report["recommended_device"] = next((device for device in ["mps", "cuda", "cpu"] if report["device_checks"].get(device, {}).get("conv_forward_backward_finite")), None)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
