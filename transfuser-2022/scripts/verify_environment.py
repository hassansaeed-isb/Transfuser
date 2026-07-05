#!/usr/bin/env python3
"""Verify GPU, Python packages, and CARLA API before Longest6 evaluation."""

from __future__ import annotations

import argparse
import importlib
import os
import sys


def check_import(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        print(f"[OK] import {module_name}")
        return True
    except Exception as exc:
        print(f"[FAIL] import {module_name}: {exc}")
        return False


def check_torch_cuda() -> bool:
    try:
        import torch

        print(f"[INFO] torch {torch.__version__}")
        if not torch.cuda.is_available():
            print("[FAIL] CUDA is not available to PyTorch")
            return False

        device_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        print(f"[OK] GPU: {device_name} (sm_{capability[0]}{capability[1]})")
        x = torch.randn(4, 4, device="cuda")
        _ = x @ x
        print("[OK] CUDA tensor matmul")
        return True
    except Exception as exc:
        print(f"[FAIL] torch CUDA check: {exc}")
        return False


def check_model_eval() -> bool:
    team_code = os.path.join(os.path.dirname(os.path.dirname(__file__)), "team_code_transfuser")
    if team_code not in sys.path:
        sys.path.insert(0, team_code)
    try:
        from model_eval import LidarCenterNet  # noqa: F401
        from crossvit_fusion import CrossViTFusionBackbone  # noqa: F401

        print("[OK] model_eval + crossvit_fusion")
        return True
    except Exception as exc:
        print(f"[FAIL] model_eval import: {exc}")
        return False


def check_carla(carla_root: str | None) -> bool:
    try:
        import carla

        print(f"[OK] carla Python API ({carla.__file__})")
        return True
    except Exception as exc:
        print(f"[FAIL] carla import: {exc}")
        if carla_root:
            dist = os.path.join(carla_root, "PythonAPI", "carla", "dist")
            print(f"[INFO] Expected wheel in: {dist}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--carla-root", default=os.environ.get("CARLA_ROOT", ""))
    args = parser.parse_args()

    ok = True
    ok &= check_torch_cuda()
    for module in ["cv2", "PIL", "timm", "shapely", "dictor", "tabulate"]:
        ok &= check_import(module)
    ok &= check_model_eval()
    ok &= check_carla(args.carla_root or None)

    if ok:
        print("\nAll checks passed.")
        return 0

    print("\nOne or more checks failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
