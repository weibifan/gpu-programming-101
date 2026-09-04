#!/usr/bin/env python3
"""GPU 环境检测脚本：驱动、CUDA Toolkit、PyTorch CUDA。

用法：
    python tools/check_gpu.py
"""

import platform
import shutil
import subprocess
import sys


def run(cmd):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or out.stderr.strip()
    except FileNotFoundError:
        return "(未找到命令)"


def check_nvidia_driver():
    print("== NVIDIA 驱动 ==")
    print(run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"]))


def check_cuda_toolkit():
    print("\n== CUDA Toolkit (nvcc) ==")
    print(run(["nvcc", "--version"]))


def check_pytorch():
    print("\n== PyTorch ==")
    try:
        import torch
        print(f"PyTorch 版本: {torch.__version__}")
        print(f"CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"当前 GPU: {torch.cuda.get_device_name(0)}")
            print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print(f"CUDA 运行时版本: {torch.version.cuda}")
            print(f"计算能力: {torch.cuda.get_device_capability(0)}")
    except ImportError:
        print("PyTorch 未安装")


def main():
    print(f"主机: {platform.node()}  ({platform.platform()})")
    print(f"Python: {sys.version.split()[0]}")

    if shutil.which("nvidia-smi"):
        check_nvidia_driver()
    else:
        print("== NVIDIA 驱动 ==")
        print("未检测到 nvidia-smi，本机可能没有 NVIDIA GPU（本地无 GPU 属正常）")

    check_cuda_toolkit()
    check_pytorch()


if __name__ == "__main__":
    main()
