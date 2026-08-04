# gpu-programming-101 🚀

> GPU 编程入门 — 从 CUDA 内核原理到大模型推理/训练加速的完整实践路线。

## 📖 概述

本仓库从 **GPU 硬件原理 → CUDA C/C++ 内核 → 性能优化 → PyTorch GPU → LLM 加速（FlashAttention/量化/KV cache/推理引擎）** 逐步深入，代码均含中文注释。

学习方式：**学 1 个概念 → 写 1 篇笔记 → 写 1 个最小代码 → 跑通 → commit**。

## 🖥️ 环境

| 机器 | 系统 | GPU | 用途 |
|---|---|---|---|
| 本地 PC | — | 无 NVIDIA GPU | 写代码、整理笔记 |
| 远端 PC | Windows 10 | GTX 1080 8G | CUDA 原理线实验（M1~M4） |
| AutoDL 云 | Linux | A100 / RTX 4090D | LLM 线实验（M5~M6） |

开发方式：本地 VS Code Remote-SSH 写代码/调试，远端 GPU 上执行；AutoDL 同理。

## 📂 目录结构

```
gpu-programming-101/
├── README.md                  # 本文件
├── LICENSE
├── docs/                      # 学习笔记
│   ├── 00_why_gpu.md          # CPU vs GPU、SIMT、硬件架构
│   ├── 01_environment.md      # CUDA Toolkit、驱动、nvidia-smi、torch.cuda
│   ├── 02_cuda_basics.md      # kernel、grid/block/thread
│   ├── 03_memory.md           # 内存层级、合并访问
│   ├── 04_performance.md      # occupancy、tiling、Nsight 分析
│   ├── 05_llm_acceleration.md # LLM 加速专题
│   ├── 06_pytorch_gpu.md      # PyTorch GPU：Python 驱动 GPU + 提速三板斧
│   └── 07_triton.md           # Triton：用 Python 写高性能内核
├── code/
│   ├── 00_hello/              # 第一个 CUDA 程序 vector_add
│   ├── 01_threads/            # 线程模型
│   ├── 03_memory/             # 内存模型
│   ├── 04_optimization/       # 性能优化（矩阵乘 tiling）
│   ├── 05_llm/                # LLM 加速实验
│   ├── 06_pytorch_gpu/        # PyTorch GPU 示例（旧仓库迁移）
│   ├── 07_triton/             # Triton 内核
│   └── 08_cuda_libs/          # cuBLAS / cuDNN / cuFFT / cuRAND 案例
├── tools/
│   └── check_gpu.py           # GPU 环境检测脚本
└── data/                      # 数据目录
```

## 🗺️ 学习路径

| 里程碑 | 内容 | 状态 |
|---|---|---|
| M0 | 仓库骨架、README、迁移旧笔记 | ✅ 完成 |
| M1 | 环境：远端 Win10 装驱动/CUDA/MSVC/PyTorch，打通 SSH，vector_add | ⏳ 进行中 |
| M2 | CUDA 基础：线程模型、内存、同步 | ⬜ |
| M3 | 优化：矩阵乘 naive→shared→tiling，Nsight | ⬜ |
| M4 | PyTorch GPU：迁移旧示例 + torch.compile | ⬜ |
| M5 | LLM 专题：Attention/FlashAttention/量化/KV cache | ⬜ |
| M6 | 云端实战：A100/4090D 上 FlashAttention、FP16 训练、llama.cpp/vLLM 部署、QLoRA 微调 | ⬜ |

## 🚀 快速开始

```bash
# 检测环境
python tools/check_gpu.py

# 第一个 CUDA 程序（需在装有 CUDA 的机器上）
cd code/00_hello
nvcc vector_add.cu -o vector_add
./vector_add
```

## 📖 参考资源

- [NVIDIA CUDA 编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA 官方教程](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [PyTorch 官方文档](https://pytorch.org/docs/stable/)
- [Triton 文档](https://triton-lang.org/)
