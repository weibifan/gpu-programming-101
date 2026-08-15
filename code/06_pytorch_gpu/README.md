# 06_pytorch_gpu — PyTorch GPU 应用层（M4）

目标：把 `docs/06_pytorch_gpu.md` 里的 PyTorch GPU 手段逐个跑一遍。

| 文件 | 演示内容 | 对应文档 |
|---|---|---|
| `ex02_tensor.py` | Tensor 的设备属性、`.to('cuda')`、CPU/GPU 计时、`set_default_device` | §2~§3 |
| `ex11_fitting.py` | 小 MLP 拟合 y=sin(x)：完整 GPU 训练循环 | §2.2 |
| `ex14_LeNet.py` | LeNet-5 在 MNIST 上训练（CNN 全流程 + DataLoader 搬数据） | §2.2 |
| `ex20_amp.py` | AMP：FP32 vs FP16(autocast + GradScaler) 加速比 | §4 |
| `ex21_torch_compile.py` | `torch.compile`：eager vs 编译后逐层加速比 | §5 |

## 运行

```powershell
# 远端 Win10（有 GPU）
python ex02_tensor.py
python ex11_fitting.py
python ex14_LeNet.py          # 首次运行会下载 MNIST 数据集到 ./data
python ex20_amp.py
python ex21_torch_compile.py  # 首次编译约几十秒

# 本地无 GPU 也能跑：自动降级 CPU 演示，但看不到加速效果
```

## 环境说明

* torch 2.7.1+cu118（远端 Win10）。环境体检与安装见 `docs/01_environment.md`。
* GTX 1080（sm_61）**不支持 BF16**（需 Ampere sm_80+），所以 `ex20_amp.py` 用 FP16 + GradScaler；A100/4090D 上可改 `torch.bfloat16`。
* `ex14_LeNet.py` 首次运行需联网下载 MNIST。

## 要点回顾

* `.to('cuda')` 走 PCIe，慢 → 一次多传；训练循环里少把结果搬回 CPU。
* PyTorch 内部数千个写好的 CUDA kernel 在替你干活（矩阵乘→cuBLAS、卷积→cuDNN、注意力→FlashAttention、随机→cuRAND），底层原理见 `docs/02_cuda_basics.md` §9.1~§9.2 和 `code/08_cuda_libs/`。
* AMP 与量化都是"省字节 = 省带宽"（`docs/04_performance.md` §2 的 memory-bound 直觉）；`torch.compile` 是"算子融合"的自动化版。
