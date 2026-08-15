# =====================================================================
# 08_cuda_libs / 05_torch_backends.py
# 观察 PyTorch 与 CUDA 类库的关系：torch 把算子分派给 cuBLAS/cuDNN/cuFFT/cuRAND
#
# 运行（需 GPU 机器，如远端 Win10）：
#   python 05_torch_backends.py
#
# 对应文档：docs/02_cuda_basics.md §9.1 / §9.2
# =====================================================================
import torch

if not torch.cuda.is_available():
    print("无可用 GPU，本脚本需要 CUDA 设备（本地无 GPU 时请到远端/云端运行）")
    raise SystemExit(1)

print("torch 版本       :", torch.__version__)
print("torch 内置 CUDA  :", torch.version.cuda)          # 与系统 nvcc 版本可不同
print("cuDNN 版本       :", torch.backends.cudnn.version())  # PyTorch 自带的副本
print("GPU              :", torch.cuda.get_device_name(0))

# 每个算子背后的 CUDA 类库（对应 docs/02 §7.2）：
#   x @ y / torch.mm    -> cuBLAS
#   nn.Conv2d 等        -> cuDNN
#   torch.fft.fft       -> cuFFT
#   torch.randn/rand    -> cuRAND（或 PyTorch 自研内核）

x = torch.randn(2048, 2048, device="cuda")
y = x @ x                                        # -> cuBLAS（矩阵乘）

img = torch.randn(1, 3, 64, 64, device="cuda")
w = torch.randn(16, 3, 3, 3, device="cuda")
c = torch.nn.functional.conv2d(img, w, padding=1)  # -> cuDNN（卷积）

f = torch.fft.fft(torch.randn(4096, device="cuda"))  # -> cuFFT（傅里叶）

print("matmul / conv2d / fft 已在 GPU 上执行成功")
print("说明：PyTorch 把算子分派给这些类库，自带副本，与系统 CUDA Toolkit 无关")
