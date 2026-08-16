# =====================================================================
# 04_pytorch_gpu / ex02_tensor.py
# Tensor 的设备属性与数据流动（docs/04_pytorch_gpu.md §2~§3）
#
# 运行（有 GPU 的机器，如远端 Win10；本地无 GPU 会自动退回 CPU 演示）：
#   python ex02_tensor.py
#
# 演示内容：
#   1. 快速体检：torch.cuda 是否可用、显卡型号、内置 CUDA 版本
#   2. Tensor 的 .device 属性（cpu vs cuda）
#   3. .to('cuda') 搬数据，走 PCIe
#   4. CPU vs GPU 矩阵乘计时（x @ x.T -> 自动调 cuBLAS）
#   5. torch.set_default_device('cuda') 省事写法
# =====================================================================
import torch
import time


def device_name():
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"


def main():
    # ---- 1) 快速体检（对应 docs/04 §2.3）----
    print("== 环境体检 ==")
    print("cuda 可用          :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("显卡型号            :", torch.cuda.get_device_name(0))
        print("torch 内置 CUDA 版本:", torch.version.cuda)
    else:
        print("（本地无 NVIDIA GPU，以下演示自动走 CPU；.to('cuda') 相关步骤会跳过）")

    # ---- 2) Tensor 的 .device 属性（docs/04 §2.2）----
    print("\n== Tensor 住在哪 ==")
    x = torch.randn(1000, 1000)          # 默认在 CPU
    print("torch.randn 默认   :", x.device)
    if torch.cuda.is_available():
        x = x.to('cuda')                 # 搬到显存：cudaMalloc + cudaMemcpy
        print(".to('cuda') 之后  :", x.device)
        # 跨设备运算会报错或隐式搬运
        y_cpu = torch.randn(1000, 1000)
        try:
            _ = x + y_cpu                # 设备不一致
        except RuntimeError as e:
            print("跨设备相加报错     :", type(e).__name__)

    # ---- 3) CPU vs GPU 矩阵乘计时（docs/04 §2.1：x @ x.T -> cuBLAS）----
    print("\n== CPU vs GPU 矩阵乘 2048x2048 ==")
    a = torch.randn(2048, 2048)
    start = time.perf_counter()
    c_cpu = a @ a.T
    print(f"CPU 耗时          : {time.perf_counter() - start:.3f} s")

    if torch.cuda.is_available():
        a_gpu = a.to('cuda')
        # GPU 上的时间用 CUDA 事件量才准（含内核执行）
        torch.cuda.synchronize()
        start = time.perf_counter()
        c_gpu = a_gpu @ a_gpu.T          # 自动调 cuBLAS SGEMM
        torch.cuda.synchronize()
        print(f"GPU 耗时          : {time.perf_counter() - start:.3f} s")
        print(f"结果一致          : {torch.allclose(c_cpu, c_gpu.cpu(), atol=1e-2)}")

        # ---- 4) 设默认设备（docs/04 §2.3，2026 年的省事写法）----
        print("\n== torch.set_default_device('cuda') ==")
        torch.set_default_device('cuda')
        b = torch.randn(1000, 1000)      # 直接在显存上创建
        print("默认设备下新建    :", b.device)
        torch.set_default_device('cpu')  # 恢复默认，避免影响后续脚本


if __name__ == "__main__":
    main()
