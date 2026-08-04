# 06 PyTorch GPU：从 Python 到 GPU 计算

> 对应里程碑 M4。前面五篇都在讲 GPU 底层（CUDA 内核、显存、性能分析），本篇回到**日常使用的 Python**：怎么用高级语言驱动 GPU？有哪些层次可选？最终落在最常用的 **PyTorch** 上，把它提供的 GPU 能力（设备管理、混合精度、编译优化、多卡）讲清楚。
>
> 本篇由旧版 02_cuda_basics.md 的「基于 Python 的 GPU 编程」「PyTorch 中的 GPU 编程」两节合并重组而来，作为仓库的"Python 应用层"篇。

**阅读路线**：先建立"用 Python 驱动 GPU 的三个层次"的整体观（§1），再进入最上层 PyTorch——看清它的黑盒（§2）和数据流动（§3），最后掌握 PyTorch 的三大 GPU 提速手段：**AMP（§4）、torch.compile（§5）、多卡（§6）**。

---

## 1. 用 Python 写 GPU 编程：三条路径

### 1.1 为什么用 Python？

直接写 CUDA C 很麻烦：

* 要手动管理显存分配/释放
* 要手动做 CPU↔GPU 数据传输
* 要理解复杂的线程模型
* 编译和调试困难

**Python 把这四件事封装了**——代价是"离硬件更远"。所以关键是选对层次。

### 1.2 三个层次：从"写内核"到"完全托管"

```
最底层：PyCUDA / Numba —— 写 CUDA kernel，仍需理解线程模型
中间层：CuPy          —— 提供 numpy 风格的 API，自动生成内核
最上层：PyTorch       —— 完全不用手动管理 GPU，自动调度内核
```

| 层次 | 代表库 | 你要做什么 | 要懂线程模型吗 | 用途 |
|---|---|---|---|---|
| 最底层 | PyCUDA / Numba | 手写 CUDA kernel | ✅ 需要 | 研究/自定义算子 |
| 中间层 | CuPy | 像写 NumPy 一样写 GPU 数组 | ❌ 基本不用 | 把 NumPy 科学计算加速 |
| 最上层 | PyTorch / TF | 只管模型和数据，GPU 全自动 | ❌ 不用 | 深度学习全流程 |

> 本仓库主线：**先用 CUDA C 学原理（docs/02~04），再用 PyTorch 做应用（本篇）**。PyCUDA/Numba 是"想在自己代码里手写内核但不想用 C 编译"时的选择，了解即可。

### 1.3 中间层示例：Numpy（CPU）vs CuPy（GPU）

```python
# CPU 版本（用 numpy）
import numpy as np
a = np.random.randn(1000000)
b = np.random.randn(1000000)
c = a + b           # CPU 串行计算

# GPU 版本（用 cupy）
import cupy as cp
a = cp.random.randn(1000000)  # 直接在显存创建
b = cp.random.randn(1000000)
c = a + b                     # GPU 并行计算
```

**API 几乎一样**，但 CuPy 的 `+` 在 GPU 上并行执行——它自动帮你管理了显存和内核启动。

### 1.4 最底层示例：Numba 手写 CUDA kernel

Numba 让你在 Python 里直接写"CUDA 内核"（等价于前面 CUDA C 篇的 `__global__` 函数），且无需 C 编译：

```python
from numba import cuda
import numpy as np

@cuda.jit
def vec_add_kernel(a, b, c):
    i = cuda.grid(1)   # 获取全局线程 ID（等价于 threadIdx.x + blockIdx.x*blockDim.x）
    if i < a.size:
        c[i] = a[i] + b[i]

n = 1000000
a = np.random.randn(n).astype(np.float32)
b = np.random.randn(n).astype(np.float32)
c = np.zeros_like(a)

d_a = cuda.to_device(a)   # CPU → 显存
d_b = cuda.to_device(b)
d_c = cuda.to_device(c)

threads_per_block = 256
blocks = (n + threads_per_block - 1) // threads_per_block
vec_add_kernel[blocks, threads_per_block](d_a, d_b, d_c)   # 启动内核

d_c.copy_to_host(c)       # 显存 → CPU
```

> 对照 02_cuda_basics.md §4.3 的 CUDA C 版本，你会发现**结构一模一样**：准备数据 → 搬到显存 → 启动网格 → 搬回。Numba 只是把 `.cu` 编译藏进了 `@cuda.jit` 装饰器。

---

## 2. PyTorch：GPU 编程的"黑盒"

### 2.1 一段代码背后发生了什么

PyTorch 把 GPU 编程简化到极致，你基本不需要知道 CUDA 的存在：

```python
import torch

x = torch.randn(1000, 1000)   # ① 分配 CPU 内存
x = x.to('cuda')              # ② 搬到显存：cudaMalloc + cudaMemcpy
y = x @ x.T                   # ③ 矩阵乘法 → 自动调用 cuBLAS
z = y.softmax(dim=-1)         # ④ softmax → 自动调用 CUDA softmax kernel
z = z.to('cpu')               # ⑤ 搬回 CPU 内存
```

**背后的魔法**：每一个算子都是一次内核启动，PyTorch 帮你选好库、管好显存、安排数据流。第 ③④ 步到底发生了什么，正是前面所有 CUDA 笔记的内容。

### 2.2 你不需要写 CUDA 代码

PyTorch 内部**已经包含了数千个**写好的 CUDA kernel：

| 运算 | 使用的 CUDA 库 |
|---|---|
| 矩阵乘法 | cuBLAS |
| 卷积 | cuDNN |
| 注意力 | FlashAttention（详见 05_llm_acceleration.md §3） |
| 各种激活函数 | 手写 CUDA kernel |
| 随机数生成 | cuRAND |

**作为 PyTorch 用户，你只需要**：

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = GPT(config).to(device)

for x, y in dataloader:
    x, y = x.to(device), y.to(device)
    loss = model(x, y)
    loss.backward()
    optimizer.step()
```

> 认知要点：**PyTorch 不是"比 CUDA 快"，而是"把 CUDA 封装到你感觉不到"**。出了性能问题，04_performance.md 的 memory-bound/带宽判断，能帮你定位是"算法不行"还是"库已经尽力了"。

---

## 3. 设备与数据流动：三大关键操作

数据在 CPU 内存和 GPU 显存之间搬一次要过 **PCIe 总线**，有延迟、有带宽上限。所有 GPU 代码的性能纪律都来自这三条：

```
1. 数据从 CPU → GPU   x.to('cuda')   走 PCIe，慢 → 尽量一次多传
2. 在 GPU 上计算      model(x)       全程在显存内，极快
3. 结果从 GPU → CPU   loss.item()    走 PCIe，慢 → 训练循环里少做
```

### 3.1 Tensor 的设备属性

PyTorch 的数据结构叫 **Tensor**，每个 Tensor 都有 `.device` 属性，决定它住在 CPU 内存还是显存：

```python
x = torch.randn(1000, 1000)      # 默认在 CPU
print(x.device)                  # cpu
x = x.to('cuda')                 # 搬到显存
print(x.device)                  # cuda:0
```

> ⚠️ 跨设备运算会报错或触发隐式搬运。写"搬到 GPU 再算"的顺序，避免代码在 CPU 上偷偷跑。

### 3.2 2026 年的省事写法：设默认设备

```python
torch.set_default_device('cuda')          # 之后创建的 Tensor 直接在显存上
a = torch.randn(1000, 1000)               # 不用 .to('cuda') 了
```

### 3.3 快速检查环境

```python
import torch
print(torch.cuda.is_available())          # 有 NVIDIA GPU + CUDA 驱动吗
print(torch.cuda.get_device_name(0))      # 显卡型号
print(torch.version.cuda)                 # PyTorch 内置 CUDA 版本
```

> 远端 Win10 体检结果：`torch.cuda.is_available()=True`（torch 2.7.1+cu118），500×500 矩阵乘实测 OK（详见 01_environment.md；可用 `tools/check_gpu.py` 复测）。

---

## 4. 自动混合精度（AMP）：让"速度"和"精度"同时保住

### 4.1 痛点

```
痛点 1：显存不够用 —— FP32 一个数占 4 字节，大模型存不下
痛点 2：算得不够快 —— 低精度（FP16/BF16）的 GPU 算力约为 FP32 的 2 倍
```

低精度的问题：

```
小数值被"截断"：梯度极小（<1e-7）在 FP16 下变 0
大数值"溢出"：FP16 最大 65504，稍大就变 Inf
```

### 4.2 思路：不是所有计算都需要 FP32

* 精度敏感（用 FP32）：损失计算、LayerNorm / softmax、优化器累加
* 精度不敏感（用 BF16/FP16）：矩阵乘法（Linear、Attention 的 QKV 投影）

**用 `autocast` 让 PyTorch 自动挑选精度**：

```python
# PyTorch 2.0+ 的自动混合精度
with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
    loss = model(x, y)
```

### 4.3 FP16 vs BF16

| 对比 | FP16 | BF16 |
|---|---|---|
| 指数位 | 5 位（最大 65504） | 8 位（和 FP32 相同，范围大） |
| 尾数位 | 10 位（精度较高） | 7 位（精度较低） |
| 结论 | 容易溢出/截断，需 GradScaler | 范围大，无需 GradScaler |

> 底层视角：这就是 00_why_gpu.md §2.2 的浮点表示——**半精度省一半字节 = 省一半带宽 = 吞吐翻倍**，正好对应 04_performance.md 的 memory-bound 直觉。量化（05 篇 §4）是同一思路的再进一步。

---

## 5. `torch.compile`：一键编译优化

`torch.compile` 把整个 forward 当作整体优化，**把能合并的算子融合成一个 kernel**：

* 单卡内部算得更快（减少 kernel 启动和显存读写）
* 平均提速 10-20%，业务代码不用改
* 代价：首次编译几十秒，输入形状需固定

```python
model = torch.compile(model)     # 一行搞定
# 指定后端：torch.compile(model, backend="inductor")
```

> 底层视角：这就是"手写算子融合"的自动化版——02_cuda_basics.md §6.1 讲过每次启动 kernel 都有加载/调度开销，`torch.compile` 把这些小 kernel 合成一个大 kernel，一次启动、少搬中间量。与 04_performance.md 的 tiling 是"底层 vs 高层"两种手段。

---

## 6. 多卡：单卡装不下 / 跑不动怎么办？

```
数据并行（DDP）：每卡一份完整模型，各算各的数据 → 目标是"加速"
模型并行：把模型切多份分给不同卡 → 目标是"装得下"
```

**DDP（标准做法）**：

```bash
torchrun --nproc_per_node=8 train.py
```

```python
model = nn.parallel.DistributedDataParallel(model)
```

**DDP 工作原理**：每个 GPU 处理不同数据 → 各算梯度 → all-reduce 求平均 → 各自更新参数。

> DDP 与 05 篇 §6 的连续批处理是两回事：DDP 把**训练**摊到多卡，vLLM 把**推理服务**的吞吐拉高。底层都是"让更多 SM 干活"。

---

## 7. 本篇与全仓库的对照

| 本篇概念 | 底层对应（前面文档） |
|---|---|
| `.to('cuda')` 搬数据 | 02 §4：PCIe 搬运、host/device 双内存 |
| `x @ x.T` 调 cuBLAS | 04 §3~§5：矩阵乘 tiling 的最终实现者 |
| `F.scaled_dot_product_attention` | 05 §3：FlashAttention 三种写法 |
| AMP / 量化 | 00 §2：浮点表示；04 §2：带宽节省 |
| Numba 内核 | 02 §4：grid/block/thread 同款结构 |

> 到这里，"GPU 底层原理 → CUDA 内核 → 性能优化 → Python/PyTorch 应用"的主线闭环了。想继续深入应用层，可再复习 05_llm_acceleration.md（LLM 加速），或直接跑 `tools/check_gpu.py` 在真实显卡上验证本篇所有代码。
