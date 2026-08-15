# 00\_hello — 第一个 CUDA 程序（M1）

目标：跑通「写 kernel → 编译 → 运行 → 校验」的完整流程。

* `vector_add.cu`：两向量相加。演示 host/device 双内存、`cudaMalloc`/`cudaMemcpy`、`<<<grid, block>>>` 启动、`cudaGetLastError` 错误检查、`cudaEvent` 计时、与 CPU 结果比对。

## 编译 / 运行

远端 Win10（需 MSVC + CUDA 11.6，见 `docs/01_environment.md`）：

```powershell
# 先加载 MSVC x64 环境（否则 nvcc 找不到 cl.exe/头文件/库），再编译
cmd /c "\"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat\" && nvcc vector_add.cu -o vector_add.exe"

.\vector_add.exe                    # 默认 1<<20 个元素
.\vector_add.exe 1000               # 可指定元素个数
```

或直接用「Developer Command Prompt for VS2019」后：

```cmd
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
nvcc vector_add.cu -o vector_add.exe
```

预期输出：kernel 耗时（毫秒）、`最大误差 ≈ 0`、`校验通过`。

<br />

## 流程图

```
CPU 内存准备数据 → cudaMemcpy(→显存) → vec_add<<<blocks, threads>>> → cudaMemcpy(←显存) → 比对校验
```

对应文档：`docs/02_cuda_basics.md` §5.2 / §6.1。

## 核心一行：全局线程编号

```cuda
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

这行是 CUDA 最核心的"线程身份换算"——把线程在网格中的位置，换算成它在整个任务里的**全局唯一编号**。执行 kernel 的每个线程都能看到三个内置变量：

| 变量 | 含义 | 例子 |
|---|---|---|
| `blockIdx.x` | 我在第几个 block（0 起） | block 2 |
| `blockDim.x` | 每个 block 有多少线程 | 256 |
| `threadIdx.x` | 我在 block 内是第几个线程（0 起） | 30 |

为什么要 `blockIdx.x × blockDim.x`？因为每个 block 负责一段连续的编号区间：block 0 负责 `0~255`，block 1 负责 `256~511`，block 2 负责 `512~767`。所以 block 2 的起点 = `2 × 256 = 512`，加上块内偏移 `threadIdx.x = 30`，得到全局编号 **542**。

它的目的是**一一对应数据**——每个线程处理一份数据：

```cuda
if (i < n) {              // 防越界
    c[i] = a[i] + b[i];   // 线程 i 只处理下标为 i 的元素
}
```

- 线程 0 算 `c[0]`，线程 542 算 `c[542]`……各干各的，天然并行
- 若 `n` 不能被 `blockDim` 整除（如 N=510、blockDim=128 → 4 blocks=512 线程），越界的线程靠 `if (i < n)` 拦下——同款演示见 `code/01_threads/index_1d.cu`

> 一句话：这行把"我该处理哪个数据"从线程坐标换算出来。要并行多少份数据就铺多少个线程，公式不变。

## 编码说明

* 本仓库所有源文件均为 **UTF-8 编码**，`.cu` 里的中文字符串也是 UTF-8。

* 简体中文版 Windows 控制台默认 GBK（代码页 936），直接运行会把 UTF-8 字节的中文显示成乱码（**不影响结果，仅显示问题**）。

* 想正常显示，运行前切到 UTF-8 代码页：

```powershell
chcp 65001
.\vector_add.exe
```

