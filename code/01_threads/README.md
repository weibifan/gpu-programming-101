# 01_threads — 线程模型（M2）

目标：搞懂 grid / block / thread 三层组织、线程索引计算，以及 1D/2D/3D 的用法。

| 文件 | 演示内容 |
|---|---|
| `index_1d.cu` | 一维网格：全局编号 `blockIdx.x*blockDim.x+threadIdx.x`，以及 N 非 blockDim 整数倍时的防越界 |
| `thread_print.cu` | device `printf` 打印每个线程自己看到的 `blockIdx/threadIdx/blockDim/gridDim`，感受层级与乱序 |
| `index_2d.cu` | 二维 block+grid 处理矩阵：从 `blockIdx.y` 解出 `row`，行主序压平 `idx = row*width+col` |
| `index_3d.cu` | 三维 block+grid 处理体数据：`(x,y,z)` → `idx=(z*H+y)*W+x` |

## 编译 / 运行

先加载 MSVC 环境（否则 `nvcc` 找不到 `cl.exe`/头文件/库，见 `docs/01_environment.md` §3.3）：

```powershell
cmd /c "\"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat\" && nvcc index_1d.cu -o index_1d.exe && .\index_1d.exe"

# 其余文件同理：把 index_1d 换成 thread_print / index_2d / index_3d
```

或用「Developer Command Prompt for VS2019」：

```cmd
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
nvcc index_1d.cu -o index_1d.exe && index_1d.exe
```

## 关键公式速记

```
全局编号(1D) = blockIdx.x * blockDim.x + threadIdx.x
col          = blockIdx.x * blockDim.x + threadIdx.x
row          = blockIdx.y * blockDim.y + threadIdx.y
idx(2D)      = row * width + col          # 行主序
idx(3D)      = (z*H + y) * W + x          # x 最快
```

> 小技巧：编译传 `-arch=sm_61`（GTX 1080）可生成针对该架构的代码；不传则用默认兼容架构。

## 编码说明

- 本仓库所有源文件均为 **UTF-8 编码**，`.cu` 里的中文字符串也是 UTF-8。
- 简体中文版 Windows 控制台默认 GBK（代码页 936），直接运行会把 UTF-8 字节的中文显示成乱码（**不影响结果，仅显示问题**）。
- 想正常显示，运行前切到 UTF-8 代码页：

```powershell
chcp 65001
.\index_1d.exe
```

对应文档：`docs/02_cuda_basics.md` §6.1 / §6.4。
