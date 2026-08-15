# 08_cuda_libs — CUDA 类库案例

配套文档：`docs/02_cuda_basics.md` §9.1（四大基础类库）、§9.2（与 PyTorch 的关系）。

| 文件 | 类库 | 演示内容 |
|---|---|---|
| `01_cublas_sgemm.cu` | cuBLAS | 矩阵乘法（列主序 + CPU 参考对比） |
| `02_curand_generate.cu` | cuRAND | 在显存批量生成均匀随机数 |
| `03_cufft_fft.cu` | cuFFT | 1D 复数 FFT，观察余弦频谱 |
| `04_cudnn_conv.cu` | cuDNN | 前向卷积（描述符 + 自动选算法） |
| `05_torch_backends.py` | torch | 观察算子分派到哪些 CUDA 类库 |

## 编译运行（远端 Win10，CUDA 11.6）

```bash
# cuBLAS / cuRAND / cuFFT 随 CUDA Toolkit 自带，直接链接即可
nvcc 01_cublas_sgemm.cu     -o 01_cublas_sgemm     -lcublas
nvcc 02_curand_generate.cu  -o 02_curand_generate  -lcurand
nvcc 03_cufft_fft.cu        -o 03_cufft_fft        -lcufft

# cuDNN 不随 CUDA Toolkit 提供，需先单独安装（见下方备注）
nvcc 04_cudnn_conv.cu       -o 04_cudnn_conv       -lcudnn

# PyTorch 自带这些库，无需手动链接
python 05_torch_backends.py
```

## 备注

- 本机体检结果：系统级 `cudnn64.dll` 未安装，PyTorch 内置 cuDNN 90100 可用（见 01_environment.md）。
  所以 `04_cudnn_conv` 若要编译，需先装系统 cuDNN（https://developer.nvidia.com/cudnn，选与 CUDA 11.6 匹配的版本）；
  若只是用 PyTorch，则完全无需安装。
- 矩阵乘 / 卷积等"重量算子"的性能王者是这些库（cuBLAS/cuDNN）。`code/04_optimization/` 手写内核是为了理解原理，
  生产环境直接用它们即可。
