# 04_optimization — 性能优化（M3）

目标：亲手跑一遍矩阵乘 `naive → shared → tiling(register)` 的优化全过程，再用 Nsight Compute 读出数字验证。

| 文件 | 版本 | 每线程 | 说明 |
|---|---|---|---|
| `sgemm_naive.cu` | naive | 算 1 个输出 | 全局内存读 O(N³)，严重 memory-bound |
| `sgemm_shared.cu` | shared tiling | 算 1 个输出 | 32x32 tile 进共享内存，全局读减 32 倍；行长 `TILE+1` 防 bank conflict |
| `sgemm_tiled.cu` | tiling + register | 算 4x4 = 16 个输出 | 每 block 覆盖 64x64；As/Bs 元素读到寄存器反复用，共享内存压力再降 |

## 编译 / 运行

先加载 MSVC 环境（见 `docs/01_environment.md` §3.3）：

```powershell
cmd /c "\"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat\" && nvcc sgemm_naive.cu -o sgemm_naive.exe -arch=sm_61 && .\sgemm_naive.exe"
# shared / tiled 同理，替换文件名即可；可用第三参指定边长如 .\sgemm_naive.exe 2048 2048 2048
```

三个版本分别跑一遍，对比 kernel 耗时（同一个程序里只跑单版本；做严谨对比可把三版 kernel 放进一个文件里跑，排除启动抖动，见 `docs/04_performance.md` §7）。

## 用 Nsight Compute 读数字（远端 Win10 已装 ncu）

```powershell
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active .\sgemm_naive.exe
```

预期趋势（`docs/04_performance.md` §6.4~§6.5）：

| 版本 | SM 吞吐 | DRAM 吞吐 | 结论 |
|---|---|---|---|
| naive | ~10% | ~95% | memory-bound，占用率已 100%，只能上 tiling |
| shared | ~40% | ~40% | 瓶颈从带宽转向计算 |
| tiled | 更高 | 更低 | 逼近 compute-bound |

## 要点回顾

* **四板斧层层递进**（`docs/04_performance.md` §5）：合并访问 → 共享内存复用（tiling）→ 寄存器复用（register tiling）→ 占用率调优。
* **bank conflict**：共享内存行长别是 32 的倍数，`TILE+1` padding 即可天然错开。
* **`__syncthreads()` 位置**：拷入 → 同步 → 计算 → 同步，漏一个结果就可能错。
* 手写 SGEMM **永远跑不过 cuBLAS**（cuBLAS 还有 Tensor Core 等大招，见 `code/08_cuda_libs/01_cublas_sgemm.cu`），手写价值在于看懂每一招为什么有效。

对应文档：`docs/03_memory.md` §4、`docs/04_performance.md` §1~§6。
