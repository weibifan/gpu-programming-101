# 04 性能优化：占用率、Tiling 与 Nsight 分析

> 对应里程碑 M3。03_memory.md 讲清了"数据放哪、怎么访存"，本篇回答最后一个核心问题：**怎么让 GPU 真正跑得快？** 很多新手把 kernel 写完能算出正确答案就觉得结束了——但"能算对"和"跑得快"之间可能差 10 倍。本篇给你一套**可量化的分析方法**：占用率怎么算、矩阵乘怎么一步步从 naive 优化到 tiling、最后用 Nsight Compute 读出数字来验证每一步。

**阅读路线**：先建立两个性能抓手——**延迟隐藏/占用率**（§1）和**内存带宽 vs 算力**（§2），然后用矩阵乘（§3~§5）亲手跑一遍 `naive → shared → tiling` 的优化全过程，最后学会用 Nsight Compute 分析（§6）并形成一套调优流程（§7）。

---

## 1. 占用率（Occupancy）：把"等人"变成"换人"

### 1.1 回顾：GPU 靠"换人"隐藏延迟

02_cuda_basics.md §3.2 讲过，GPU 的线程等数据时（全局内存 ~200-400 周期），调度器**立刻换下一个 warp 去算**——这叫延迟隐藏。它的前提是：**SM 上得有足够多待命的 warp**。驻留的 warp 越多，越能填满这段等待时间。

**占用率（occupancy）** = 实际驻留在 SM 上的 warp 数 ÷ SM 能容纳的最大 warp 数。

```
例：一个 SM 最多能驻留 64 个 warp（GTX 1080 的极限是 64）
  当前 kernel 只驻留 32 个 → occupancy = 50%

occupancy 低 → 等待数据时没 warp 可换 → 计算单元空转 → 变慢
```

> 一句话：**占用率衡量的是"SM 上有多少活等着干"**。它是性能的"地基"——但注意，占用率高 ≠ 一定快（见 §2），它只是必要条件。

### 1.2 四个决定占用率的"墙"

SM 能驻留多少 warp，受四个资源限制，谁先到顶谁就是瓶颈（**取最小值**）：

| 限制来源 | GTX 1080（Pascal）参考 | 计算方法 |
|---|---|---|
| 每 SM 最大线程数 | 2048 线程 | `2048 ÷ block_size` 得最多 block 数 |
| 每 SM 最大 block 数 | 32 | 直接数 block |
| **寄存器** | 每 SM 65536 个 | `65536 ÷ (每线程寄存器数 × block_size)` |
| **共享内存** | 每 SM 96 KB（默认块内上限 48 KB） | `96KB ÷ (每 block 共享内存)` |

```
例子：block_size = 256，每线程用 32 寄存器，每 block 用 8 KB 共享内存

  线程墙：2048 / 256         = 8 个 block
  块数墙：32                  = 32 个 block
  寄存器墙：65536 / (32×256)  = 8 个 block  ← 卡这里
  共享内存墙：96 / 8          = 12 个 block

实际驻留 = min(8, 32, 8, 12) = 8 个 block = 2048 线程 = 100% occupancy
```

> ⚠️ 关键结论：**每线程用太多寄存器、或每 block 用太多共享内存，都会"堵死"驻留数**。这就是 03_memory.md §2/§3 提到的权衡：寄存器/共享内存给得越多，单线程越"舒服"，但能同时干活的人越少。

### 1.3 占用率与 block 大小的关系

- block 太小（如 64 线程）：block 数墙先到顶，驻留线程数上不去
- block 太大（如 1024）：调度不够灵活，且受 2048 线程墙限制只能放 2 个
- **经验值**：128/256 通常是甜点；warp 是 32 线程，block 取 32 的倍数避免尾部空 warp

> 实测：Nsight Compute 里直接读 `Achieved Occupancy` 这个指标，不必手算。手算是为了**理解瓶颈在哪面墙**。

---

## 2. 两个吞吐：算力 vs 带宽，谁是瓶颈？

### 2.1 GPU 是"内存墙"机器

GPU 有两张"吞吐能力"的牌：

```
算力吞吐：每秒能算多少次浮点运算（TFLOPS）
内存吞吐：每秒能从显存搬多少字节（GB/s）

一个 kernel 能跑多快 ≈ min( 受算力限制, 受带宽限制 )
```

| | 算力（GTX 1080 参考） | 带宽（GTX 1080 参考） |
|---|---|---|
| FP32 | ~8.9 TFLOPS | — |
| 显存带宽 | — | ~320 GB/s |

**把问题反过来算**：如果一个内核每读 1 字节要做 8 次浮点运算（算术强度 = 8 FLOP/byte），那么 320 GB/s 的带宽能支撑 8.9 TFLOPS 吗？

```
320 GB/s × 8 FLOP/byte = 2560 GFLOPs = 2.56 TFLOPS < 8.9 TFLOPS
→ 内存先把算力"饿死"了 → 这类内核是"带宽受限"（memory-bound）
```

### 2.2 算术强度（Arithmetic Intensity）：判断受限类型的钥匙

> 这个概念在 00_why_gpu.md §11 首次出现，这里是它的实战用法。

```
算术强度 = 一次内核总共的浮点运算数 ÷ 一次内核从全局内存搬的字节数
          （FLOP/byte）

- 算术强度高 → 数据搬进来能算很久 → 容易吃满算力 → compute-bound
- 算术强度低 → 数据刚搬进来就算完了 → 卡在带宽上 → memory-bound
```

| 典型内核 | 算术强度 | 瓶颈类型 |
|---|---|---|
| 向量加法 `c[i]=a[i]+b[i]` | ~0.33 FLOP/byte | 严重 memory-bound |
| softmax / LayerNorm | 低 | memory-bound |
| 矩阵乘法（tiling 后） | 随 tile 增大而升高 | 接近 compute-bound |
| 卷积（大 channel） | 高 | compute-bound |

**优化方向完全不同**：
- **memory-bound**（向量加、attention、量化后的 LLM 解码）：优化重点是**减少字节数**——合并访问、用 FP16/int8 减数据量、多算子融合
- **compute-bound**（GEMM 大矩阵）：优化重点是**提高计算效率**——tiling、寄存器复用、用 Tensor Core

> 💡 记住这张图，它是整个性能优化的"总开关"：**先判断自己是 memory-bound 还是 compute-bound，再决定往哪使劲**。

---

## 3. 实战一：矩阵乘法 naive → tiling 的完整动机

### 3.1 目标：手写 SGEMM（单精度矩阵乘）

`C[M][N] = A[M][K] × B[K][N]`，每个输出元素 `C[i][j]` 需要读 A 的一行 K 个、B 的一列 K 个：

```cuda
// naive 版：每个线程算一个输出元素
__global__ void sgemm_naive(const float* A, const float* B, float* C,
                            int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;   // 输出行 i
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // 输出列 j
    float sum = 0.f;
    for (int k = 0; k < K; k++)
        sum += A[row * K + k] * B[k * N + col];        // 每个 k 读两次全局内存
    C[row * N + col] = sum;
}
```

**它为什么慢？** 数一下访问量（假设 M=N=K=1024）：

```
每个线程算 1 个输出：
  读 A：K 次     读 B：K 次     共 2K 次全局访问
全部线程：M×N 个输出 × 2K 次 = 2×1024×1024×1024 ≈ 2.1G 次浮点运算
对应全局内存读取：每个 A[i][k] 被读 N 次、每个 B[k][j] 被读 M 次！

算术强度 ≈ 2 FLOP ÷ (2×4 字节) ≈ 0.25 FLOP/byte → 严重 memory-bound
```

**核心问题：数据复用为零**。A 的一行明明可以反复用，naive 版却每次都重新从显存读。

### 3.2 优化第一步：tiling（分块）→ 用共享内存做复用

思路：既然相邻线程算的输出会**共用同一块 A 的子矩阵、同一块 B 的子矩阵**，那就把这些子矩阵**先整体搬进共享内存**，再慢慢用。

```
把输出 C 分成 BLOCK×BLOCK 的小块（tile），每块交给一个 block：

  一个 block 负责 C 的一个 BLOCK×BLOCK 小块：
  1. 循环 K/BLOCK_TILE 次：
      把 A 的一个 BLOCK×BLOCK_TILE 小块拷进共享内存 As（合并访问）
      把 B 的一个 BLOCK_TILE×BLOCK 小块拷进共享内存 Bs（合并访问）
      __syncthreads()
      块内线程从 As/Bs 里算累加（读共享内存，快）
      __syncthreads()
  2. 写回 C 的小块

全局内存访问量从 O(N³) 降到 O(N³ / BLOCK)：
  例：BLOCK=16 → 全局读减少 16 倍；BLOCK=32 → 减少 32 倍
```

**这是 03_memory.md §3.2 的落地实现**，代码在 `code/04_optimization/`（`sgemm_naive.cu → sgemm_shared.cu`）。

### 3.3 优化第二步：register tiling —— 每个线程算多个输出

共享内存版每个线程只算 1 个输出，读写共享内存也要花周期。更进一步：**让每个线程算 2×2 或 4×4 个输出**，把 As/Bs 的元素读到**寄存器**后重复用 4 次，进一步降低共享内存带宽压力：

```
每线程算 2×2 个输出：
  读 As 一个元素 → 可参与 2 个输出的累加（寄存器复用）
  读 Bs 一个元素 → 可参与 2 个输出的累加
  共享内存访问量再减半，同时寄存器里的中间结果被反复用
```

### 3.4 三版对比（期望趋势，数值以 Nsight 实测为准）

| 版本 | 全局读次数（1024³） | 瓶颈类型 | 相对速度 |
|---|---|---|---|
| naive | ~2.1G | memory-bound（0.25 FLOP/byte） | 1× |
| shared tiling (32×32) | ~67M | 过渡 | ~5-10× |
| tiling + register (4×4) | ~67M | 逼近 compute-bound | ~15-20× |
| 调参后（Tensor Core/cuBLAS） | — | compute-bound | 30×+（cuBLAS 才是终点） |

> 提醒：**手写 SGEMM 永远跑不过 cuBLAS**——cuBLAS 还叠加了寄存器分块、双缓冲、按 K 展开、甚至 Tensor Core。手写的价值是**理解优化每一招为什么有效**，最终 PyTorch 调 cuBLAS 时才看得懂它的报告。

---

## 4. 实战二：tiling 里的两个隐藏坑

### 4.1 bank conflict：共享内存 tile 的行长

03_memory.md §3.4 说过，共享内存一次一个 warp 只能访问 32 个不同的 bank。tiling 后，一个 warp 的 32 个线程往往同时读 `As[threadIdx.y][k]`（同一行的不同列）——**行下标相同 → 列地址连续 → bank 恰好错开，通常没事**。

真正要注意的是**行长必须是 32 的倍数时会撞 bank**：

```cuda
// 行长 = 32（恰好是 bank 数）→ 线程读同一行时全部命中同一列 bank
__shared__ float As[32][32];
float v = As[ty][tx];          // tx 连续 → bank = tx，ok
float w = As[ty + 1][tx];      // 行距 32 → 和上一行同一 bank → 潜在冲突

// 修复：padding 一列，把行错开
__shared__ float As[32][32 + 1];   // 行长 33，非 32 倍数，天然错开 bank
```

### 4.2 `__syncthreads()` 放错位置 = 结果错误

共享内存版的核心纪律：

```
拷入 As/Bs → __syncthreads() → 计算读 As/Bs → __syncthreads() → 进入下一轮
```

漏掉任何一个同步点，都可能让线程 B 读到还没被线程 A 写好的数据。同步点放对，结果才对。

---

## 5. 更高层视角：把优化分成四板斧

| 招数 | 解决的问题 | 对应本篇 |
|---|---|---|
| **合并访问** | 带宽浪费 | 03_memory.md §5 |
| **共享内存复用（tiling）** | 全局内存读太多次 | §3.2 |
| **寄存器复用（register tiling）** | 共享内存带宽也用满 | §3.3 |
| **占用率调优** | 延迟藏不住 | §1 |

这四招层层递进：先让每次搬运"不浪费"（合并），再让搬运次数"变少"（tiling），最后让算力"吃满"（register tiling + occupancy）。**FlashAttention 也是这套逻辑**（05_llm_acceleration.md 会看到同样的 tiling 思路）。

---

## 6. 用 Nsight Compute 分析：让数字说话

### 6.1 工具链

| 工具 | 干什么 | 何时用 |
|---|---|---|
| `ncu`（Nsight Compute） | **逐 kernel 性能分析**：占用率、吞吐、stall 原因 | 优化每个 kernel |
| `nsys`（Nsight Systems） | 全局时间线：kernel 启动、memcpy、空隙 | 找"整体瓶颈在哪" |
| `nvidia-smi` | GPU 利用率、显存占用（粗粒度） | 快速扫一眼 |

> 远端 Win10 已装 **Nsight Compute 2022.1.1**（见 01_environment.md），与 CUDA 11.6 配套，可直接 `ncu`。

### 6.2 基本用法

```bash
# 用 nsys 看整体：哪个 kernel 最耗时、有没有 memcpy 空隙
nsys profile ./sgemm

# 用 ncu 分析单个 kernel 的关键指标
ncu --set full ./sgemm

# 只看最关心的几个指标（快很多）
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed, \
    gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed, \
    sm__warps_active.avg.pct_of_peak_sustained_active ./sgemm
```

### 6.3 五个必看的指标

| 指标 | 含义 | 判断 |
|---|---|---|
| `Achieved Occupancy` | 实际占用率 | 太低（<50%）→ 查寄存器/共享内存墙 |
| `SM [%]` | 计算单元利用率 | 接近 100% → 已 compute-bound |
| `Memory [%]` | 内存系统利用率 | 接近 100% 且 SM 低 → memory-bound |
| `DRAM Throughput` | 显存带宽用了多少 | 瓶颈判断的核心 |
| `Warp Stall 原因` | 每个 warp 在等什么 | 等内存 → memory-bound；等 barrier → 同步问题 |

```
看懂组合：
  SM 高 + Memory 低 → 计算密集，还能优化算法本身
  SM 低 + Memory 高 → 带宽受限，省字节才有效（§2 的判断落地）
  SM 低 + Memory 低 → 占用率太低 / 同步太密 / 启动开销大
```

### 6.4 一个分析例子：naive SGEMM

```
$ ncu ./sgemm_naive
Achieved Occupancy : 100%          ← 占用率没问题
SM Throughput      : ~10%          ← 计算单元几乎闲着
DRAM Throughput    : ~95%          ← 显存带宽被拉满

结论：memory-bound 确诊。占用率已经很高，继续堆线程没用，
     必须减少全局读次数 → 上 tiling。这就是 §3 方案的数据依据。
```

### 6.5 分析 sgemm_shared 后的预期变化

```
SM Throughput   : ~10% → ~40%
DRAM Throughput : ~95% → ~40%
→ 瓶颈从"带宽"转向"计算"，此时再上 register tiling / Tensor Core 才有效
```

> 这个"瓶颈转移"的过程就是优化：**每一步都先把当前最紧的瓶颈解开，再让下一个瓶颈暴露出来**。

---

## 7. 调优迭代流程（作业方法论）

```
1. 先跑通（结果正确）
2. ncu 看指标 → 判断 memory-bound 还是 compute-bound（§6.3）
3. 针对瓶颈选招（§5）
4. 改完再测：关注目标指标有没有变（而不是只看 wall time）
5. 重复，直到收益递减
```

**注意事项**：
- 对比必须**同一次程序里跑所有版本**，排除启动/驱动抖动
- 用 `cudaEvent` 计时（01_environment.md 有写法），不要用 CPU 墙钟直接量 kernel
- 每个优化单独 commit，方便回退和对照（本仓库学习闭环的 commit 规范）

---

## 8. 本篇与代码的对应

| 概念 | 对应代码/文件 |
|---|---|
| naive → shared → tiling | `code/04_optimization/`（sgemm_naive / sgemm_shared / sgemm_tiled） |
| bank conflict padding | `code/04_optimization/` 里 `As[TILE][TILE+1]` |
| occupancy 手算 | `nvcc -Xptxas -v` 看寄存器数 + §1.2 公式 |
| Nsight 实测 | 远端 Win10 `ncu --set full`（Nsight Compute 2022.1.1） |

> 下一篇 05_llm_acceleration.md：同一个"tiling + 内存感知"的思路，直接解释为什么 FlashAttention 能把大模型推理提速数倍——性能优化不白学。
