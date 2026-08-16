# 06 用 Python 写高性能内核：Triton 与 TileLang

> 对应里程碑 M5 之后（配合 `code/06_dsl_kernels/`）。前面我们用了两条路写 GPU 程序：**CUDA C 手写内核**（docs/02~03，理解原理）和 **PyTorch 黑盒调用**（docs/04，日常使用）。本篇补上中间那块拼图：**用 Python 语法写接近手写性能的内核**。主角有两个——**Triton**（OpenAI 开源、PyTorch `torch.compile` 的代码生成后端、生态主流）和 **TileLang**（北大团队、构建在 TVM 上、国产新秀）。
>
> 阅读对象：已经看完 03_cuda_advanced.md 的 tiling 和 05_llm_acceleration.md 的 FlashAttention，想在"自己写内核"这件事上少受 C/C++ 的苦。

**阅读路线**：先建立"DSL 内核编译器"的整体观——它站在哪一层、对标 Java 生态的什么（§1）；然后掌握 **Triton**：定位与现状（§2）、最小语法（§3）、两个实战 SGEMM / FlashAttention（§4）；再看 Triton 的局限，引出 **TileLang**（§5~§7）；最后一张表选型（§8）。

---

## 1. DSL 内核编译器：站在哪一层

### 1.1 三条路径回顾：缺的正是"中间一块"

回顾 docs/04 §1.2 的"用 Python 驱动 GPU 的三条路径"：

| 路径 | 代表 | 你写什么 | 共享内存/同步 | 性能 |
|---|---|---|---|---|
| 手写内核 | CUDA C（docs/02~03） | 线程级代码 | 手写 `__shared__`/`__syncthreads` | 上限最高 |
| **写内核但用 Python** | **Triton / TileLang（本篇）** | **块（tile）级代码** | **编译器自动安排** | **接近手写** |
| 完全托管 | PyTorch（docs/04） | 算子/模型 | 黑盒，管不到 | 库级 |

**DSL（Domain-Specific Language，领域特定语言）** 就站在中间：保留"自定义高性能算子"的能力，把"怎么铺线程、怎么搬共享内存、怎么选指令"全部交给编译器。

> 一句话：**Triton/TileLang = 用 Python 写"分块"逻辑，把"怎么铺线程、怎么搬共享内存"全交给编译器**。上手难度远低于 CUDA C，性能又远高于 PyTorch 的自定义写法。

### 1.2 它在 CUDA 平台栈里的位置：Java 平台类比

把 CUDA 平台和 Java 平台对齐（编译链详见 docs/02 §7），DSL 编译器恰好补上"最后一格"：

| 层级 | CUDA 平台 | Java 平台 |
|---|---|---|
| 最终机器码 | SASS | x86 / ARM 机器码 |
| 平台无关中间表示 | **PTX** | **Java 字节码（.class）** |
| 主流语言 | CUDA C++ | Java |
| 源码→IR 编译器 | nvcc | javac |
| IR→机器码（运行时） | GPU 驱动 JIT | JVM HotSpot JIT（C1/C2） |
| 预优化高性能库 | cuBLAS / cuDNN / NCCL | JDK 标准库 / Native 库 |
| AOT 优化器 | TensorRT / CUTLASS | GraalVM Native Image |
| **DSL + 自带优化编译器** | **Triton / TileLang** | **≈ Scala/Kotlin + GraalVM** |

> **Triton/TileLang ≈ "Scala/Kotlin 语言 + GraalVM 编译器"**：用更简洁的类 Python 语法描述计算意图，由一个更聪明的编译器自动完成优化（内联 / 布局推理 / 流水线）和多后端代码生成（PTX / MUSA / Ascend C）。它不是"又一个语言"，而是**语言 + 编译器 + 自动调优器**的三位一体——这正是它需要靠 GraalVM 全家桶才能类比的原因。
>
> 对应到 02 篇 §7 的编译链：你写的 DSL 代码 → 编译器生成 PTX（≈ 字节码）→ 驱动 JIT 成 SASS（≈ 机器码）。差别只在"谁、在什么时候做编译"。

---

## 2. Triton：用 Python 写 GPU 内核

### 2.1 三件事帮你记住

```
1. 语言层面：写内核用 Python，不用 C/C++ 编译链
2. 思想层面：不写"线程级"代码，写"块级（tile 级）"代码
3. 落地层面：编译器自动帮你做 共享内存搬运 + 同步 + 选指令
```

### 2.2 现状（2026）：已是生态标配

- **归属**：由 OpenAI 开源，现托管于 **triton-lang 组织**，MIT 协议，Meta / AMD / NVIDIA / OpenAI / Intel / Google 等共同贡献
- **版本**：3.7.x（2026 年中），编译器后端已整体迁移到 **MLIR / LLVM**
- **硬件后端**：NVIDIA（CC 8.0+，Ampere 起）、AMD（ROCm 6.2+）；CPU 后端开发中
- **最大用户**：PyTorch `torch.compile` 的 **Inductor** 后端现场生成 Triton 代码（docs/04 §5）；FlashAttention、SGLang、vLLM 等也大量用它写自定义算子
- **调试**：`TRITON_INTERPRET=1` 可进 Python 解释器、打断点，无 GPU 也能跑通逻辑

### 2.3 为什么用 Triton，而不是直接写 CUDA C？

- **不用管线程**：CUDA C 里你要自己算 `blockIdx*blockDim+threadIdx`，还要手动 `__shared__` + `__syncthreads()`。Triton 里你只描述"这块 tile 干什么"，编译器把它展开成线程。
- **自动选指令**：`tl.dot` 会自动用 FFMA 甚至 Tensor Core（A100 上），不用像 03 篇手写 register tiling 那样抠细节。
- **自动调优**：同一份代码，改几个 `BLOCK_*` 编译期常量就能换 tile 大小，编译器为当前硬件重新生成内核。
- **不牺牲多少性能**：官方 SGEMM / FlashAttention 教程都能跑到对应手写/库的 7~9 成。

> 代价：你只能表达"编译器的目标语言能表达的东西"——极致的指令级优化（比如 03 篇 §10.3 那种手工寄存器复用）仍然要靠 CUDA C。但 95% 的"自定义高性能算子"，Triton 都够用了。

---

## 3. Triton 语法速览：最小例子

Triton 内核用 `@triton.jit` 装饰，运行在 GPU 上。看一个向量加（对照 `code/00_hello/vector_add.cu`）：

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)                      # 我在第几个"块"（类似 blockIdx.x）
    offs = pid * BLOCK + tl.arange(0, BLOCK)    # 本块负责的一整段下标
    mask = offs < n                             # 防越界（数组长度不是 BLOCK 整数倍时）
    x = tl.load(x_ptr + offs, mask=mask)        # 一次加载整块
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)  # 一次存回整块

n = 1_000_000
add_kernel[(n + 255) // 256](x, y, out, n, BLOCK=256)   # 网格 = 块数，BLOCK 是编译期常量
```

五个关键点：

| 语法 | 作用 | 类比 CUDA C |
|---|---|---|
| `@triton.jit` | 把函数 JIT 编译成 GPU kernel | `__global__` |
| `tl.program_id(0)` | 我是第几个块 | `blockIdx.x` |
| `tl.arange(0, BLOCK)` | 生成 0..BLOCK-1 的索引向量（**必须是 2 的幂**） | 手动循环 |
| `tl.load / tl.store` | 带 mask 的批量读写 | 指针 + 边界判断 |
| `BLOCK: tl.constexpr` | **编译期常量**：换一个值就重新编译一次 | `#define` / 模板参数 |

> 注意心智转变：**CUDA C 里你写"一个线程干什么"，Triton 里你写"一块数据干什么"**。`offs`、`x`、`y` 都是长度 BLOCK 的向量，编译器负责把它们拆给 32 线程的 warp 并做合并访问。

---

## 4. Triton 实战

### 4.1 SGEMM（对照 `code/06_dsl_kernels/sgemm.py`）

这就是 03_cuda_advanced.md §10 的 tiling，用 Triton 表达只有三步：

```python
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn,
                  stride_cm, stride_cn,
                  BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                  BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    # 1) 算出我这个块负责 C 的哪一小块
    pid = tl.program_id(0)
    ...
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)   # 行索引
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)   # 列索引
    offs_k  = tl.arange(0, BLOCK_SIZE_K)                          # K 切片

    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # 2) 沿 K 循环，每次搬一小块进来算（tiling！）
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        acc = tl.dot(a, b, acc)                 # 3) 累加：自动选矩阵乘指令
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # 写回
    tl.store(c_ptrs, acc, mask=c_mask)
```

**和 03 篇手写版的对应关系**：

| 03 篇手写（sgemm_shared / sgemm_tiled） | Triton 版 |
|---|---|
| `blockIdx.x/blockIdx.y` 决定输出小块 | `pid_m / pid_n` |
| 手写 `__shared__ As[BLOCK][BLOCK+1]` + 拷贝循环 | `tl.load` 一条语句（编译器搬到共享内存） |
| 手写 `__syncthreads()` | 编译器在 load 与 dot 之间自动插入 |
| 手写 `sum += As[ty][kk]*Bs[kk][tx]` | `tl.dot(a, b, acc)` |
| padding 防 bank conflict | 编译器自动处理 |

> 额外一提 `GROUP_SIZE_M`：它让**同一行的小块挨着调度**，相邻块共享同一段 A 的 K 行，L2 命中率更高——这是 03 篇 §12"四板斧"之外的一招"调度优化"，写 `sgemm.py` 时可以直接对比有无。

预期：`code/06_dsl_kernels/sgemm.py` 跑 4096³，正确性对 torch 通过，速度约为 cuBLAS 的 **70%~90%**。

### 4.2 FlashAttention（对照 `code/06_dsl_kernels/flash_attention.py`）

这就是 05_llm_acceleration.md §3 的 FlashAttention，tiling + 在线 softmax 各就位：

```python
acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
m_i = tl.zeros((BLOCK_M,), dtype=tl.float32) - float("inf")   # 每行的 running max
l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)                  # 每行的 running sum(exp)

# Q 的一个块，沿 K/V 一块块扫（因果：只看前面的块）
for start_n in range(0, (start_m + 1) * BLOCK_M, BLOCK_N):
    qk = tl.dot(q, k) * sm_scale
    qk = tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :],
                  qk, float("-inf"))                          # 因果掩码
    m_ij = tl.maximum(m_i, tl.max(qk, 1))                     # 在线 softmax
    p = tl.exp(qk - m_ij[:, None])
    l_ij = tl.sum(p, 1)
    alpha = tl.exp(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]
    acc = tl.dot(p.to(tl.float16), v, acc)
    m_i = m_ij
acc = acc / l_i[:, None]                                      # 最后归一
```

**为什么这就是 05 篇 §3.2 说的"IO-aware"**：`p`（softmax 权重）这个 `[BLOCK_M, BLOCK_N]` 的中间矩阵**从头到尾只活在寄存器/共享内存里**，从不写回全局内存——整个 kernel 对显存的读写只有 Q/K/V 各一遍。而 05 篇 §3.1 手写版要先存整个 `[T, T]` 的 S。

**两个容易写错的地方**：

1. **`tl.exp` 用自然对数域**：`m_i/l_i` 统计量、`alpha`、最终 `acc / l_i` 全部用自然指数/对数，才能和 PyTorch 的 `softmax` 严格一致。用 `tl.math.exp2`（基 2）等价于给 logits 乘了 `ln2`，结果会系统性偏差。
2. **head 维（BLOCK_D）必须是 2 的幂**（代码里用 128），`tl.arange` 不接受任意长度。

预期：`code/06_dsl_kernels/flash_attention.py` 跑 `[4,8,2048,128]` 因果注意力，与 `F.scaled_dot_product_attention(is_causal=True)` 误差 < 1e-2，速度接近 `F.sdpa`。

---

## 5. Triton 的局限：为什么还需要 TileLang

Triton 很好，但手写复杂算子时仍有三重困境（这也正是 TileLang 诞生的背景）：

1. **CUDA 开发门槛极高**：CUDA C++ / PTX 需要深入理解线程模型、共享内存、Bank Conflict、Tensor Core 等底层细节，一个高性能 GEMM 内核动辄数百甚至上千行（03 篇你亲自写过）。
2. **Triton 这类高层 DSL 仍有局限**：`tl.arange` 长度必须是 2 的幂；线程绑定、数据布局（layout）基本由编译器黑箱决定，想显式控制很费劲；对 **MLA（Multi-head Latent Attention）、FP8 混合精度**等复杂算子，编译器难以自动生成最优代码。
3. **硬件碎片化**：NVIDIA、AMD、华为昇腾、摩尔线程的指令集差异巨大，每适配一种硬件几乎要重写一遍算子——Triton 的官方后端目前只有 NVIDIA / AMD。

**一句话**：Triton 帮你把 90% 的活干完了，剩下 10%（显式布局控制、多后端 NPU、超复杂融合）就需要一个**更可控、更"可编程调度"**的 DSL——**TileLang**。

---

## 6. TileLang：可编程调度的块级 DSL

### 6.1 定位与背景

**TileLang**（Tile Language）是专为 AI 算子设计的领域特定语言，核心目标是：**既保留 Triton 的易用性，又给你"调度（scheduling）"的细粒度控制**。

| 基本信息 | 内容 |
|:---|:---|
| **开发团队** | 北京大学计算机学院 杨智副教授团队（论文一作：王磊） |
| **技术基座** | 构建于 **Apache TVM** 之上，用 TVM TensorIR 作为中间表示 |
| **语法风格** | 类 Python 声明式语法 |
| **开源** | 完全开源（`github.com/tile-ai/tilelang`），ICLR 2026 论文 |
| **典型用户** | DeepSeek（V3.2-Exp 研发）、SGLang、摩尔线程、华为昇腾 |

> ⚠️ 曾有人误传"TileLang 是 DeepSeek 自主研发"——不对。**DeepSeek 是重要的使用者**，开发者是北大杨智团队。

### 6.2 核心设计理念：把"数据流"和"调度"解耦

Triton 里"怎么并行"由编译器全包；TileLang 则把它拆开：**你描述数据流（dataflow），编译器给默认调度；不满意时可以手动干预调度（scheduling）**。

- **Tile 级抽象**：核心编程对象是 **Tile（张量分块）**，贯穿整个内存层级：`Global → Shared → Register（fragment）→ Compute → Accumulator → Global`
- **声明式语法 + 自动优化**：你写"算什么东西"，编译器自动做布局推理、并行推理、循环转换、软件流水线、线程绑定
- **三种使用模式**，按能力分层：

| 模式 | 面向对象 | 特点 |
|:---|:---|:---|
| **Beginner** | 入门用户 | 极致简洁，无需关注硬件细节 |
| **Developer** | 中级开发者 | 可控制 Shared Memory、Register Fragment 等资源 |
| **Expert** | 高性能专家 | 显式控制 Pipeline、Warp 级原语，追求极致性能 |

### 6.3 核心原语 + 一个 GEMM 例子

原语是 DSL 的一部分，描述的是 GPU kernel 里的硬件结构：

```python
T.alloc_shared([block_M, block_K], dtype)   # 分配共享内存 Tile
T.alloc_fragment([block_M, block_N], dtype) # 分配寄存器 Fragment
T.copy(src, dst)                            # 数据搬运（Global→Shared→Register…）
T.gemm(A, B, C)                             # 矩阵乘法（映射到 Tensor Core）
```

和 Triton 的 SGEMM（§4.1）对照着看——结构几乎一样，但你能显式看到"数据住在哪"：

```python
import tilelang
import tilelang.language as T

@tilelang.jit
def matmul(M, N, K, block_M=128, block_N=128, block_K=64, dtype='float16'):
    @T.prim_func
    def main(A: T.Tensor((M, K), dtype),
             B: T.Tensor((K, N), dtype),
             C: T.Tensor((M, N), 'float32')):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M),
                      threads=128) as (bx, by):
            A_s = T.alloc_shared((block_M, block_K), dtype)   # 显式声明在共享内存
            B_s = T.alloc_shared((block_K, block_N), dtype)
            C_f = T.alloc_fragment((block_M, block_N), 'float32')  # 累加在寄存器
            T.clear(C_f)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):  # 软件流水线
                T.copy(A[by * block_M, ko * block_K], A_s)   # 全局 → 共享
                T.copy(B[ko * block_K, bx * block_N], B_s)
                T.gemm(A_s, B_s, C_f)                        # 映射到 Tensor Core
            T.copy(C_f, C[by * block_M, bx * block_N])
    return main
```

> 核心逻辑约 30 行即可实现带 3 级流水线的高性能 GEMM。**和 Triton 的关键差异**：内存层级（shared / fragment）是你显式声明、且可以手动调度的——Triton 把这些细节藏起来了。

---

## 7. TileLang 的现状与生态（2026）

### 7.1 技术现状

- **版本**：v0.1.13（2026-08），多后端语言方言，已支持 **CUDA（SM70~SM120）、AMD ROCm/HIP、Apple Metal、LLVM CPU**（WebGPU、NVIDIA CuTe DSL 实验性），Windows 有预编译 wheel
- **编译器**：构建在 TVM 上，自动用 TMA/WGMMA（Hopper）、异步拷贝（Ampere）、布局推理 + 自动调优（autotuning）
- **开发者体验**：官方开源了 TileLang LSP（语法高亮、布局 inlay hints、诊断）

### 7.2 性能（注意别神化）

ICLR 2026 论文（《TileLang: Bridge Programmability and Performance in Modern Neural Kernels》）报告：**特定融合 kernel 上相对 Triton 可达约 5×（H100）/ 6×（AMD）加速**，且能 <80 行写出各种 FlashAttention 变体（比手写少 ~90% 代码）。

> ⚠️ 这是"特定算子、特定场景"的最优结果，不是通用提速。真实收益来自：显式 layout/流水线控制、自动选 Tensor Core/TMA 指令、动态形状优化——和 Triton 是同一套"块级编程"思想，只是调度空间更细。**"无脑比 Triton 快 5 倍"不是普遍结论。**

### 7.3 产业应用

| 用户 | 干什么 |
|:---|:---|
| **DeepSeek** | V3.2-Exp 研发明确提到"采用高级语言 TileLang 进行快速原型开发"；梁文锋表示 TileLang + AI 写代码有望重写 CUDA 算子生态 |
| **SGLang** | 用 TileLang 写注意力等高性能算子（FlashMLA 系） |
| **摩尔线程** | 开源 TileLang-MUSA，国产 GPU 上自动调 Tensor Core |
| **华为昇腾** | TileLang AscendNPU IR 接入 CANN 生态，JIT 生成 Ascend C |

> 一句话：TileLang 的价值是**硬件无关的中间层**——一份 Python 代码，多后端输出（CUDA / HIP / MUSA / Ascend C），恰好对国产 AI 芯片"百花齐放但软件生态薄弱"的现状很重要。

---

## 8. 选型：Triton vs TileLang vs 手写 CUDA

| 维度 | CUDA C++ / PTX | Triton | **TileLang** |
|:---|:---|:---|:---|
| **抽象层级** | 线程级，最底层 | 块级（隐藏布局） | **块级 + 可显式调度** |
| **语法** | C++ | 类 Python | 类 Python（声明式） |
| **Shared Memory 控制** | 手动 | 隐式 | 可显式可隐式（按模式） |
| **线程绑定 / 布局控制** | 手动 | 编译器黑箱 | **可手动干预** |
| **硬件后端** | 仅 NVIDIA | NVIDIA / AMD（CPU 开发中） | **NVIDIA / AMD / Metal / CPU / 摩尔线程 / 昇腾** |
| **编译器基座** | nvcc | LLVM / MLIR | TVM TensorIR |
| **生态 / 成熟度** | 最成熟 | **最主流**（torch.compile 后端） | 新秀，增长快 |
| **学习门槛** | 极高 | 中等 | 低（Beginner 模式） |

**怎么选（结合本仓库）**：

- **想学内核思路 / 跑通第一个融合算子** → **Triton**（`code/06_dsl_kernels/`，教程多、跟 torch.compile 同源）
- **要写复杂算子且受困于 Triton 的"黑箱布局"** → **TileLang**（更细的调度控制、多后端）
- **要极致指令级性能 / 目标硬件是单一 NVIDIA 卡** → CUDA C（03 篇那条路）
- **只是用 PyTorch** → 你大概率不需要本篇（04 篇已够）

### 本篇与代码的对应

| 概念 | 对应代码 |
|---|---|
| 最小语法（§3） | `code/06_dsl_kernels/sgemm.py` 里的 `@triton.jit`/`tl.arange`/`tl.load` |
| SGEMM tiling（§4.1） | `code/06_dsl_kernels/sgemm.py`（对照 `code/04_optimization/` 三版手写） |
| FlashAttention（§4.2） | `code/06_dsl_kernels/flash_attention.py` |
| 在线 softmax / 因果掩码 | `code/06_dsl_kernels/flash_attention.py` 主循环 |

> 下一篇可以回到 `docs/05_llm_acceleration.md` 的 M6 计划：在 A100/4090D 上把 `code/06_dsl_kernels/` 跑起来，和官方 FlashAttention、llama.cpp/vLLM 做吞吐对比——学到这里，"从原理到应用"的闭环就完整了。