# 07 Triton：用 Python 写高性能 GPU 内核

> 对应里程碑 M5 之后（配合 `code/07_triton/`）。前面我们用了两条路写 GPU 程序：**CUDA C 手写内核**（docs/02~04，理解原理）和 **PyTorch 黑盒调用**（docs/06，日常使用）。本篇补上中间那块拼图：**用 Python 语法写接近手写性能的内核**——这就是 Triton。
>
> 阅读对象：已经看完 04_performance.md 的 tiling 和 05_llm_acceleration.md 的 FlashAttention，想在"自己写内核"这件事上少受 C/C++ 的苦。

**阅读路线**：先搞清 Triton 定位与"块级编程"思想（§1~§2），掌握最小语法（§3），再对照 `code/07_triton/` 把两个例子读透——**SGEMM**（§4，就是 04 篇 tiling 的 Python 版）和 **FlashAttention**（§5，就是 05 篇 §3 的落地）。读完你应能看懂/改写 `code/07_triton/` 里的两个 kernel。

---

## 1. Triton 是什么：三件事帮你记住

```
1. 语言层面：写内核用 Python，不用 C/C++ 编译链
2. 思想层面：不写"线程级"代码，写"块级(tile)级"代码
3. 落地层面：编译器自动帮你做 共享内存搬运 + 同步 + 选指令
```

对照 docs/06 §1.2 的"三条路径"：

| 路径 | 代表 | 你写什么 | 共享内存/同步 | 性能 |
|---|---|---|---|---|
| 手写内核 | CUDA C（docs/02~04） | 线程级代码 | 手写 `__shared__`/`__syncthreads` | 上限最高 |
| **写内核但用 Python** | **Triton（本篇）** | **块级代码** | **编译器自动安排** | **接近手写** |
| 完全托管 | PyTorch（docs/06） | 算子/模型 | 黑盒，管不到 | 库级 |

> 一句话：**Triton = 用 Python 写"分块"逻辑，把"怎么铺线程、怎么搬共享内存"全交给编译器**。所以它的上手难度远低于 CUDA C，性能又远高于 PyTorch 的自定义写法。

## 2. 为什么用 Triton，而不是直接写 CUDA C？

- **不用管线程**：CUDA C 里你要自己算 `blockIdx*blockDim+threadIdx`，还要手动 `__shared__` + `__syncthreads()`。Triton 里你只描述"这块 tile 干什么"，编译器把它展开成线程。
- **自动选指令**：`tl.dot` 会自动用 FFMA 甚至 Tensor Core（A100 上），不用像 04 篇手写 register tiling 那样抠细节。
- **自动调优**：同一份代码，改几个 `BLOCK_*` 编译期常量就能换 tile 大小，编译器会为当前硬件重新生成内核。
- **不牺牲多少性能**：官方 SGEMM/FlashAttention 教程都能跑到对应手写/库的 7~9 成。

> 代价：你只能表达"编译器的目标语言能表达的东西"——极致的指令级优化（比如 04 篇 §3.3 那种手工寄存器复用）仍然要靠 CUDA C。但 95% 的"自定义高性能算子"，Triton 都够用了。

---

## 3. 语法速览：最小例子

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
| `BLOCK: tl.constexpr` | **编译期常量**：换一个值就重新编译一次 | `#define`/模板参数 |

> 注意心智转变：**CUDA C 里你写"一个线程干什么"，Triton 里你写"一块数据干什么"**。`offs`、`x`、`y` 都是长度 BLOCK 的向量，编译器负责把它们拆给 32 线程的 warp 并做合并访问。

---

## 4. 实战一：SGEMM（对照 `code/07_triton/sgemm.py`）

这就是 04_performance.md §3 的 tiling，用 Triton 表达只有三步：

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

**和 04 篇手写版的对应关系**：

| 04 篇手写（sgemm_shared / sgemm_tiled） | Triton 版 |
|---|---|
| `blockIdx.x/blockIdx.y` 决定输出小块 | `pid_m / pid_n` |
| 手写 `__shared__ As[BLOCK][BLOCK+1]` + 拷贝循环 | `tl.load` 一条语句（编译器搬到共享内存） |
| 手写 `__syncthreads()` | 编译器在 load 与 dot 之间自动插入 |
| 手写 `sum += As[ty][kk]*Bs[kk][tx]` | `tl.dot(a, b, acc)` |
| padding 防 bank conflict | 编译器自动处理 |

> 额外一提 `GROUP_SIZE_M`：它让**同一行的小块挨着调度**，相邻块共享同一段 A 的 K 行，L2 命中率更高——这是 04 篇 §5"四板斧"之外的一招"调度优化"，写 `sgemm.py` 时可以直接对比有无。

预期：`code/07_triton/sgemm.py` 跑 4096³，正确性对 torch 通过，速度约为 cuBLAS 的 **70%~90%**。

---

## 5. 实战二：FlashAttention（对照 `code/07_triton/flash_attention.py`）

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

预期：`code/07_triton/flash_attention.py` 跑 `[4,8,2048,128]` 因果注意力，与 `F.scaled_dot_product_attention(is_causal=True)` 误差 < 1e-2，速度接近 `F.sdpa`。

---

## 6. 一张表总结

| 问题 | 答案 |
|---|---|
| Triton 是什么 | 用 Python 写 GPU 内核的框架，块级编程 |
| 和 CUDA C 的区别 | 不用管线程/共享内存/同步，编译器自动做 |
| 和 PyTorch 的区别 | 能自定义高性能算子，不是黑盒 |
| 在哪里跑 | 需要 NVIDIA GPU（AutoDL A100/4090D），`pip install triton` |
| 适合干什么 | 自定义算子、学习 SGEMM/Attention 内核思路、写 FA 类融合 kernel |
| 不适合干什么 | 极致的指令级微优化（仍要 CUDA C） |

---

## 7. 本篇与代码的对应

| 概念 | 对应代码 |
|---|---|
| 最小语法（§3） | `code/07_triton/sgemm.py` 里的 `@triton.jit`/`tl.arange`/`tl.load` |
| SGEMM tiling（§4） | `code/07_triton/sgemm.py`（对照 `code/04_optimization/` 三版手写） |
| FlashAttention（§5） | `code/07_triton/flash_attention.py` |
| 在线 softmax / 因果掩码 | `code/07_triton/flash_attention.py` 主循环 |

> 下一篇可以回到 `docs/05_llm_acceleration.md` 的 M6 计划：在 A100/4090D 上把 `code/07_triton/` 跑起来，和官方 FlashAttention、llama.cpp/vLLM 做吞吐对比——学到这里，"从原理到应用"的闭环就完整了。
