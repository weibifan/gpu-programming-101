# 05 LLM 加速专题：Attention、量化、KV Cache 与推理引擎

> 对应里程碑 M5~M6。前面的笔记都在讲"一块 GPU 怎么算得快"；本篇把镜头拉远：**跑一个大语言模型（LLM），瓶颈在哪？** 你会发现：LLM 推理慢，恰恰不是"算得慢"，而是"**数据搬得慢**"。学过 04_performance.md 的 memory-bound / compute-bound 之后，你会看明白 FlashAttention、量化、KV cache、vLLM 这些"花活"，本质都在做同一件事——**省显存、省带宽、把内存墙捅破**。

**阅读路线**：先搞清 LLM 推理为什么"卡在内存上"（§1），再看三个核心优化：KV cache（§2）、FlashAttention（§3）、量化（§4），最后落到推理引擎 llama.cpp / vLLM（§5~§6）。读完你应能解释：为什么解码阶段快不起来？FlashAttention 到底省了什么？int4 量化为什么能提 4 倍速？vLLM 凭什么能服务很多用户？

---

## 1. 大模型推理为什么慢：从"算力"到"内存墙"

### 1.1 自回归生成：一个 token 一个 token 地挤出来

LLM（如 GPT 系列）是**自回归（autoregressive）**模型：生成时每步只产出一个 token，再把新 token 接回输入继续预测下一个。

```
输入: [今天天气]
  Step 1: 预测下一个 → "很"       输入变为 [今天天气, 很]
  Step 2: 预测下一个 → "好"       输入变为 [今天天气, 很, 好]
  Step 3: 预测下一个 → "，"
  ...
生成 100 个 token 就要跑 100 次前向传播
```

**关键事实**：每次前向传播，**整个模型的权重（几千亿参数）都要被读一遍**——哪怕只是为了预测一个 token。

### 1.2 算术强度与"内存墙"的第一次相遇

拿一个 7B 模型（70 亿参数 ≈ 14 GB 的 FP16）举例，解码一个 token：

```
每步要搬的字节：14 GB（全部权重从显存读一遍）
每步能做的运算：约 2×70 亿 次浮点 ≈ 14 GFLOP

算术强度 = 14 GFLOP / 14 GB ≈ 1 FLOP/byte  ← 非常低！
```

对比 04_performance.md 的结论：算力 8.9 TFLOPS、带宽 320 GB/s 的 GTX 1080，**喂满算力需要 8.9T/320G ≈ 27 FLOP/byte**。而 7B 解码只有 1 FLOP/byte——

```
搬 14 GB 数据所需时间 = 14 GB / 320 GB/s ≈ 44 ms
用这些数据算 14 GFLOP 所需时间 = 14G / 8.9T ≈ 1.6 ms

→ 27 倍的时间都花在"搬权重"，GPU 算力闲置
```

> **这就是 LLM 推理慢的根本原因：解码阶段是 100% memory-bound。** 所以所有加速手段都围绕一个字：**省**——省字节数（量化）、省重复搬（KV cache）、省中间量（FlashAttention）、省浪费（batch 起来算）。

### 1.3 两阶段：prefill 和 decode

| 阶段 | 干什么 | 瓶颈 | 特点 |
|---|---|---|---|
| **prefill**（预填充） | 一次性处理整个输入 prompt，并行算出每个位置的注意力 | compute-bound | 短促但计算密集，能喂满算力 |
| **decode**（解码） | 逐个生成 token | **memory-bound** | 长而慢，几乎全靠带宽 |

> 优化的对象因此不同：prefill 靠 FlashAttention 提速（§3），decode 靠 KV cache（§2）+ 量化（§4）+ 连续批处理（§6）提速。

---

## 2. KV Cache：把算过的注意力中间量存起来

### 2.1 为什么需要它

标准 attention（03 篇没有直接讲，这里展开）：

```
Q = X @ Wq      K = X @ Wk      V = X @ Wv      （X 是每层的输入，W 是权重）
Attention = softmax(Q @ K^T / √d) @ V
```

每一层都会为**每个历史 token** 算出一组 K 和 V。自回归解码时，第 t 步只需要**新 token 自己的 Q**，但 attention 还要跟**前面所有 token 的 K、V** 做计算。

**如果每步都重新算前面的 K、V**（因为它们依赖历史输入），那计算量随序列长度平方增长——序列长 1000，第 1000 步要白算 1000 遍前面的东西。

**KV cache 的做法**：第一次见到某 token 时，算出它的 K、V，**存在显存里**，之后每一步直接拿来用，不再重算。

```
无 KV cache：每步重算全部历史 K、V     → O(T²) 计算
有 KV cache：每步只算新 token 的 K、V → O(T) 计算，历史直接读缓存
```

### 2.2 KV cache 有多大？（显存开销）

```
KV cache 大小 = 2（K 和 V）× 层数 × 每层 head 数 × head 维度 × 序列长度 × 每元素字节

例：7B 模型（32 层，32 head，head 维度 128，即每层 32×128=4096 维），FP16：
  每个 token = 2 × 32 × 4096 × 2 字节 = 512 KB
  序列长度 2048 → 约 1 GB；长度 8192 → 约 4 GB

一个 8 GB 显存（GTX 1080）≈ 只够 7B 权重（14GB）放不下 + KV cache 更放不下
```

> ⚠️ **KV cache 是"以显存换计算"**：它省了重算的算力，但吃显存。这也解释了为什么：
> - 长上下文很贵（KV cache 线性膨胀）
> - 量化能同时压缩权重**和** KV cache
> - GQA/MQA（§6.4）专门用来减小 KV cache

### 2.3 prefill vs decode 的 KV 流动

```
prefill 阶段：整个 prompt 的 KV 全部算好 → 写入 KV cache
decode 阶段：每步只算新 token 的 K、V → 追加到 cache → 用全 cache 做 attention
```

`code/05_llm/` 里会写一个"无 cache vs 有 cache"的对比实验，直接测 decode 延迟的差距。

---

## 3. FlashAttention：把"省"做到算子和显存层面

### 3.1 标准 Attention 的问题：中间量 S 太大

标准实现（PyTorch 里 `x @ y` 那种）：

```
S = Q @ K^T          # shape [T, T]，每个位置都要存
P = softmax(S)       # 又一个 [T, T]
O = P @ V            # 结果 [T, d]

序列长 T=4096、head 维 128、batch 32：
  S 矩阵 = 32 × 32 × 4096 × 4096 × 4 字节 ≈ 64 GB —— 显存根本放不下！
```

**两个致命点**：
1. `S` 和 `P` 都要**写回全局内存再读回来**，带宽浪费巨大
2. 显存里**放不下**长序列的 S 矩阵，只能退化成小 batch

### 3.2 FlashAttention 的思路：04 篇的 tiling 换了个马甲

> 04_performance.md §3.2 我们刚学过：**把数据分块（tile）搬进共享内存，不写回全局内存，就地算完**。FlashAttention 就是把这个套路用在 attention 上——所以它叫 **IO-aware attention（感知输入输出的注意力）**。

```
1. 把 Q、K、V 分块，一块块搬进共享内存/寄存器
2. 对每块算局部 S 块、softmax 块、V 乘
3. 用一个"在线 softmax"技巧（online softmax），
   边算边累计正确的归一化，不需要先把整个 S 存下来
4. 全程 S、P 从不出共享内存 → 全局内存读写降到接近"只读 Q/K/V 各一遍"

效果：全局内存读写从 O(T²) 降到 O(T) → prefill 速度提升 2~4 倍
```

**关键点**：FlashAttention 不是改变数学，是**改变数据流动的位置**——中间量不再经过慢速的全局内存。这正是 04 篇"内存感知优化"的极致版。

### 3.3 三代演进

| 版本 | 改进 | 效果 |
|---|---|---|
| FA1（2022） | tiling + online softmax | 显存省 O(T²)，prefill 快 2-3× |
| FA2（2023） | 并行策略、寄存器优化、减少非矩阵运算 | 再快 ~2×，decode 也受益 |
| FA3（2024+） | 深度绑定 Hopper/Blackwell，用 Tensor Core | 大模型吞吐再升 |

### 3.4 三种写法的对照（实践在 code/05_llm/）

```python
# 写法 1：手写 naive（教学用，别在生产跑）
import torch, math
S = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d)
P = torch.softmax(S, dim=-1)
O = torch.matmul(P, V)

# 写法 2：PyTorch 官方接口（自动选最优实现，日常就用它）
import torch.nn.functional as F
O = F.scaled_dot_product_attention(Q, K, V)
#   → GPU 有 FlashAttention 支持时自动走 FA，否则走 memory-efficient 版

# 写法 3：显式调 FlashAttention（fa2/fa3 库，M6 在 A100/4090D 上实测）
```

> 2022 年手写 attention（旧仓库 `ex31` 的写法）→ 2026 年 `F.scaled_dot_product_attention` 一行搞定且自动加速，是 PyTorch 生态最大的变化之一（详见 06_pytorch_gpu.md）。

---

## 4. 量化：压缩字节，直接打带宽

### 4.1 为什么量化对 LLM 特别有效

04 篇 §2 说 memory-bound 的内核优化重点就是**减字节数**。LLM 解码正好是 memory-bound——把每个权重从 FP16（2 字节）压到 INT4（0.5 字节），**搬运量直接省 4 倍**，解码速度理论上接近 4 倍。同时显存占用也省 4 倍，模型"装得下"。

### 4.2 精度与代价

| 精度 | 每权重字节 | 相对 FP16 | 特点 |
|---|---|---|---|
| FP16 | 2 | 1× | 训练/高精度基线 |
| BF16 | 2 | 1× | 范围与 FP32 同，深度学习首选 |
| INT8 | 1 | 2× 省 | 掉点很小，推理常用 |
| INT4 | 0.5 | **4× 省** | 掉点可接受，llama.cpp 默认路线 |
| INT4 + 更小 | <0.5 | >4× | 花活多，掉点渐大 |

### 4.3 量化方法：先训后量化（PTQ）vs 微调（QAT）

```
PTQ（训练后量化，主流）：
  权重已经训好，直接按统计信息压成低精度
    - GPTQ：逐层量化，误差反馈补偿（适合 4bit）
    - AWQ：按"对激活影响大的通道"重点保护（适合 4bit）
    - llama.cpp 的 GGUF q4_0/q4_K 系列：离线量化，人人可跑

QAT（量化感知训练）：训练时就把量化误差算进去，更稳但成本高，少用

QLoRA（微调时的量化）：4bit 加载模型 + LoRA 微调，8G 显存能训 7B（M6 实践）
```

### 4.4 实测心里预期（M5~M6 会用 4090D/A100 跑）

```
7B FP16：14 GB 权重，8G 显存放不下，3090/4090 勉强
7B INT4：~4 GB 权重，GTX 1080 的 8G 也能装！
→ 这就是 llama.cpp "8G 显存跑 7B" 的秘密
```

> 量化不是白拿：INT4 相比 FP16 通常有 1~3% 的困惑度上升（perplexity），多数场景无所谓，但要注意**敏感层**（如 attention 的 Q/K 投影）可以保更高精度——AWQ 就是这么干的。

---

## 5. 推理引擎之一：llama.cpp

### 5.1 为什么选它

- **单文件、零依赖**，CPU/GPU 都能跑，量化支持最成熟
- GGUF 格式一站式（量化权重 + 元数据 + KV cache 配置）
- 8G 显存也能本地跑 7B int4（M6 目标）

### 5.2 基本用法（code/05_llm/ 会给出具体脚本）

```bash
# 1. 下载 int4 GGUF 权重（如 Qwen2-7B-Instruct-Q4_K_M.gguf）
# 2. 编译（带 CUDA 后端）
cmake -B build -DGGML_CUDA=ON
cmake --build build -j

# 3. 跑起来，指定 GPU 层数（-ngl 层数放 GPU，其余 CPU）
./build/bin/llama-cli -m qwen2-7b-instruct-q4_k_m.gguf \
    -ngl 32 -p "讲一个 GPU 编程的故事" -n 200
```

**关键参数**：`-ngl`（offload 到 GPU 的层数）是显存和速度的权衡；`-t` 是 CPU 线程。

### 5.3 测速指标：tokens/s

llama.cpp 会打印 **decode speed**（如 `32 tokens/s`）。拿它做量化、层数、硬件的横向对比。

---

## 6. 推理引擎之二：vLLM 与"连续批处理"

### 6.1 为什么单个 decode 这么浪费

decode 阶段算力闲置（§1.2），但一个用户独占整块 GPU 显然浪费。直觉：**同时塞多个用户的请求，batch 起来算**，把算力喂满。

但普通 batch 有个致命伤：**每个请求长度不同，快的要等慢的**（padding 浪费），而且 KV cache 每请求独占一大块。

### 6.2 vLLM 的两个核心创新

```
1. Continuous Batching（连续批处理 / 动态批处理）：
   不等人齐、不统一等最慢的。每个解码步结束，
   完成的请求立刻腾出位置、新请求立刻插进来
   → GPU 始终有活干，吞吐大幅提升

2. PagedAttention（分页注意力）：
   把 KV cache 切成固定大小的"页"，像操作系统虚拟内存一样按需分配
   → 显存利用率从 ~60% 提到 ~90%+，碎片几乎为零
```

### 6.3 什么时候用哪个

| 引擎 | 定位 | 适合 |
|---|---|---|
| llama.cpp | 单机轻量，CPU/小显存 | 个人本地、8G 卡跑 7B int4、教学 |
| vLLM | 服务多用户、高吞吐 | 线上 API、大卡集群、prefill/decode 分离 |
| Triton（本仓库 06_triton/） | 自己写高性能 kernel | 学习 SGEMM/attention 内核思路 |

### 6.4 模型侧的"省"：GQA / MQA

- MHA（多头注意力）：每个头一套 K/V → KV cache 最大
- **GQA（分组查询注意力）**：多个 Q 头共享一组 K/V → KV cache 减到 1/4~1/8
- **MQA（多查询注意力）**：所有 Q 头共享一组 K/V → 减到 1/头数

**现代 7B~70B 模型几乎都用 GQA**（如 Llama 3、Qwen2），就是为了让 KV cache 装得下长上下文。KV cache 减下来，decode 带宽压力同步减——和量化是"省带宽"的两条互补路径。

---

## 7. 一张图：LLM 加速全家桶

```
瓶颈           加速手段                原理               章节
─────          ──────                 ──────             ──
decode memory-bound   → KV cache     少重算，显存换算力       §2
prefill compute-bound → FlashAttention 中间量不进显存         §3
decode 带宽/显存       → 量化         字节数省 2~4 倍         §4
解码算力闲置          → 连续批处理     多人共享一块卡          §6
KV cache 太大        → GQA / PagedAttention  按需分页        §6
```

> 所有这些加在一起，就是"为什么现在的 8G 显卡也能流畅跑 7B 对话模型"——不是算力变强了，是**每一步都在省显存、省带宽**。

---

## 8. 本篇与代码/实验的对应

| 概念 | 对应实验/文件 |
|---|---|
| attention 三种写法 | `code/05_llm/`（manual → F.sdpa → FlashAttention） |
| KV cache 有/无对比 | `code/05_llm/`（decode 延迟实测） |
| 量化 | `code/05_llm/`（GGUF 各档位困惑度/速度对比） |
| llama.cpp 部署 7B int4 | `code/05_llm/` + M6 云端实测 |
| vLLM 吞吐测试 | M6 在 AutoDL A100/4090D |
| 手写内核思路 | `code/07_triton/`（Triton 写 SGEMM / FlashAttention，语法见 `docs/07_triton.md`） |

> 下一篇 06_pytorch_gpu.md 回到 PyTorch 应用层——你会看到，前面学的这些 CUDA 概念，在 PyTorch 里不过是 `.to('cuda')` 一行而已，但懂了底层，才懂得它背后发生了什么。
