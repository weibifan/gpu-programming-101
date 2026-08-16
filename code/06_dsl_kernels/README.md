# 06_dsl_kernels — 用 Triton 写高性能内核（M5 之后）

目标：把 `docs/03_cuda_advanced.md` 的 tiling、`docs/05_llm_acceleration.md` 的 FlashAttention，用 **Triton**（Python 语法）各写一遍，理解 kernel 编写思路。语法与逐行讲解见 **`docs/06_dsl_kernels.md`**。

| 文件 | 演示内容 | 对应概念 |
|---|---|---|
| `sgemm.py` | Triton 手写 SGEMM：tiling + `tl.dot` + 分组调度，与 torch（cuBLAS）对比 | 04 §3 tiling |
| `flash_attention.py` | Triton 手写 FlashAttention：Q 分块扫 K/V、在线 softmax、因果掩码 | 05 §3 |

## 运行（AutoDL / 有 NVIDIA GPU 的机器）

```bash
pip install triton
python sgemm.py            # 4096³ 矩阵乘，正确性 + 相对 cuBLAS 的速度
python flash_attention.py  # [4,8,2048,128] 因果注意力，与 F.sdpa 对比
```

> 本地无 GPU、纯 CPU 机器装 triton 意义不大（Triton 面向 GPU 生成内核）。放在 05 之后、上云时跑。

## 要点回顾

* **Triton 帮你做了**：tile 划分到线程、共享内存搬运、同步、`tl.dot` 选矩阵乘指令——你只需要描述"数据怎么分块、怎么算"。
* `sgemm.py` 就是 03 篇 `naive → shared → tiling` 的最终形态：一个 Python kernel 就拿到了接近 cuBLAS 的性能（通常 7~9 成）。
* `flash_attention.py` 是 05 篇 §3 的落地：S/P 矩阵从不出共享内存，全局内存读写 O(T²)→O(T)。
* 手写内核的意义是**看懂报告/理解原理**；生产环境直接用 cuBLAS / FlashAttention（`code/08_cuda_libs/`、`F.scaled_dot_product_attention`）。
