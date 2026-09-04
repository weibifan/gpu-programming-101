# =====================================================================
# 05_llm / 01_attention_three_ways.py
# Attention 三种写法对比：manual -> F.scaled_dot_product_attention
#
# 运行（建议有 GPU，如远端 Win10 / AutoDL；无 GPU 也能跑但看不到加速）：
#   python 01_attention_three_ways.py
#
# 对应 docs/05_llm_acceleration.md §3.4
#
# 要点：
#   写法 1 手写 naive：先算 S=QK^T，再 softmax，再 @V —— 中间量 S 巨大
#   写法 2 F.scaled_dot_product_attention：一行搞定，GPU 支持时自动走
#           FlashAttention / memory-efficient，中间量不进显存
#   写法 3 显式调 fa2/fa3 库：M6 在 A100/4090D 上实测，这里仅注释说明
# =====================================================================
import math
import time

import torch
import torch.nn.functional as F

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')
print("运行设备:", device)

torch.manual_seed(0)
BATCH, HEADS, T, D = 4, 4, 2048, 128          # 序列长 2048，head 维 128


def manual_attention(Q, K, V):
    """写法 1：手写 naive（docs/05 §3.1）。中间量 S、P 都 [T, T] 巨大。"""
    S = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)
    P = torch.softmax(S, dim=-1)
    O = torch.matmul(P, V)
    return O


def sdpa_attention(Q, K, V, causal=False):
    """写法 2：PyTorch 官方接口，自动选最优实现。"""
    if causal:
        return F.scaled_dot_product_attention(Q, K, V, is_causal=True)
    return F.scaled_dot_product_attention(Q, K, V)


def main():
    Q = torch.randn(BATCH, HEADS, T, D, device=device)
    K = torch.randn(BATCH, HEADS, T, D, device=device)
    V = torch.randn(BATCH, HEADS, T, D, device=device)

    # ---- 正确性：两种写法的输出应一致 ----
    O_manual = manual_attention(Q, K, V)
    O_sdpa = sdpa_attention(Q, K, V)
    diff = (O_manual - O_sdpa).abs().max().item()
    print(f"manual vs sdpa 最大误差: {diff:.2e}"
          f"  =>  {'一致' if diff < 1e-3 else '不一致'}")

    # ---- 性能对比（GPU 上测才有意义）----
    def bench(fn, name, reps=10):
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            out = fn()
        if cuda:
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / reps * 1000
        print(f"{name:20s}: {ms:8.3f} ms")
        return ms

    print(f"\n序列长 T={T}（batch {BATCH} x heads {HEADS} x {T} x {D}）")
    m1 = bench(lambda: manual_attention(Q, K, V), "manual(naive)")
    m2 = bench(lambda: sdpa_attention(Q, K, V), "F.sdpa")

    if cuda:
        print(f"\nF.sdpa 相对手写加速比: {m1 / m2:.1f}x")
        print("（GTX 1080 sm_61 无 FlashAttention 内核，会走 memory-efficient 版；")
        print("  A100/4090D sm_80/89 上自动走 FlashAttention，prefill 更快）")

    print("\n说明：写法 2 中加 is_causal=True 可算因果注意力（LLM 自回归用）。")
    print("写法 3（显式 fa2/fa3）见 M6：在 AutoDL A100/4090D 上安装 flash-attn 库实测。")


if __name__ == "__main__":
    main()
