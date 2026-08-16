# =====================================================================
# 06_dsl_kernels / flash_attention.py
# 用 Triton 手写一个 FlashAttention kernel（因果注意力，前向）
#
# 运行（AutoDL / 装有 triton + NVIDIA GPU 的机器，head 维需 128 的倍数）：
#   pip install triton
#   python flash_attention.py
#
# 对应 docs/05_llm_acceleration.md §3（FlashAttention）
#
# 核心思想（就是 docs/04 §3 的 tiling 用在 attention 上）：
#   * Q 分块，沿 K/V 一块块扫，S/P 全程不落全局内存
#   * 在线 softmax：只维护 (m_i, l_i) 两个统计量，不需要先把整个 S 存下来
#   * 所以全局内存读写从 O(T²) 降到 O(T)
# =====================================================================
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:
    print("缺少 triton，请先安装：pip install triton（需要 NVIDIA GPU 环境）")
    raise SystemExit(1)

BLOCK_M = 64
BLOCK_N = 64
BLOCK_D = 128
SM_SCALE = 1.0 / (BLOCK_D ** 0.5)


@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qkv_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = Q + qkv_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + qkv_offset + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk
    v_ptrs = V + qkv_offset + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vn
    o_ptrs = Out + qkv_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on

    q = tl.load(q_ptrs)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    m_i = tl.zeros((BLOCK_M,), dtype=tl.float32) - float("inf")
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # 因果掩码：每个 Q 块只看它之前的 K/V 块（start_n < start_m+1）
    for start_n in range(0, (start_m + 1) * BLOCK_M, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(k_ptrs + start_n * stride_kn)
        qk = tl.dot(q, k) * sm_scale
        qk = tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :],
                      qk, float("-inf"))                     # 掩掉未来位置
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])                       # 在线 softmax
        l_ij = tl.sum(p, 1)
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        v = tl.load(v_ptrs + start_n * stride_vk)
        acc = tl.dot(p.to(tl.float16), v, acc)
        m_i = m_ij

    acc = acc / l_i[:, None]
    tl.store(o_ptrs, acc.to(tl.float16))


def flash_attn(q, k, v):
    Z, H, N_CTX, D = q.shape
    out = torch.empty_like(q)
    grid = (triton.cdiv(N_CTX, BLOCK_M), Z * H)
    _fwd_kernel[grid](
        q, k, v, SM_SCALE, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        Z, H, N_CTX,
        BLOCK_M=BLOCK_M, BLOCK_D=D, BLOCK_N=BLOCK_N,
    )
    return out


def main():
    if not torch.cuda.is_available():
        print("需要 NVIDIA GPU（本脚本用 Triton 在 GPU 上跑）")
        return
    torch.manual_seed(0)
    Z, H, N_CTX, D = 4, 8, 2048, 128
    q = torch.randn(Z, H, N_CTX, D, device='cuda', dtype=torch.float16)
    k = torch.randn(Z, H, N_CTX, D, device='cuda', dtype=torch.float16)
    v = torch.randn(Z, H, N_CTX, D, device='cuda', dtype=torch.float16)

    o_triton = flash_attn(q, k, v)
    o_ref = F.scaled_dot_product_attention(
        q, k, v, is_causal=True).float()
    err = (o_triton.float() - o_ref).abs().max().item()
    print(f"输入 [4, 8, 2048, 128]，最大误差 {err:.2e}"
          f"  =>  {'一致' if err < 1e-2 else '不一致'}")

    import time
    def bench(fn, reps=10):
        for _ in range(3): fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps): fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps * 1000

    t_triton = bench(lambda: flash_attn(q, k, v))
    t_torch = bench(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))
    print(f"Triton FA     : {t_triton:7.2f} ms")
    print(f"F.sdpa        : {t_torch:7.2f} ms")
    print(f"相对 sdpa     : {t_torch / t_triton * 100:.0f}%")
    print("（说明：Triton FA 已贴近官方实现；若机器有 FlashAttention 内核，"
          "F.sdpa 会更快）")


if __name__ == "__main__":
    main()
