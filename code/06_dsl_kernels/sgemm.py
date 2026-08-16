# =====================================================================
# 06_dsl_kernels / sgemm.py
# 用 Triton 写一个矩阵乘 kernel（SGEMM），对比 torch.matmul
#
# 运行（AutoDL / 装有 triton + NVIDIA GPU 的机器）：
#   pip install triton
#   python sgemm.py
#
# 对应 docs/05_llm_acceleration.md §8（手写内核思路）
#
# 思路（就是 docs/03_cuda_advanced.md §10 的 tiling，用 Python 表达）：
#   * 每个 program 算 C 的一个 BLOCK_M x BLOCK_N 小块
#   * 沿 K 循环，每次把 BLOCK_K 的 A/B 小块搬进"寄存器/共享内存"（由 Triton 自动安排）
#   * tl.dot 自动选矩阵乘指令
#   * GROUP_SIZE_M 分组调度：让同一行的块挨着跑，提高 L2 命中
# =====================================================================
import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    print("缺少 triton，请先安装：pip install triton（需要 NVIDIA GPU 环境）")
    raise SystemExit(1)


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # 分组调度：让 (pid_m, pid_n) 先按行扫描，同一行的块挨在一起跑
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def matmul(a, b):
    M, K = a.shape
    K_, N = b.shape
    assert K == K_
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M'])
                         * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64, GROUP_SIZE_M=8,
    )
    return c


def main():
    if not torch.cuda.is_available():
        print("需要 NVIDIA GPU（本脚本用 Triton 在 GPU 上跑）")
        return
    torch.manual_seed(0)
    M, N, K = 4096, 4096, 4096
    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)

    c_triton = matmul(a, b)
    c_torch = a @ b

    err = (c_triton.float() - c_torch.float()).abs().max().item()
    print(f"{M}x{N}x{K} 矩阵乘，最大误差 {err:.2e}"
          f"  =>  {'一致' if err < 1e-1 else '不一致'}")

    # 测速：与 torch（cuBLAS）对比，手写 Triton 通常能到 cuBLAS 的 7~9 成
    def bench(fn, reps=20):
        for _ in range(3): fn()
        torch.cuda.synchronize()
        import time
        t0 = time.perf_counter()
        for _ in range(reps): fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps * 1000

    t_triton = bench(lambda: matmul(a, b))
    t_torch = bench(lambda: a @ b)
    print(f"Triton SGEMM   : {t_triton:7.2f} ms")
    print(f"torch(cuBLAS)  : {t_torch:7.2f} ms")
    print(f"相对 cuBLAS    : {t_torch / t_triton * 100:.0f}%")


if __name__ == "__main__":
    main()
