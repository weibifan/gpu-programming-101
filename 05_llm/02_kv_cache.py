# =====================================================================
# 05_llm / 02_kv_cache.py
# KV cache 有/无对比：模拟自回归 decode，实测每步延迟
#
# 运行（建议有 GPU）：
#   python 02_kv_cache.py
#
# 对应 docs/05_llm_acceleration.md §2.3
#
# 背景：自回归每步只产出 1 个 token。无 cache 时每步都要重算
#       前面所有 token 的 K、V（计算量随序列长度平方增长）；
#       有 cache 时每步只算新 token 的 K、V，历史直接读缓存（O(T)）。
#
# 本脚本用一个小"投影层"模拟一层 attention：结果一致的前提下，
# 对比"无 cache 重算"与"有 cache 增量算"的总耗时。
# =====================================================================
import time

import torch
import torch.nn as nn

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')
print("运行设备:", device)

torch.manual_seed(0)
D = 128                 # 隐藏维度
TOTAL = 200             # 总共解码的 token 数


class FakeLayer(nn.Module):
    """模拟一层：输入 x -> 投影出 q/k/v，再做一个简化 attention。"""

    def __init__(self):
        super().__init__()
        self.proj_q = nn.Linear(D, D)
        self.proj_k = nn.Linear(D, D)
        self.proj_v = nn.Linear(D, D)

    def forward(self, x, k_cache=None, v_cache=None):
        """x: [T, D]。k_cache/v_cache: [已见, D]，None 表示从头重算。"""
        q = self.proj_q(x[-1:])                  # 只取新 token 的 Q
        k_new = self.proj_k(x[-1:])
        v_new = self.proj_v(x[-1:])

        if k_cache is not None and v_cache is not None:
            k_all = torch.cat([k_cache, k_new], dim=0)   # 增量：只需 cat 新 K/V
            v_all = torch.cat([v_cache, v_new], dim=0)
        else:
            # 无 cache：把整段历史重新投影一遍（重算代价随长度增长）
            k_all = self.proj_k(x)
            v_all = self.proj_v(x)

        scores = q @ k_all.transpose(-2, -1) / (D ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        out = weights @ v_all
        return out, k_all, v_all


def run(use_cache: bool) -> float:
    torch.manual_seed(0)                 # 两次 run 用相同初始权重，否则结果没法对比
    layer = FakeLayer().to(device)
    # 随机 token 向量，第 0 个当 prompt 起点
    all_x = torch.randn(TOTAL, D, device=device)
    k_cache = v_cache = None
    outputs = []
    t0 = time.perf_counter()
    with torch.no_grad():                      # 只测延迟，不建 autograd 图
        for t in range(TOTAL):
            x = all_x[: t + 1]                 # 无 cache 时要用完整历史
            if use_cache:
                out, k_cache, v_cache = layer(all_x[: t + 1], k_cache, v_cache)
            else:
                out, _, _ = layer(x)           # 每步从零重算 K/V
            outputs.append(out)
            if cuda:
                torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return dt, outputs


def main():
    print(f"解码 {TOTAL} 个 token，隐藏维 {D}")
    t_nocache, o_nocache = run(use_cache=False)
    print(f"无 KV cache : {t_nocache:.3f} s（每步重算全部历史 K/V，O(T^2)）")
    t_cache, o_cache = run(use_cache=True)
    print(f"有 KV cache : {t_cache:.3f} s（每步只算新 K/V，O(T)）")

    # 结果一致性：cache 与否的输出应相同
    # （前几步误差大些，取最后 20 步做比较）
    errs = [float((a - b).abs().max()) for a, b in
            zip(o_nocache[-20:], o_cache[-20:])]
    max_err = max(errs)
    print(f"两种写法的输出最大误差: {max_err:.2e}"
          f"  =>  {'一致' if max_err < 1e-4 else '不一致'}")

    speedup = t_nocache / t_cache
    print(f"\nKV cache 总加速比: {speedup:.1f}x")
    print("注：真实实现还会预分配 KV cache 缓冲区（避免每步 cat 拷贝），"
          "提速更明显（docs/05 §2.2）。")


if __name__ == "__main__":
    main()
