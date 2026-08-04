# =====================================================================
# 06_pytorch_gpu / ex21_torch_compile.py
# torch.compile 一键编译优化：把 forward 当整体，融合算子
#
# 运行（建议有 GPU；无 GPU 也能跑，但 inductor 在 CPU 上收益有限）：
#   python ex21_torch_compile.py
#
# 对应 docs/06_pytorch_gpu.md §5
#
# 注意：首次编译要几十秒（建 graph / 生成内核），之后的调用才快；
#      输入形状需固定，否则会重新编译。
# =====================================================================
import time

import torch
import torch.nn as nn

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')
print("运行设备:", device)

torch.manual_seed(0)
BATCH, IN_F, HID, OUT_F = 256, 512, 1024, 128

model = nn.Sequential(
    nn.Linear(IN_F, HID), nn.ReLU(),
    nn.Linear(HID, HID), nn.ReLU(),
    nn.Linear(HID, OUT_F),
).to(device)
loss_fn = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

x = torch.randn(BATCH, IN_F, device=device)
y = torch.randn(BATCH, OUT_F, device=device)


def run(model, reps=50):
    t0 = time.perf_counter()
    for _ in range(reps):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    if cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps


def main():
    print("先测 eager 模式（普通 PyTorch）...")
    t_eager = run(model)
    print(f"eager 每步耗时     : {t_eager * 1000:.3f} ms")

    print("开始 torch.compile（首次编译约需几十秒）...")
    try:
        compiled = torch.compile(model)
        # 先跑一步触发编译
        run(compiled, reps=1)
        t_compiled = run(compiled, reps=50)
        print(f"compiled 每步耗时  : {t_compiled * 1000:.3f} ms")
        print(f"加速比             : {t_eager / t_compiled:.2f}x")
    except Exception as e:
        print(f"torch.compile 失败（{type(e).__name__}: {e}）")
        print("提示：torch.compile 的 inductor 后端在部分环境会退回解释执行；")
        print("      可用 torch.compile(model, backend='eager') 关闭以作对照。")


if __name__ == "__main__":
    main()
