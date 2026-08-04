# =====================================================================
# 06_pytorch_gpu / ex20_amp.py
# 自动混合精度 AMP：FP32 vs FP16(autocast + GradScaler)
#
# 运行（建议有 GPU；无 GPU 会自动降级为 CPU 演示并提示）：
#   python ex20_amp.py
#
# 对应 docs/06_pytorch_gpu.md §4（AMP）
#
# 注意：GTX 1080（Pascal, sm_61）硬件不支持 BF16，BF16 需要 Ampere(sm_80)+。
#      所以本示例用 FP16 + GradScaler（Pascal 及以上的通用做法）。
#      在 A100/4090D（sm_80/89）上可把 dtype 换成 torch.bfloat16，省掉 GradScaler。
# =====================================================================
import time

import torch
import torch.nn as nn

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')
print("运行设备:", device)
if not cuda:
    print("警告：无 GPU，AMP 无法加速，仅演示代码结构。远端 Win10（GTX 1080）运行即可看到效果。")

torch.manual_seed(0)
BATCH, IN_F, MID, OUT_F = 512, 512, 1024, 1

model = nn.Sequential(
    nn.Linear(IN_F, MID), nn.ReLU(),
    nn.Linear(MID, MID), nn.ReLU(),
    nn.Linear(MID, OUT_F),
).to(device)
loss_fn = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

x = torch.randn(BATCH, IN_F, device=device)
y = torch.randn(BATCH, OUT_F, device=device)

scaler = torch.amp.GradScaler('cuda') if cuda else None   # FP16 梯度缩放着，防下溢


def train_one_epoch(use_amp: bool):
    t0 = time.perf_counter()
    for _ in range(50):
        opt.zero_grad()
        if use_amp and cuda:
            with torch.amp.autocast('cuda', dtype=torch.float16):   # 精度不敏感算子自动用 FP16
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
    torch.cuda.synchronize() if cuda else None
    return time.perf_counter() - t0


def main():
    t_fp32 = train_one_epoch(use_amp=False)
    print(f"FP32 训练 50 步  : {t_fp32:.3f} s")

    if cuda:
        t_amp = train_one_epoch(use_amp=True)
        print(f"FP16(AMP) 50 步  : {t_amp:.3f} s")
        print(f"加速比           : {t_fp32 / t_amp:.2f}x")
        print("说明：FP16 省一半字节 = 省一半带宽 = 吞吐近似翻倍（docs/06 §4）")
    else:
        print("（跳过 AMP 计时：需要 GPU）")


if __name__ == "__main__":
    main()
