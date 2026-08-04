# =====================================================================
# 06_pytorch_gpu / ex11_fitting.py
# 用一个小 MLP 在 GPU 上拟合 y = sin(x)（曲线拟合）
#
# 运行（有 GPU 优先，无 GPU 自动走 CPU）：
#   python ex11_fitting.py
#
# 演示内容：
#   1. 数据、模型、优化器整条链路都搬上 device
#   2. 训练循环三行核心：loss.backward() / optimizer.step() / zero_grad
#   3. 观察 loss 下降，最终打印拟合采样点对比
#
# 对应 docs/06_pytorch_gpu.md §2.2（device 贯穿的训练循环）
# =====================================================================
import math

import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("运行设备:", device)

# ---- 数据：x 在 [-2pi, 2pi] 均匀采样，y = sin(x) + 一点噪声 ----
torch.manual_seed(0)
N = 512
x = torch.linspace(-2 * math.pi, 2 * math.pi, N).unsqueeze(1).to(device)
y = torch.sin(x) + 0.05 * torch.randn_like(x)


# ---- 模型：1 -> 16 -> 16 -> 1 ----
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16), nn.Tanh(),
            nn.Linear(16, 16), nn.Tanh(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


model = MLP().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.MSELoss()

# ---- 训练 ----
EPOCHS = 3000
for epoch in range(1, EPOCHS + 1):
    pred = model(x)
    loss = loss_fn(pred, y)

    opt.zero_grad()
    loss.backward()
    opt.step()

    if epoch % 500 == 0:
        print(f"epoch {epoch:4d}  loss = {loss.item():.6f}")

# ---- 验证：取几个点看拟合效果 ----
with torch.no_grad():
    test_x = torch.tensor(
        [-6.0, -4.0, -2.0, -0.5, 0.0, 1.5, 3.0, 5.0]
    ).unsqueeze(1).to(device)
    pred = model(test_x).squeeze()
    target = torch.sin(test_x).squeeze()
    for i in range(test_x.shape[0]):
        print(f"x={test_x[i, 0]:6.2f}  预测={pred[i]:.4f}  sin={target[i]:.4f}")

print(f"最终 MSE = {loss.item():.6f}")
