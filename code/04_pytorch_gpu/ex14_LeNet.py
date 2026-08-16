# =====================================================================
# 04_pytorch_gpu / ex14_LeNet.py
# LeNet-5 在 MNIST 上的 GPU 训练（卷积神经网络全流程）
#
# 运行（有 GPU 优先；无 GPU 自动走 CPU，能跑但慢）：
#   python ex14_LeNet.py
#
# 依赖：torchvision（torch 配套安装）；首次运行会自动下载 MNIST 数据集。
#
# 演示内容：
#   1. 定义 LeNet-5：2 个卷积 + 3 个全连接
#   2. DataLoader 搬数据（每次取一批 .to(device)）
#   3. 完整训练循环：forward -> loss -> backward -> step
#   4. 简单验证集精度
#
# 对应 docs/04_pytorch_gpu.md §2.1 / §2.4（训练循环的标准骨架）
# =====================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("运行设备:", device)


class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)   # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)             # 14x14 -> 10x10
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)               # -> 14x14
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)               # -> 5x5
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def main():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False)

    model = LeNet5().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    EPOCHS = 3
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)   # 搬数据上 GPU
            pred = model(images)
            loss = loss_fn(pred, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += labels.size(0)
            correct += (pred.argmax(1) == labels).sum().item()
            loss_sum += loss.item() * labels.size(0)
        print(f"epoch {epoch}: loss={loss_sum / total:.4f}  acc={correct / total:.4f}")

    # ---- 验证集精度 ----
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            pred = model(images)
            correct += (pred.argmax(1) == labels).sum().item()
            total += labels.size(0)
    print(f"测试集精度: {correct / total:.4f}")


if __name__ == "__main__":
    main()
