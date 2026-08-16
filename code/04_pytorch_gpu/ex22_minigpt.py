"""
ex22_minigpt.py — 极简 decoder-only Transformer（"玩具 GPT"）

目标：把 `docs/附件3_Transformer 架构入门.md` 讲过的组件，用 PyTorch 逐个落成一个能跑、
能训练、能生成文字的最小模型。每一块的注释都标了它对应附件3 的哪一节：

  §1.3 自回归生成      → generate()：预测一个词 → 拼回输入 → 再预测……
  §2.2 token embedding → GPT.transformer.wte（查表，C 程序员的"哈希表"）
  §2.3 位置编码        → GPT.transformer.wpe（可学习版；早期正弦 / nanoChat RoPE 见文档）
  §3   缩放点积注意力   → CausalSelfAttention：Q/K/V 投影 + softmax(QK^T / √d_k)V
  §4   因果 mask       → CausalSelfAttention 里把上三角填成 -inf
  §5.1 pre-norm Block  → Block：x = x + attn(norm(x));  x = x + mlp(norm(x))
  §5.3 归一化          → nn.LayerNorm（有可学习参数版；nanoChat 用无参数 RMSNorm）
  §5.4 FFN             → MLP：d → 4d → d，中间 ReLU
  §5.5 组装            → GPT：wte + wpe + N×Block + LayerNorm + lm_head

设计取舍（先跑通，再花哨）：
  * 权重共享：lm_head 复用 wte（经典省参技巧），两个方向的变换都"过一遍词表向量"
  * 全部 fp32：GTX 1080（sm_61）不支持 bf16。nanoChat 的 ReLU² 才必须 bf16（附件3 §7.3 ③）
  * 配置调小：CPU 也能跑完（几百步 loss 明显下降，末尾能看到生成的文字）

运行：
  python ex22_minigpt.py        # 有 CUDA 走 GPU；没有（本机）走 CPU
"""

import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# 配置（调小：CPU 友好；模型结构 = 附件3 §5.5 那套，只是更小）
# ----------------------------------------------------------------------
@dataclass
class GPTConfig:
    vocab_size: int = 0        # 词表大小，下面按语料自动填
    block_size: int = 32       # 上下文长度 T：每个位置最多看前面 32 个 token
    n_layer: int = 2           # block 数 N
    n_head: int = 4            # 注意力头数 h
    n_embd: int = 64           # embedding 维度 d（= head_dim × n_head）
    batch: int = 8             # 一次看 8 个句子窗口


# ----------------------------------------------------------------------
# §3~§5 的组件
# ----------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    """§3 缩放点积注意力 + §4 因果 mask + 多头（§3.5）。

    公式：Attention(Q,K,V) = softmax(QK^T / √d_k)V
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "d 必须能被头数整除"
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head   # d_k
        # 一次线性层算出 Q、K、V 三份（等价于三个独立投影 W_q/W_k/W_v，省一次搬运）
        self.c_qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)   # §3.5 的 W_o
        # §4.2 因果 mask：下三角为 1、上三角为 0。注册成 buffer（随模型搬设备、不训练）
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_qkv(x).chunk(3, dim=2)                      # [B,T,C] ×3

        # 拆头：每个头只看 head_dim 维（§3.5 的"拆成 h 份"）
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # [B,h,T,d_k]
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # §3.3 第 (1)(2) 步：打分 + 归一化 → [B,h,T,T] 相关度矩阵
        att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(self.head_dim))  # QK^T / √d_k
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))  # §4
        att = F.softmax(att, dim=-1)

        # §3.3 第 (3) 步：按权重加权混合 V
        out = att @ v                                                  # [B,h,T,d_k]
        out = out.transpose(1, 2).contiguous().view(B, T, C)          # 拼回头
        return self.c_proj(out)


class MLP(nn.Module):
    """§5.4 FFN：d → 4d → d，中间一次 ReLU。每个位置"各自思考"。"""

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)))


class Block(nn.Module):
    """§5.1 pre-norm 两段式：先归一化再子层、再残差加。"""

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # 注意力半边：跨位置交换信息
        x = x + self.mlp(self.ln2(x))    # FFN 半边：每个位置各自消化
        return x


class GPT(nn.Module):
    """§5.5 组装：wte + wpe + N×Block + ln_f + lm_head。"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),  # §2.2 查表
                wpe=nn.Embedding(config.block_size, config.n_embd),  # §2.3 位置编码
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # 经典省参技巧：输出投影复用词表权重（训练时它俩被绑在一起更新）
        self.lm_head.weight = self.transformer.wte.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """idx: [B,T] token id；有 targets 时算交叉熵损失（训练），否则只给打分。"""
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)           # 位置 0..T-1
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)       # token + 位置
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)                                        # [B,T,vocab]
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size), targets.view(-1)
            )
        return logits, loss

    def generate(self, idx, max_new_tokens=30):
        """§1.3 自回归生成：预测一个词 → 拼回输入 → 再预测……"""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]                # 只留最近 block_size 个
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]                                   # 只看最后一个位置
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)          # 按概率采样
            idx = torch.cat((idx, idx_next), dim=1)                     # 拼回去，循环
        return idx


# ----------------------------------------------------------------------
# 训练：字符级小语料（不用下载，够演示就行）
# ----------------------------------------------------------------------
def main():
    torch.manual_seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    text = (
        "the quick brown fox jumps over the lazy dog. "
        "the dog sleeps. the fox runs. the fox is quick. "
        "attention lets every token see all others. "
        "predict the next word, again and again."
    )
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}

    config = GPTConfig(vocab_size=len(chars))
    print(f"vocab={config.vocab_size} chars | block_size={config.block_size} "
          f"| n_layer={config.n_layer} n_head={config.n_head} n_embd={config.n_embd}")

    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    def get_batch():
        ix = torch.randint(len(data) - config.block_size, (config.batch,))
        x = torch.stack([data[i : i + config.block_size] for i in ix]).to(device)
        y = torch.stack([data[i + 1 : i + config.block_size + 1] for i in ix]).to(device)
        return x, y

    model = GPT(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量 = {n_params:,}（一个 1 亿参数模型身上的千分之一都不到）\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    # 训练前先看 loss 起点：-ln(1/vocab) ≈ 均匀随机该有的样子
    xb, yb = get_batch()
    _, loss0 = model(xb, yb)

    steps = 400
    t0 = time.time()
    for step in range(1, steps + 1):
        xb, yb = get_batch()
        _, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()          # 附件2：反向传播 = 链式法则机械执行
        optimizer.step()
        if step % 100 == 0:
            print(f"step {step:4d}  loss {loss.item():.4f}")

    print(f"\n训练 {steps} 步耗时 {time.time() - t0:.1f}s（CPU）")
    print(f"loss：{loss0.item():.4f} → {loss.item():.4f}（降得越多说明越学会「猜下一个词」）\n")

    # §1.3 自回归生成：给定开头，让模型自己往下写
    prompt = "the "
    idx = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=30)
    print("生成结果：")
    print("".join(itos[i] for i in out[0].tolist()))


if __name__ == "__main__":
    main()
