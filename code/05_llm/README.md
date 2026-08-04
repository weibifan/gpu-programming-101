# 05_llm — 大模型推理/训练加速（M5~M6）

目标：把 `docs/05_llm_acceleration.md` 的核心优化逐个跑一遍：attention 三种写法、KV cache、量化、llama.cpp 部署。

| 文件 | 演示内容 | 对应文档 |
|---|---|---|
| `01_attention_three_ways.py` | attention 三种写法：manual → `F.scaled_dot_product_attention`（自动走 FlashAttention/memory-efficient），对比正确性与速度 | §3.4 |
| `02_kv_cache.py` | KV cache 有/无对比：模拟 decode，实测每步延迟（O(T²) vs O(T)） | §2.3 |
| `03_quantize_gguf.py` | GGUF 各档位测速（tokens/s + 显存占用），对比 q4_0 / q4_K_M / q8_0 | §4 |
| `04_llama_cpp_deploy.sh` | llama.cpp 编译 + 下载 Q4_K_M GGUF + 部署 7B int4 的完整命令 | §5 |

## 运行

```bash
# 1、2 在远端 Win10（有 GPU）即可：
python 01_attention_three_ways.py
python 02_kv_cache.py

# 3、4 建议在 AutoDL（Linux，有 CUDA 驱动 + 大显存）：
pip install llama-cpp-python
python 03_quantize_gguf.py --model qwen2-7b-instruct-q4_k_m.gguf
bash 04_llama_cpp_deploy.sh
```

## 环境与数据

* `03_quantize_gguf.py` 需要先下载 GGUF 权重（`04_llama_cpp_deploy.sh` 里有示例 URL）。
* `04_llama_cpp_deploy.sh` 里的下载地址是 [Qwen/Qwen2-7B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2-7B-Instruct-GGUF)，可自选 q4_0 / q4_K_M / q8_0 档位做横向对比。
* GTX 1080（sm_61）无 FlashAttention 内核，`F.sdpa` 会走 memory-efficient 版；A100/4090D（sm_80/89）自动走 FlashAttention。

## 要点回顾

* 解码阶段是 100% memory-bound（算术强度 ~1 FLOP/byte），所有加速都是"省字节/省重复搬"：KV cache 省重算、FlashAttention 省中间量、量化直接减字节（`docs/05 §1`）。
* `F.scaled_dot_product_attention` 一行 = 手写 naive 的加速版，是 PyTorch 生态最大的变化之一。
* 8G 显存能跑 7B 的秘密：FP16 权重 14GB 装不下，INT4 约 4GB 装得下（`docs/05 §4.4`）。
