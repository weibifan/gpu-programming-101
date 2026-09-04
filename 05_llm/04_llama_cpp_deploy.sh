#!/usr/bin/env bash
# =====================================================================
# 05_llm / 04_llama_cpp_deploy.sh
# llama.cpp 部署 7B int4（docs/05_llm_acceleration.md §5.2）
#
# 在 AutoDL / Linux（有 CUDA 驱动）上执行：
#   bash 04_llama_cpp_deploy.sh
#
# 对应文档 docs/05_llm_acceleration.md §5；部署结论见 M6。
# =====================================================================
set -euo pipefail

# ---- 1) 拉源码 + 编译（CUDA 后端）----
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build -j

# ---- 2) 下载 int4 GGUF 权重（示例为 Qwen2-7B-Instruct Q4_K_M）----
# 各档位可在 https://huggingface.co/Qwen/Qwen2-7B-Instruct-GGUF 挑选：
#   q4_0（最小/最快）/ q4_K_M（推荐均衡）/ q8_0（更高精度，更大）
MODEL_URL="https://huggingface.co/Qwen/Qwen2-7B-Instruct-GGUF/resolve/main/qwen2-7b-instruct-q4_k_m.gguf"
mkdir -p models && cd models
wget -c "$MODEL_URL"
cd ..

# ---- 3) 跑起来：-ngl 层数放 GPU，其余给 CPU ----
#  -ngl 32 表示把 32 层全 offload 到 GPU；8G 显存建议 -ngl 8~20
./build/bin/llama-cli -m models/qwen2-7b-instruct-q4_k_m.gguf \
    -ngl 32 -p "讲一个 GPU 编程的故事" -n 200

# ---- 4) 测速对比（看 tokens/s）----
# 换不同档位/不同 -ngl 分别跑，对比 decode speed：
#   llama-cli -m ...q8_0.gguf  -ngl 32 -p "..."
#   llama-cli -m ...q4_0.gguf  -ngl 8  -p "..."
#
# 期望：7B FP16=14GB 装不进 8G；INT4≈4GB 装得下 —— 这就是 8G 显存跑 7B 的秘密
